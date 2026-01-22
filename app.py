import json
import base64
import time
from datetime import datetime, date

import httpx
import streamlit as st
from openai import OpenAI

# =========================
# Oracle Fusion config
# =========================
FUSION_REST_VERSION = "11.13.18.05"  # change if your pod uses a different resource version

def build_basic_auth(username: str, password: str) -> str:
    token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("utf-8")
    return f"Basic {token}"

def build_fusion_url(base_url: str, endpoint: str) -> str:
    return f"{base_url.rstrip('/')}/fscmRestApi/resources/{FUSION_REST_VERSION}/{endpoint}"

def handle_oracle_error(e: Exception) -> str:
    if isinstance(e, httpx.HTTPStatusError):
        code = e.response.status_code
        if code == 401:
            return json.dumps({"error": "Oracle auth failed (401). Check username/password."})
        if code == 403:
            return json.dumps({"error": "Oracle permission denied (403). Check Fusion roles."})
        if code == 404:
            return json.dumps({"error": "Oracle endpoint not found (404). Check base URL/version/endpoint."})
        if code == 429:
            return json.dumps({"error": "Oracle rate limit (429). Retry later."})
        try:
            detail = e.response.json()
        except Exception:
            detail = e.response.text
        return json.dumps({"error": f"Oracle API error ({code})", "detail": detail})
    if isinstance(e, httpx.TimeoutException):
        return json.dumps({"error": "Oracle request timed out."})
    if isinstance(e, httpx.ConnectError):
        return json.dumps({"error": "Oracle connection failed. Check base URL / network."})
    return json.dumps({"error": f"Unexpected error: {type(e).__name__}: {str(e)}"})

# =========================
# Session state init
# =========================
def init_state():
    defaults = {
        "openai_key": "",
        "oracle_base_url": "",
        "oracle_username": "",
        "oracle_password": "",
        "oracle_auth_header": "",
        "oracle_validated": False,
        "oracle_status_msg": "Not connected",
        "oracle_creds_dirty": True,
        "last_validate_sig": "",
        "chat": [],  # list of {role, content}
        "tool_traces": [],  # list of {turn_index, tool_log}
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# Mark creds dirty when user edits any Oracle field
def mark_oracle_dirty():
    st.session_state.oracle_creds_dirty = True
    st.session_state.oracle_validated = False
    st.session_state.oracle_auth_header = ""
    st.session_state.oracle_status_msg = "Not connected"

# =========================
# Oracle validate + request using stored auth
# =========================
def oracle_validate_if_needed():
    base_url = (st.session_state.oracle_base_url or "").strip()
    username = (st.session_state.oracle_username or "").strip()
    password = (st.session_state.oracle_password or "")

    # If missing, don't try validating
    if not base_url or not username or not password:
        st.session_state.oracle_validated = False
        st.session_state.oracle_auth_header = ""
        st.session_state.oracle_status_msg = "Not connected (missing fields)"
        return

    # Avoid validating repeatedly on every rerun if nothing changed
    sig = f"{base_url}|{username}|{len(password)}"
    if (not st.session_state.oracle_creds_dirty) and st.session_state.oracle_validated and st.session_state.last_validate_sig == sig:
        return

    # Validate once per change
    st.session_state.last_validate_sig = sig
    st.session_state.oracle_status_msg = "Validating..."

    auth_header = build_basic_auth(username, password)
    url = build_fusion_url(base_url, "receivablesInvoices")

    try:
        with httpx.Client(timeout=30.0, verify=False) as client:
            r = client.get(
                url,
                headers={"Authorization": auth_header, "Content-Type": "application/json"},
                params={"limit": 1},
            )
            r.raise_for_status()

        st.session_state.oracle_auth_header = auth_header
        st.session_state.oracle_validated = True
        st.session_state.oracle_creds_dirty = False
        st.session_state.oracle_status_msg = "Connected ✅"

    except Exception as e:
        st.session_state.oracle_validated = False
        st.session_state.oracle_auth_header = ""
        st.session_state.oracle_creds_dirty = True
        # keep message short; details will show in tool outputs
        st.session_state.oracle_status_msg = "Connection failed ❌"

def oracle_get(endpoint: str, params: dict | None = None) -> dict:
    base_url = (st.session_state.oracle_base_url or "").strip()
    auth_header = st.session_state.oracle_auth_header

    if not st.session_state.oracle_validated or not base_url or not auth_header:
        raise RuntimeError(json.dumps({"error": "Oracle not connected. Fill creds; wait for Connected ✅."}))

    url = build_fusion_url(base_url, endpoint)
    with httpx.Client(timeout=60.0, verify=False) as client:
        resp = client.get(
            url,
            headers={"Authorization": auth_header, "Content-Type": "application/json"},
            params=params or {},
        )
        resp.raise_for_status()
        return resp.json()

# =========================
# Tools (model calls these)
# =========================
def tool_list_invoices(args: dict) -> str:
    customer_account_id = args.get("customer_account_id")
    invoice_number = args.get("invoice_number")
    status = args.get("status")
    limit = int(args.get("limit", 25))
    offset = int(args.get("offset", 0))

    params = {"limit": limit, "offset": offset}
    filters = []
    if customer_account_id:
        filters.append(f"CustomerAccountId={customer_account_id}")
    if invoice_number:
        filters.append(f"TransactionNumber={invoice_number}")
    if status:
        filters.append(f"Status={status}")
    if filters:
        params["q"] = ";".join(filters)

    try:
        data = oracle_get("receivablesInvoices", params)
        items = data.get("items", [])
        invoices = [{
            "invoice_number": inv.get("TransactionNumber"),
            "customer_account_id": inv.get("CustomerAccountId"),
            "customer_name": inv.get("BillToCustomerName"),
            "invoice_date": inv.get("TransactionDate"),
            "due_date": inv.get("DueDate"),
            "amount": inv.get("EnteredAmount"),
            "balance_due": inv.get("BalanceDue"),
            "currency": inv.get("EnteredCurrencyCode"),
            "status": inv.get("Status"),
            "business_unit": inv.get("BusinessUnit"),
        } for inv in items]

        return json.dumps({
            "count": len(invoices),
            "invoices": invoices,
            "has_more": data.get("hasMore", False),
            "offset": offset,
            "limit": limit,
        }, indent=2)
    except Exception as e:
        if isinstance(e, RuntimeError):
            return str(e)
        return handle_oracle_error(e)

def tool_list_receipts(args: dict) -> str:
    customer_account_id = args.get("customer_account_id")
    receipt_number = args.get("receipt_number")
    limit = int(args.get("limit", 25))
    offset = int(args.get("offset", 0))

    params = {"limit": limit, "offset": offset}
    filters = []
    if customer_account_id:
        filters.append(f"CustomerAccountId={customer_account_id}")
    if receipt_number:
        filters.append(f"ReceiptNumber={receipt_number}")
    if filters:
        params["q"] = ";".join(filters)

    try:
        data = oracle_get("standardReceipts", params)
        items = data.get("items", [])
        receipts = [{
            "receipt_number": r.get("ReceiptNumber"),
            "customer_account_id": r.get("CustomerAccountId"),
            "customer_name": r.get("CustomerName"),
            "receipt_date": r.get("ReceiptDate"),
            "amount": r.get("Amount"),
            "currency": r.get("CurrencyCode"),
            "status": r.get("Status"),
            "payment_method": r.get("PaymentMethod"),
            "business_unit": r.get("BusinessUnit"),
        } for r in items]

        return json.dumps({
            "count": len(receipts),
            "receipts": receipts,
            "has_more": data.get("hasMore", False),
            "offset": offset,
            "limit": limit,
        }, indent=2)
    except Exception as e:
        if isinstance(e, RuntimeError):
            return str(e)
        return handle_oracle_error(e)

def tool_customer_summary(args: dict) -> str:
    customer_account_id = (args.get("customer_account_id") or "").strip()
    if not customer_account_id:
        return json.dumps({"error": "customer_account_id is required."})

    try:
        inv_data = oracle_get("receivablesInvoices", {"q": f"CustomerAccountId={customer_account_id}", "limit": 500})
        rcpt_data = oracle_get("standardReceipts", {"q": f"CustomerAccountId={customer_account_id}", "limit": 500})

        invoices = inv_data.get("items", [])
        receipts = rcpt_data.get("items", [])

        total_invoiced = sum(inv.get("EnteredAmount") or 0 for inv in invoices)
        total_balance_due = sum(inv.get("BalanceDue") or 0 for inv in invoices)
        total_paid = sum(rcpt.get("Amount") or 0 for rcpt in receipts)

        customer_name = None
        if invoices:
            customer_name = invoices[0].get("BillToCustomerName")
        elif receipts:
            customer_name = receipts[0].get("CustomerName")

        return json.dumps({
            "customer": {"customer_account_id": customer_account_id, "customer_name": customer_name},
            "summary": {
                "total_invoiced": round(total_invoiced, 2),
                "total_paid": round(total_paid, 2),
                "outstanding_balance": round(total_balance_due, 2),
                "invoice_count": len(invoices),
                "receipt_count": len(receipts),
            }
        }, indent=2)
    except Exception as e:
        if isinstance(e, RuntimeError):
            return str(e)
        return handle_oracle_error(e)

def tool_aging_summary(args: dict) -> str:
    customer_account_id = args.get("customer_account_id")
    limit = int(args.get("limit", 25))
    offset = int(args.get("offset", 0))

    params = {"limit": limit, "offset": offset}
    if customer_account_id:
        params["q"] = f"CustomerAccountId={customer_account_id}"

    try:
        data = oracle_get("receivablesInvoices", params)
        invoices = data.get("items", [])

        today = date.today()
        buckets = {
            "current": {"count": 0, "amount": 0.0},
            "1_30_days": {"count": 0, "amount": 0.0},
            "31_60_days": {"count": 0, "amount": 0.0},
            "61_90_days": {"count": 0, "amount": 0.0},
            "over_90_days": {"count": 0, "amount": 0.0},
        }

        for inv in invoices:
            bal = inv.get("BalanceDue") or 0
            if bal <= 0:
                continue
            due_str = inv.get("DueDate")
            if not due_str:
                continue
            try:
                due_dt = datetime.fromisoformat(due_str.replace("Z", "+00:00")).date()
            except Exception:
                continue

            days = (today - due_dt).days
            if days <= 0:
                b = "current"
            elif days <= 30:
                b = "1_30_days"
            elif days <= 60:
                b = "31_60_days"
            elif days <= 90:
                b = "61_90_days"
            else:
                b = "over_90_days"

            buckets[b]["count"] += 1
            buckets[b]["amount"] += float(bal)

        return json.dumps({
            "aging_buckets": {k: {"count": v["count"], "amount": round(v["amount"], 2)} for k, v in buckets.items()},
            "total_outstanding": round(sum(v["amount"] for v in buckets.values()), 2),
        }, indent=2)
    except Exception as e:
        if isinstance(e, RuntimeError):
            return str(e)
        return handle_oracle_error(e)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "oracle_ar_list_invoices",
            "description": "List receivables invoices. Optional filters: customer_account_id, invoice_number, status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_account_id": {"type": "string"},
                    "invoice_number": {"type": "string"},
                    "status": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 25},
                    "offset": {"type": "integer", "minimum": 0, "default": 0},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "oracle_ar_list_receipts",
            "description": "List standard receipts. Optional filters: customer_account_id, receipt_number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_account_id": {"type": "string"},
                    "receipt_number": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 25},
                    "offset": {"type": "integer", "minimum": 0, "default": 0},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "oracle_ar_get_customer_summary",
            "description": "Get AR summary for a customer account.",
            "parameters": {
                "type": "object",
                "properties": {"customer_account_id": {"type": "string"}},
                "required": ["customer_account_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "oracle_ar_get_aging_summary",
            "description": "Get aging buckets for open invoices. customer_account_id optional.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_account_id": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 25},
                    "offset": {"type": "integer", "minimum": 0, "default": 0},
                },
            },
        },
    },
]

TOOL_DISPATCH = {
    "oracle_ar_list_invoices": tool_list_invoices,
    "oracle_ar_list_receipts": tool_list_receipts,
    "oracle_ar_get_customer_summary": tool_customer_summary,
    "oracle_ar_get_aging_summary": tool_aging_summary,
}

def run_agent(messages, openai_key: str):
    client = OpenAI(api_key=openai_key)

    tool_log = []

    for _ in range(4):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            return (msg.content or "").strip(), tool_log

        messages.append(
            {"role": "assistant", "content": msg.content, "tool_calls": [tc.model_dump() for tc in msg.tool_calls]}
        )

        for tc in msg.tool_calls:
            name = tc.function.name
            raw_args = tc.function.arguments or "{}"
            try:
                args = json.loads(raw_args)
            except Exception:
                args = {}

            tool_log.append({"tool": name, "arguments": args})
            fn = TOOL_DISPATCH.get(name)
            result = fn(args) if fn else json.dumps({"error": f"Unknown tool: {name}"})

            messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})

    return "Stopped: too many tool calls.", tool_log


# =========================
# UI
# =========================
st.set_page_config(page_title="Oracle Receivables AI Agent", page_icon="🧾", layout="wide")
st.title("Oracle Receivables AI Agent")

with st.sidebar:
    st.header("Inputs")

    st.subheader("OpenAI")
    st.session_state.openai_key = st.text_input("OpenAI API Key", type="password", value=st.session_state.openai_key)

    st.divider()

    st.subheader("Oracle Fusion")
    st.session_state.oracle_base_url = st.text_input(
        "Base URL",
        value=st.session_state.oracle_base_url,
        placeholder="https://<pod>.oraclecloud.com",
        on_change=mark_oracle_dirty,
    )
    st.session_state.oracle_username = st.text_input(
        "Username", value=st.session_state.oracle_username, on_change=mark_oracle_dirty
    )
    st.session_state.oracle_password = st.text_input(
        "Password", type="password", value=st.session_state.oracle_password, on_change=mark_oracle_dirty
    )

    # Auto-validate (no button)
    oracle_validate_if_needed()
    st.info(st.session_state.oracle_status_msg)

    if st.button("Clear all"):
        for k in ["openai_key", "oracle_base_url", "oracle_username", "oracle_password",
                  "oracle_auth_header", "oracle_validated", "oracle_status_msg", "oracle_creds_dirty",
                  "last_validate_sig"]:
            st.session_state[k] = "" if "key" in k or "url" in k or "user" in k or "pass" in k or "header" in k else False
        st.session_state.oracle_status_msg = "Not connected"
        st.session_state.chat = []
        st.session_state.tool_traces = []
        st.rerun()

# Render chat history
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input (keeps prior questions)
user_text = st.chat_input("Ask: invoices, receipts, customer summary, aging...")

if user_text:
    ok = (st.session_state.openai_key or "").strip()
    if not ok:
        st.error("Paste your OpenAI API key in the sidebar.")
    else:
        # Add user message
        st.session_state.chat.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.write(user_text)

        # Build agent messages from chat
        agent_messages = [
            {
                "role": "system",
                "content": (
                    "You are an Oracle Fusion AR assistant. "
                    "Use tools to query Oracle Fusion AR when needed. "
                    "Do NOT ask for Oracle credentials; they are already provided in the sidebar. "
                    "If Oracle is not connected, tell the user to fix Base URL/Username/Password and wait for Connected ✅."
                ),
            }
        ] + st.session_state.chat

        answer, tool_log = run_agent(agent_messages, ok)

        # Show assistant response
        st.session_state.chat.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.write(answer)
            if tool_log:
                with st.expander("Tool calls"):
                    st.json(tool_log)
