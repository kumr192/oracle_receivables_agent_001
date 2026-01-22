import json
import base64
from datetime import datetime, date
from typing import Optional, Dict, Any

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
    # base_url example: https://xyz.fa.oraclecloud.com  (no trailing slash required)
    return f"{base_url.rstrip('/')}/fscmRestApi/resources/{FUSION_REST_VERSION}/{endpoint}"

def get_oracle_creds_from_ui():
    base_url = (st.session_state.get("oracle_base_url") or "").strip()
    username = (st.session_state.get("oracle_username") or "").strip()
    password = (st.session_state.get("oracle_password") or "")
    if not base_url or not username or not password:
        return None, None, None, json.dumps({
            "error": "Missing Oracle inputs. Fill Base URL, Username, Password in the sidebar."
        })
    return base_url, username, password, None

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

def oracle_get(endpoint: str, params: Optional[dict] = None) -> dict:
    base_url, username, password, err = get_oracle_creds_from_ui()
    if err:
        raise RuntimeError(err)

    url = build_fusion_url(base_url, endpoint)
    auth_header = build_basic_auth(username, password)

    with httpx.Client(timeout=60.0, verify=False) as client:
        resp = client.get(
            url,
            headers={"Authorization": auth_header, "Content-Type": "application/json"},
            params=params or {},
        )
        resp.raise_for_status()
        return resp.json()

# =========================
# Tool implementations (called by agent)
# =========================
def tool_test_connection(_: dict) -> str:
    try:
        _ = oracle_get("receivablesInvoices", {"limit": 1})
        base_url = (st.session_state.get("oracle_base_url") or "").strip()
        return json.dumps({
            "status": "connected",
            "base_url": base_url,
            "message": "Successfully connected to Oracle Fusion AR REST API."
        }, indent=2)
    except Exception as e:
        # If oracle_get threw a RuntimeError with JSON string, surface it
        if isinstance(e, RuntimeError):
            return str(e)
        return handle_oracle_error(e)

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

        invoices = []
        for inv in items:
            invoices.append({
                "invoice_id": inv.get("CustomerTransactionId"),
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
            })

        return json.dumps({
            "endpoint_called": "receivablesInvoices",
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

        receipts = []
        for r in items:
            receipts.append({
                "receipt_id": r.get("CashReceiptId"),
                "receipt_number": r.get("ReceiptNumber"),
                "customer_account_id": r.get("CustomerAccountId"),
                "customer_name": r.get("CustomerName"),
                "receipt_date": r.get("ReceiptDate"),
                "amount": r.get("Amount"),
                "currency": r.get("CurrencyCode"),
                "status": r.get("Status"),
                "payment_method": r.get("PaymentMethod"),
                "business_unit": r.get("BusinessUnit"),
            })

        return json.dumps({
            "endpoint_called": "standardReceipts",
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
            "endpoint_called": "receivablesInvoices",
            "aging_buckets": {k: {"count": v["count"], "amount": round(v["amount"], 2)} for k, v in buckets.items()},
            "total_outstanding": round(sum(v["amount"] for v in buckets.values()), 2),
        }, indent=2)
    except Exception as e:
        if isinstance(e, RuntimeError):
            return str(e)
        return handle_oracle_error(e)


# =========================
# OpenAI tool schema + dispatch
# =========================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "oracle_ar_test_connection",
            "description": "Test Oracle Fusion connection using sidebar base URL/username/password.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
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
    "oracle_ar_test_connection": tool_test_connection,
    "oracle_ar_list_invoices": tool_list_invoices,
    "oracle_ar_list_receipts": tool_list_receipts,
    "oracle_ar_get_customer_summary": tool_customer_summary,
    "oracle_ar_get_aging_summary": tool_aging_summary,
}

def run_agent(user_text: str, openai_key: str):
    client = OpenAI(api_key=openai_key)

    messages = [
        {
            "role": "system",
            "content": (
                "You are an Oracle Fusion AR assistant. "
                "When the user asks for invoices, receipts, customer summary, or aging, call the correct tool. "
                "Do not ask the user for passwords. Credentials are in the sidebar. "
                "If Oracle creds are missing, instruct the user to fill the sidebar fields."
            ),
        },
        {"role": "user", "content": user_text},
    ]

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

        # record assistant tool calls
        messages.append(
            {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
            }
        )

        for tc in msg.tool_calls:
            tool_name = tc.function.name
            raw_args = tc.function.arguments or "{}"
            try:
                args = json.loads(raw_args)
            except Exception:
                args = {}

            tool_log.append({"tool": tool_name, "arguments": args})

            fn = TOOL_DISPATCH.get(tool_name)
            result = fn(args) if fn else json.dumps({"error": f"Unknown tool: {tool_name}"})

            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": str(result)}
            )

    return "Stopped: too many tool calls.", tool_log


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Oracle Fusion AR Agent (BYOK)", page_icon="🧾", layout="wide")
st.title("Oracle Fusion AR Agent (BYOK)")

# session defaults
for k in ["openai_key", "oracle_base_url", "oracle_username", "oracle_password"]:
    if k not in st.session_state:
        st.session_state[k] = ""

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
    )
    st.session_state.oracle_username = st.text_input("Username", value=st.session_state.oracle_username)
    st.session_state.oracle_password = st.text_input("Password", type="password", value=st.session_state.oracle_password)

    if st.button("Clear all"):
        for k in ["openai_key", "oracle_base_url", "oracle_username", "oracle_password"]:
            st.session_state[k] = ""
        st.rerun()

prompt = st.text_input(
    "Ask something",
    placeholder="Example: List open invoices for customer account 12345 (limit 10)",
)

col1, col2 = st.columns(2)
with col1:
    run_clicked = st.button("Run")
with col2:
    test_clicked = st.button("Test Oracle Connection")

if test_clicked:
    st.subheader("Test Result")
    st.code(tool_test_connection({}), language="json")

if run_clicked:
    ok = (st.session_state.openai_key or "").strip()
    if not ok:
        st.error("Paste your OpenAI API key in the sidebar.")
    elif not prompt.strip():
        st.warning("Type a question first.")
    else:
        answer, tool_log = run_agent(prompt.strip(), ok)

        if tool_log:
            st.subheader("Tool calls")
            st.json(tool_log)

        st.subheader("Answer")
        st.write(answer)
