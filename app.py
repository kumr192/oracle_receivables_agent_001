import json
import base64
from datetime import datetime, date

import httpx
import streamlit as st
from openai import OpenAI

FUSION_REST_VERSION = "11.13.18.05"


# =========================
# Helpers
# =========================
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
# Session State
# =========================
def init_state():
    defaults = {
        "openai_key": "",
        "oracle_base_url": "",
        "oracle_username": "",
        "oracle_password": "",
        "oracle_auth_header": "",
        "oracle_validated": False,
        "oracle_creds_dirty": True,
        "last_validate_sig": "",
        "oracle_status": "",
        "tls_verify": True,
        "chat": [],
        "debug_mode": True,  # NEW: Enable debug by default
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


def mark_oracle_dirty():
    st.session_state.oracle_creds_dirty = True
    st.session_state.oracle_validated = False
    st.session_state.oracle_auth_header = ""
    st.session_state.oracle_status = ""


def oracle_validate_if_needed():
    base_url = (st.session_state.oracle_base_url or "").strip()
    username = (st.session_state.oracle_username or "").strip()
    password = st.session_state.oracle_password or ""

    if not base_url or not username or not password:
        st.session_state.oracle_validated = False
        st.session_state.oracle_auth_header = ""
        st.session_state.oracle_status = ""
        return

    sig = f"{base_url}|{username}|{len(password)}|verify={st.session_state.tls_verify}"
    if (
        (not st.session_state.oracle_creds_dirty)
        and st.session_state.oracle_validated
        and st.session_state.last_validate_sig == sig
    ):
        return

    st.session_state.last_validate_sig = sig
    auth_header = build_basic_auth(username, password)

    # Use receivablesInvoices for connectivity check
    url = build_fusion_url(base_url, "receivablesInvoices")

    try:
        with httpx.Client(timeout=30.0, verify=st.session_state.tls_verify) as client:
            r = client.get(
                url,
                headers={
                    "Authorization": auth_header,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                params={"limit": 1},
            )
            r.raise_for_status()

        st.session_state.oracle_auth_header = auth_header
        st.session_state.oracle_validated = True
        st.session_state.oracle_creds_dirty = False
        st.session_state.oracle_status = "Connected ✅"
    except Exception as ex:
        st.session_state.oracle_validated = False
        st.session_state.oracle_auth_header = ""
        st.session_state.oracle_creds_dirty = True
        st.session_state.oracle_status = f"Connection failed ❌: {type(ex).__name__}"


def oracle_get(endpoint: str, params: dict | None = None) -> dict:
    base_url = (st.session_state.oracle_base_url or "").strip()
    auth_header = st.session_state.oracle_auth_header

    if not st.session_state.oracle_validated or not base_url or not auth_header:
        raise RuntimeError(
            json.dumps({"error": "Oracle not connected. Fill creds and ensure status shows Connected ✅."})
        )

    url = build_fusion_url(base_url, endpoint)
    
    # Debug logging
    if st.session_state.get("debug_mode"):
        st.session_state.setdefault("debug_logs", []).append({
            "action": "oracle_get",
            "url": url,
            "params": params,
        })
    
    with httpx.Client(timeout=60.0, verify=st.session_state.tls_verify) as client:
        resp = client.get(
            url,
            headers={
                "Authorization": auth_header,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            params=params or {},
        )
        resp.raise_for_status()
        return resp.json()


# =========================
# Tools - SIMPLIFIED & MORE ROBUST
# =========================

def tool_list_invoices(args: dict) -> str:
    """
    List invoices. This is the MOST RELIABLE endpoint.
    Start here if you need customer info - invoices contain customer names.
    """
    customer_account_id = (args.get("customer_account_id") or "").strip() or None
    customer_name = (args.get("customer_name") or "").strip() or None
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
        
        # Extract unique customers from invoice data
        customers_found = {}
        invoices = []
        
        for inv in items:
            cust_id = inv.get("CustomerAccountId")
            cust_name = inv.get("BillToCustomerName")
            
            # Track unique customers
            if cust_id and cust_id not in customers_found:
                customers_found[cust_id] = cust_name
            
            # Filter by customer_name if provided (case-insensitive partial match)
            if customer_name:
                bill_to = (cust_name or "").lower()
                if customer_name.lower() not in bill_to:
                    continue
            
            invoices.append({
                "invoice_number": inv.get("TransactionNumber"),
                "customer_account_id": cust_id,
                "customer_name": cust_name,
                "invoice_date": inv.get("TransactionDate"),
                "due_date": inv.get("DueDate"),
                "amount": inv.get("EnteredAmount"),
                "balance_due": inv.get("BalanceDue"),
                "currency": inv.get("EnteredCurrencyCode"),
                "status": inv.get("Status"),
                "business_unit": inv.get("BusinessUnit"),
            })

        return json.dumps({
            "count": len(invoices),
            "invoices": invoices,
            "customers_in_results": [
                {"customer_account_id": k, "customer_name": v} 
                for k, v in customers_found.items()
            ],
            "has_more": data.get("hasMore", False),
            "offset": offset,
            "limit": limit,
        }, indent=2)
        
    except Exception as e:
        if isinstance(e, RuntimeError):
            return str(e)
        return handle_oracle_error(e)


def tool_list_receipts(args: dict) -> str:
    """List receipts (payments)."""
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
        receipts = [
            {
                "receipt_number": r.get("ReceiptNumber"),
                "customer_account_id": r.get("CustomerAccountId"),
                "customer_name": r.get("CustomerName"),
                "receipt_date": r.get("ReceiptDate"),
                "amount": r.get("Amount"),
                "currency": r.get("CurrencyCode"),
                "status": r.get("Status"),
                "payment_method": r.get("PaymentMethod"),
                "business_unit": r.get("BusinessUnit"),
            }
            for r in items
        ]

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
    """Get AR summary for a customer account id."""
    customer_account_id = (args.get("customer_account_id") or "").strip()
    if not customer_account_id:
        return json.dumps({"error": "customer_account_id is required."})

    try:
        inv_data = oracle_get(
            "receivablesInvoices", {"q": f"CustomerAccountId={customer_account_id}", "limit": 500}
        )
        rcpt_data = oracle_get(
            "standardReceipts", {"q": f"CustomerAccountId={customer_account_id}", "limit": 500}
        )

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
            },
        }, indent=2)
    except Exception as e:
        if isinstance(e, RuntimeError):
            return str(e)
        return handle_oracle_error(e)


def tool_aging_summary(args: dict) -> str:
    """Get aging buckets for open invoices."""
    customer_account_id = args.get("customer_account_id")
    limit = int(args.get("limit", 100))
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


def tool_discover_customers(args: dict) -> str:
    """
    Discover customers by fetching invoices and extracting unique customer info.
    This is MORE RELIABLE than the LOV finder which may not exist in all pods.
    """
    name_filter = (args.get("name_contains") or "").strip().lower()
    limit = int(args.get("limit", 100))
    
    try:
        # Fetch a batch of invoices to extract customer data
        data = oracle_get("receivablesInvoices", {"limit": limit})
        items = data.get("items", [])
        
        customers = {}
        for inv in items:
            cust_id = inv.get("CustomerAccountId")
            cust_name = inv.get("BillToCustomerName")
            acct_num = inv.get("CustomerAccountNumber")
            
            if not cust_id:
                continue
                
            # Apply name filter if provided
            if name_filter and name_filter not in (cust_name or "").lower():
                continue
            
            if cust_id not in customers:
                customers[cust_id] = {
                    "customer_account_id": cust_id,
                    "customer_name": cust_name,
                    "account_number": acct_num,
                    "invoice_count": 0,
                }
            customers[cust_id]["invoice_count"] += 1
        
        result = list(customers.values())
        result.sort(key=lambda x: x.get("customer_name") or "")
        
        return json.dumps({
            "method": "extracted_from_invoices",
            "name_filter": name_filter or None,
            "count": len(result),
            "customers": result,
        }, indent=2)
        
    except Exception as e:
        if isinstance(e, RuntimeError):
            return str(e)
        return handle_oracle_error(e)


# Simplified tool definitions - removed the problematic LOV-based customer search
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "oracle_ar_discover_customers",
            "description": "Discover customers by extracting from invoice data. Use name_contains for filtering. This is the primary way to find customers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name_contains": {"type": "string", "description": "Optional filter: customer name must contain this string (case-insensitive)"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 100},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "oracle_ar_list_invoices",
            "description": "List invoices. Can filter by customer_account_id, customer_name (partial match), invoice_number, or status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_account_id": {"type": "string"},
                    "customer_name": {"type": "string", "description": "Partial match on customer name"},
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
            "description": "List receipts (payments).",
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
            "description": "Get AR summary (total invoiced, paid, outstanding) for a specific customer_account_id.",
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
            "description": "Get aging buckets (current, 1-30, 31-60, 61-90, 90+) for open invoices.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_account_id": {"type": "string", "description": "Optional: filter to specific customer"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 100},
                    "offset": {"type": "integer", "minimum": 0, "default": 0},
                },
            },
        },
    },
]

TOOL_DISPATCH = {
    "oracle_ar_discover_customers": tool_discover_customers,
    "oracle_ar_list_invoices": tool_list_invoices,
    "oracle_ar_list_receipts": tool_list_receipts,
    "oracle_ar_get_customer_summary": tool_customer_summary,
    "oracle_ar_get_aging_summary": tool_aging_summary,
}


def run_agent(messages, openai_key: str):
    client = OpenAI(api_key=openai_key)
    tool_log = []
    max_iterations = 8  # Increased from 4

    for iteration in range(max_iterations):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
        except Exception as e:
            return f"OpenAI API error: {e}", tool_log
            
        msg = resp.choices[0].message

        # No tool calls = we have a final answer
        if not msg.tool_calls:
            return (msg.content or "").strip(), tool_log

        # Process tool calls
        messages.append({
            "role": "assistant", 
            "content": msg.content, 
            "tool_calls": [tc.model_dump() for tc in msg.tool_calls]
        })

        for tc in msg.tool_calls:
            name = tc.function.name
            raw_args = tc.function.arguments or "{}"
            try:
                args = json.loads(raw_args)
            except Exception:
                args = {}

            tool_log.append({"iteration": iteration + 1, "tool": name, "arguments": args})
            
            fn = TOOL_DISPATCH.get(name)
            if fn:
                try:
                    result = fn(args)
                except Exception as e:
                    result = json.dumps({"error": f"Tool execution failed: {e}"})
            else:
                result = json.dumps({"error": f"Unknown tool: {name}"})
            
            tool_log[-1]["result_preview"] = result[:500] if len(result) > 500 else result
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})

    return "Agent completed maximum iterations. Results may be incomplete.", tool_log


# =========================
# UI
# =========================
st.set_page_config(page_title="Oracle AR Agent", page_icon="🧾", layout="wide")
st.title("Oracle Receivables Agent")

with st.sidebar:
    st.header("Configuration")

    st.subheader("OpenAI")
    st.session_state.openai_key = st.text_input(
        "OpenAI API Key", 
        type="password", 
        value=st.session_state.openai_key
    )

    st.divider()

    st.subheader("Oracle Fusion")
    st.session_state.oracle_base_url = st.text_input(
        "Base URL",
        value=st.session_state.oracle_base_url,
        on_change=mark_oracle_dirty,
        placeholder="https://<pod>.oraclecloud.com",
    )
    st.session_state.oracle_username = st.text_input(
        "Username", 
        value=st.session_state.oracle_username, 
        on_change=mark_oracle_dirty
    )
    st.session_state.oracle_password = st.text_input(
        "Password", 
        type="password", 
        value=st.session_state.oracle_password, 
        on_change=mark_oracle_dirty
    )

    st.session_state.tls_verify = st.checkbox(
        "TLS verify",
        value=st.session_state.tls_verify,
        help="Disable only for sandbox/test environments with self-signed certs",
        on_change=mark_oracle_dirty,
    )

    oracle_validate_if_needed()

    if st.session_state.oracle_status.startswith("Connected"):
        st.success(st.session_state.oracle_status)
    elif st.session_state.oracle_status:
        st.error(st.session_state.oracle_status)
    
    st.divider()
    
    st.session_state.debug_mode = st.checkbox(
        "Debug mode", 
        value=st.session_state.get("debug_mode", True)
    )
    
    if st.button("Clear chat"):
        st.session_state.chat = []
        st.rerun()


# Chat display
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("tool_log") and st.session_state.debug_mode:
            with st.expander("🔧 Tool calls"):
                st.json(msg["tool_log"])

# Chat input
user_text = st.chat_input("Try: 'show me all customers' or 'list invoices' or 'aging summary'")

if user_text:
    if not (st.session_state.openai_key or "").strip():
        st.error("Enter your OpenAI API key in the sidebar.")
    elif not st.session_state.oracle_validated:
        st.error("Oracle connection not established. Check your credentials.")
    else:
        st.session_state.chat.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.write(user_text)

        with st.spinner("Querying Oracle..."):
            agent_messages = [
                {
                    "role": "system",
                    "content": """You are an Oracle Fusion AR assistant. 

IMPORTANT RULES:
1. To find customers, use oracle_ar_discover_customers (extracts from invoice data)
2. To see invoices, use oracle_ar_list_invoices  
3. To see payments, use oracle_ar_list_receipts
4. For customer totals, use oracle_ar_get_customer_summary (requires customer_account_id)
5. For aging, use oracle_ar_get_aging_summary

When user asks for "customers" or "customer list", call oracle_ar_discover_customers FIRST.
When user mentions a customer name, call oracle_ar_list_invoices with customer_name filter.

Be concise. Present data in readable format. If an error occurs, explain it clearly.""",
                }
            ] + [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat]

            answer, tool_log = run_agent(agent_messages, st.session_state.openai_key)

        st.session_state.chat.append({
            "role": "assistant", 
            "content": answer,
            "tool_log": tool_log if tool_log else None
        })
        
        with st.chat_message("assistant"):
            st.write(answer)
            if tool_log and st.session_state.debug_mode:
                with st.expander("🔧 Tool calls"):
                    st.json(tool_log)
