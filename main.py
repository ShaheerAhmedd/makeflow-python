"""
Makeflow-Python: a FastAPI clone of your Make.com scenario.

Flow:
1) POST /webhook  <-- your external system hits this with JSON
2) We classify the payload (rule-based by default; optional LLM if OPENAI_API_KEY is set)
3) Router:
   - intent == "clarify"  -> send_email()
   - intent == "create"   -> create_monday_item()

Quickstart:
1) python -m venv .venv && source .venv/bin/activate
2) pip install -r requirements.txt
3) cp .env.example .env  # then fill values
4) uvicorn main:app --reload

Test:
curl -X POST http://127.0.0.1:8000/webhook \
  -H "Content-Type: application/json" \
  -d '{"sender":"alice@example.com","subject":"Need access","message":"Please create a task to add me to the workspace"}'
"""

import os, json
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, Request, Header, HTTPException

# --------- ENV / CONFIG ----------
MONDAY_API_TOKEN      = os.getenv("MONDAY_API_TOKEN", "")
MONDAY_BOARD_ID       = os.getenv("MONDAY_BOARD_ID", "")
MONDAY_GROUP_NEW      = os.getenv("MONDAY_GROUP_NEW", "topics")

# Column IDs on your board
COL_EMAIL             = os.getenv("MONDAY_COLUMN_EMAIL", "email_mkw8932y")
COL_CATEGORY          = os.getenv("MONDAY_COLUMN_CATEGORY", "color_mkwx44c")
COL_PRIORITY          = os.getenv("MONDAY_COLUMN_PRIORITY", "color_mkwten2j")
COL_DESCRIPTION       = os.getenv("MONDAY_COLUMN_DESCRIPTION", "text_mkw4jrjy")

FORMS_SHARED_SECRET   = os.getenv("FORMS_SHARED_SECRET", "forms-shared-secret-123")

if not MONDAY_API_TOKEN or not MONDAY_BOARD_ID:
    raise RuntimeError("Set MONDAY_API_TOKEN and MONDAY_BOARD_ID in environment.")

app = FastAPI(title="Makeflow-Python (New Tickets Only)")

@app.get("/")
def health():
    return {"status": "ok"}

# --------- Monday helpers ----------
async def monday_create_item(item_name: str, column_values: Dict[str, Any]) -> Dict[str, Any]:
    query = """
    mutation ($boardId: ID!, $groupId: String!, $itemName: String!, $columnVals: JSON!) {
      create_item(
        board_id: $boardId,
        group_id: $groupId,
        item_name: $itemName,
        column_values: $columnVals
      ) { id name }
    }
    """
    variables = {
        "boardId": MONDAY_BOARD_ID,
        "groupId": MONDAY_GROUP_NEW,
        "itemName": item_name,
        "columnVals": json.dumps(column_values),
    }
    headers = {"Authorization": MONDAY_API_TOKEN, "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post("https://api.monday.com/v2", headers=headers, json={"query": query, "variables": variables})
        r.raise_for_status()
        data = r.json()
        # Bubble up GraphQL errors cleanly
        if "errors" in data:
            raise HTTPException(status_code=502, detail=data["errors"])
        return data

# --------- Webhook (Google Form -> here) ----------
@app.post("/webhook")
async def webhook(
    request: Request,
    x_forms_secret: Optional[str] = Header(None)
):
    if x_forms_secret != FORMS_SHARED_SECRET:
        raise HTTPException(status_code=401, detail="bad secret")

    try:
        payload = await request.json()
    except Exception:
        payload = {}

    # Normalize incoming fields
    description    = (payload.get("description") or "").strip()
    categoria      = (payload.get("categoria") or "").strip().upper()
    priorita       = (payload.get("priorita")  or "").strip().upper()
    reporter_email = (payload.get("email")     or "").strip()
    attachments    = payload.get("attachments") or []           # not used unless you add a column
    link_of_record = (payload.get("link_of_record") or "").strip()

    if not description:
        raise HTTPException(status_code=400, detail="description is required")

    # Item title: keep short and useful
    item_name = (description[:30] + "…") if len(description) > 30 else description or "Richiesta"

    # Build Monday column values
    column_values: Dict[str, Any] = {
        COL_DESCRIPTION: description,
        COL_EMAIL: {"email": reporter_email, "text": reporter_email} if reporter_email else {"email": "", "text": ""},
        COL_CATEGORY: {"label": categoria} if categoria else None,
        COL_PRIORITY: {"label": priorita} if priorita else None,
    }
    # Remove Nones (GraphQL chokes on them)
    column_values = {k: v for k, v in column_values.items() if v is not None}

    # NOTE: if you also want to set "Link of the record" in a Link/Text column,
    # add its column ID as env and include here, e.g.:
    # COL_LINK = os.getenv("MONDAY_COLUMN_LINK_OF_RECORD")
    # if link_of_record and COL_LINK:
    #     column_values[COL_LINK] = {"url": link_of_record, "text": "Record"}

    result = await monday_create_item(item_name=item_name, column_values=column_values)
    return {"ok": True, "created": result}
