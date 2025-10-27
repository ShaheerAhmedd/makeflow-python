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

import os, json, re
from typing import Optional, Dict, Any
import httpx
from fastapi import FastAPI, Request, Header, HTTPException

MONDAY_API_TOKEN = os.getenv("MONDAY_API_TOKEN", "")
MONDAY_BOARD_ID  = os.getenv("MONDAY_BOARD_ID", "")
MONDAY_GROUP_NEW = os.getenv("MONDAY_GROUP_NEW", "topics")

COL_EMAIL        = os.getenv("MONDAY_COLUMN_EMAIL", "")
COL_CATEGORY     = os.getenv("MONDAY_COLUMN_CATEGORY", "")
COL_PRIORITY     = os.getenv("MONDAY_COLUMN_PRIORITY", "")
COL_DESCRIPTION  = os.getenv("MONDAY_COLUMN_DESCRIPTION", "")
COL_ATTACHMENTS  = os.getenv("MONDAY_COLUMN_ATTACHMENTS", "")
COL_LINK_LONGTXT = os.getenv("MONDAY_COLUMN_LINK_LONGTEXT", "")

FORMS_SHARED_SECRET = os.getenv("FORMS_SHARED_SECRET", "forms-shared-secret-123")
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY", "")   # optional
OPENAI_MODEL        = "gpt-4o-mini"

if not MONDAY_API_TOKEN or not MONDAY_BOARD_ID:
    raise RuntimeError("Set MONDAY_API_TOKEN and MONDAY_BOARD_ID.")

app = FastAPI(title="Aessefin Ticket Router")

def is_vague(text: str) -> bool:
    t = text.lower().strip()
    if len(t.split()) < 10:
        return True
    vague = [
        r"\bi have (an )?issue\b",
        r"\bsomething (is )?not working\b",
        r"\bplease fix\b",
        r"^help\b",
        r"\bhelp\b$",
    ]
    return any(re.search(p, t) for p in vague)

def normalize_title(desc: str, categoria: str) -> str:
    base = re.split(r"[.!?\n]", desc.strip())[0] or desc.strip()
    words = base.split()
    base = " ".join(words[:6]) if len(words) > 6 else base
    base = base[:1].upper() + base[1:] if base else "Ticket"
    if categoria:
        base = f"{categoria}: {base}"
    return (base[:27] + "…") if len(base) > 30 else base

async def openai_title(desc: str, categoria: str) -> Optional[str]:
    if not OPENAI_API_KEY:
        return None
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": "Generate a clear IT ticket title under 30 chars."},
            {"role": "user", "content": f"Category: {categoria}\nDescription: {desc}"}
        ],
        "temperature": 0.2
    }
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            r.raise_for_status()
            title = r.json()["choices"][0]["message"]["content"].strip()
            return (title[:27] + "…") if len(title) > 30 else title
    except Exception:
        return None

async def monday_create(item_name: str, column_values: Dict[str, Any]) -> Dict[str, Any]:
    query = """
    mutation ($boardId: ID!, $groupId: String!, $itemName: String!, $columnVals: JSON!) {
      create_item(board_id: $boardId, group_id: $groupId, item_name: $itemName, column_values: $columnVals) {
        id
        name
      }
    }
    """
    variables = {
        "boardId": MONDAY_BOARD_ID,
        "groupId": MONDAY_GROUP_NEW,
        "itemName": item_name,
        "columnVals": json.dumps(column_values)
    }
    headers = {"Authorization": MONDAY_API_TOKEN, "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post("https://api.monday.com/v2", headers=headers, json={"query": query, "variables": variables})
        r.raise_for_status()
        data = r.json()
        if "errors" in data:
            raise HTTPException(status_code=502, detail=data["errors"])
        return data

@app.post("/webhook")
async def webhook(req: Request, x_forms_secret: Optional[str] = Header(None)):
    if x_forms_secret != FORMS_SHARED_SECRET:
        raise HTTPException(status_code=401, detail="bad secret")

    p = await req.json()

    description    = (p.get("description") or "").strip()
    categoria      = (p.get("categoria") or "").strip()
    priorita       = (p.get("priorita")  or "").strip()
    reporter_email = (p.get("email")     or "").strip()
    link_of_record = (p.get("link_of_record") or "").strip()
    attachments    = p.get("attachments") or []  # array of strings if present

    # Gatekeeping: reject vague/short unless OPERATIONS
    is_ops = categoria.upper().startswith("OPERATIONS")
    if is_vague(description) and not is_ops:
        raise HTTPException(status_code=400, detail="Rejected: description too short/vague")

    # Title
    title = await openai_title(description, categoria) or normalize_title(description, categoria)

    # Map to Monday columns
    col_vals: Dict[str, Any] = {}
    if COL_DESCRIPTION:
        col_vals[COL_DESCRIPTION] = description

    if COL_EMAIL:
        col_vals[COL_EMAIL] = {"email": reporter_email, "text": reporter_email} if reporter_email else {"email": "", "text": ""}

    if COL_CATEGORY and categoria:
        col_vals[COL_CATEGORY] = {"label": categoria}

    if COL_PRIORITY and priorita:
        col_vals[COL_PRIORITY] = {"label": priorita}

    if COL_ATTACHMENTS and attachments:
        col_vals[COL_ATTACHMENTS] = "\n".join(attachments)

    if COL_LINK_LONGTXT and link_of_record:
        # this is a long_text column, so store plain text
        col_vals[COL_LINK_LONGTXT] = link_of_record

    created = await monday_create(item_name=title, column_values=col_vals)
    return {"ok": True, "created": created}

