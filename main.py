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

"""
Makeflow-Python: webhook -> guard -> (create | ask_clarify)
"""

"""
Aessefin Ticket Router (FastAPI) with Gemini gating and Monday.com creation.

Flow:
1) POST /webhook with JSON + X-Forms-Secret header
2) We call Gemini with your strict JSON prompt. If Gemini says "create", we create a Monday item.
   If it says "ask_clarify", we return that and DO NOT create a ticket.
3) If Gemini fails, a conservative rule-based gate is applied.

This service expects Monday column IDs via environment variables.
"""

import os
import json
import re
from typing import Optional, Dict, Any, Tuple

import httpx
from fastapi import FastAPI, Request, Header, HTTPException

# -----------------------
# Environment
# -----------------------
FORMS_SHARED_SECRET = os.getenv("FORMS_SHARED_SECRET", "forms-shared-secret-123")

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# Monday
MONDAY_API_TOKEN = os.getenv("MONDAY_API_TOKEN", "")
MONDAY_BOARD_ID = os.getenv("MONDAY_BOARD_ID", "")  # integer or string
MONDAY_GROUP_NEW = os.getenv("MONDAY_GROUP_NEW", "topics")

# Monday column IDs (exact column "id", not title)
COL_EMAIL        = os.getenv("MONDAY_COLUMN_EMAIL", "")
COL_CATEGORY     = os.getenv("MONDAY_COLUMN_CATEGORY", "")
COL_PRIORITY     = os.getenv("MONDAY_COLUMN_PRIORITY", "")
COL_DESCRIPTION  = os.getenv("MONDAY_COLUMN_DESCRIPTION", "")
COL_ATTACHMENTS  = os.getenv("MONDAY_COLUMN_ATTACHMENTS", "")
COL_LINK_LONGTXT = os.getenv("MONDAY_COLUMN_LINK_LONGTEXT", "")

if not (MONDAY_API_TOKEN and MONDAY_BOARD_ID):
    raise RuntimeError("Set MONDAY_API_TOKEN and MONDAY_BOARD_ID env vars.")

if not GEMINI_API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY to enable Gemini routing.")

# -----------------------
# Prompt (your exact text)
# -----------------------
PROMPT_SYSTEM = (
    'You are "Ticket Verificator Bot". You evaluate Google Form responses to decide if they can '
    'become tickets in Monday.com.\n\n'
    "Your job:\n"
    "1) Use the category provided in the form (must be one of: CRM, AMMINISTRAZIONE, CONSOLE, "
    "OPERATIONS (Cambi asseganzione, etc.). Do NOT infer or change it.\n"
    "2) Evaluate the description:\n"
    "   - If the description has fewer than 10 words → reject.\n"
    '   - If it’s vague and doesn’t clearly describe the issue (e.g., “I have an issue”, '
    '“something not working”, “please fix”) → reject.\n'
    "   - Otherwise → accept.\n"
    '   - Exception: if the category is "OPERATIONS (Cambi asseganzione, etc.)", accept with fewer than 10 words only if still meaningful.\n'
    "3) If accepted (Ticket is ACCEPTED):\n"
    "   - Generate a clear, normalized title (≤ 30 characters).\n"
    "   - Provide a concise, helpful summary (≥ 20 words) and update the description in Italian if needed.\n"
    "   - Priority is given by user (URGENTE, MEDIA, ALTA, BASSA).\n"
    "   - Collect only Google Drive file links in `Allegati` (array).\n"
    "   - Copy the exact “Link of the record”.\n"
    "   - Build the `Email` object.\n"
    '   - Set next_action = "create" AND router_decision = "create".\n'
    "   - Populate ALL fields in monday_fields.\n"
    "4) If rejected (Ticket is REJECTED):\n"
    '   - Set next_action = "ask_clarify" AND router_decision = "ask_clarify".\n'
    "   - Do NOT populate monday_fields beyond placeholders.\n\n"
    "Rules: Reply ONLY valid JSON, no code fences. Do not invent data.\n\n"
    "OUTPUT JSON:\n"
    "{\n"
    '  "next_action": "create|ask_clarify",\n'
    '  "router_decision": "create|ask_clarify",\n'
    '  "normalized_title": "string",\n'
    '  "categoria": "CRM|AMMINISTRAZIONE|CONSOLE|OPERATIONS (Cambi asseganzione, etc.)",\n'
    '  "priorita": "URGENTE|MEDIA|ALTA|BASSA",\n'
    '  "monday_fields": {\n'
    '    "Item": "string",\n'
    '    "Categoria": "string",\n'
    '    "Priorità": "string",\n'
    '    "Descrizione Dettagliata": "string",\n'
    '    "Allegati": ["drive_url_1"],\n'
    '    "Link_of_the_record": "string",\n'
    '    "Email": { "email": "user@domain.com", "text": "user@domain.com" }\n'
    "  }\n"
    "}\n"
)

# -----------------------
# FastAPI
# -----------------------
app = FastAPI(title="Aessefin Ticket Router")

# -----------------------
# Helpers
# -----------------------
def is_vague_rule(text: str, categoria: str) -> bool:
    """Conservative fallback gate if Gemini fails."""
    t = (text or "").strip().lower()
    words = len(t.split())
    if categoria.upper().startswith("OPERATIONS"):
        return False  # exception for operations (still meaningful)
    if words < 10:
        return True
    vague = [
        r"\bi have (an )?issue\b",
        r"\bsomething (is )?not working\b",
        r"\bplease fix\b",
        r"^help\b",
        r"\bhelp\b$",
    ]
    return any(re.search(p, t) for p in vague)

def dedupe_category_prefix(title: str, categoria: str) -> str:
    t = title.strip()
    if not categoria:
        return t
    c = categoria.strip()
    pattern = re.compile(rf"^{re.escape(c)}:\s*", re.IGNORECASE)
    t = pattern.sub("", t)
    base = f"{c}: {t}" if t else c
    return (base[:27] + "…") if len(base) > 30 else base

def only_drive_links(urls: list) -> list:
    out = []
    for u in urls or []:
        if isinstance(u, str) and ("drive.google.com" in u or "docs.google.com" in u):
            out.append(u)
    return out

def norm_priority(label: str) -> str:
    if not label: return ""
    m = label.strip().upper()
    mapping = {"URGENTE": "URGENTE", "MEDIA": "MEDIA", "ALTA": "ALTA", "BASSA": "BASSA",
               "URGENT": "URGENTE", "MEDIUM": "MEDIA", "HIGH": "ALTA", "LOW": "BASSA"}
    return mapping.get(m, m)

# -----------------------
# Gemini call
# -----------------------
async def gemini_route(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calls Gemini with the strict system prompt and returns a parsed JSON dict:
    {
      next_action, router_decision, normalized_title, categoria, priorita, monday_fields:{...}
    }
    """
    body = {
        "contents": [
            {
                "parts": [
                    {"text": PROMPT_SYSTEM},
                    {"text": "Input JSON to evaluate:\n" + json.dumps(payload, ensure_ascii=False)}
                ]
            }
        ],
        # Ask for JSON output explicitly
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    async with httpx.AsyncClient(timeout=45) as client:
        r = await client.post(url, json=body)
        r.raise_for_status()
        data = r.json()
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Gemini malformed response: {e}")

    try:
        parsed = json.loads(text)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini did not return JSON: {e}")

    # Minimal shape check
    for k in ["next_action", "router_decision", "normalized_title", "monday_fields"]:
        if k not in parsed:
            raise HTTPException(status_code=502, detail=f"Gemini missing key: {k}")
    return parsed

# -----------------------
# Monday creation
# -----------------------
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

# -----------------------
# Routes
# -----------------------
@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/webhook")
async def webhook(req: Request, x_forms_secret: Optional[str] = Header(None)):
    # 1) Secret check
    if x_forms_secret != FORMS_SHARED_SECRET:
        raise HTTPException(status_code=401, detail="bad secret")

    # 2) Read input
    p = await req.json()
    description    = (p.get("description") or "").strip()
    categoria      = (p.get("categoria") or "").strip()
    priorita       = (p.get("priorita")  or "").strip()
    reporter_email = (p.get("email")     or "").strip()
    link_of_record = (p.get("link_of_record") or "").strip()
    attachments    = p.get("attachments") or []

    # 3) Ask Gemini
    try:
        g = await gemini_route(p)
    except HTTPException:
        # If Gemini fails hard, use conservative fallback
        if is_vague_rule(description, categoria):
            return {"ok": False, "routed_to": "ask_clarify", "reason": "fallback_vague"}
        # else create with a simple title
        title = description.split("\n")[0][:30] or "Ticket"
        title = dedupe_category_prefix(title, categoria)
        col_vals: Dict[str, Any] = {}
        if COL_DESCRIPTION:  col_vals[COL_DESCRIPTION] = description
        if COL_EMAIL:        col_vals[COL_EMAIL]       = {"email": reporter_email, "text": reporter_email}
        if COL_CATEGORY and categoria: col_vals[COL_CATEGORY] = {"label": categoria}
        if COL_PRIORITY and priorita:  col_vals[COL_PRIORITY]  = {"label": priorita}
        if COL_ATTACHMENTS:  col_vals[COL_ATTACHMENTS] = "\n".join(only_drive_links(attachments))
        if COL_LINK_LONGTXT and link_of_record: col_vals[COL_LINK_LONGTXT] = link_of_record
        created = await monday_create(item_name=title, column_values=col_vals)
        return {"ok": True, "routed_to": "create_fallback", "created": created}

    # 4) Obey Gemini router
    action = str(g.get("next_action", "")).lower()
    if action != "create":
        # ask_clarify: do not create the item
        return {"ok": False, "routed_to": "ask_clarify", "ai": g}

    # 5) Build title from Gemini (and dedupe category)
    ai_title = (g.get("normalized_title") or "").strip()
    title = ai_title or (description.split("\n")[0][:30] or "Ticket")
    title = dedupe_category_prefix(title, categoria)

    # 6) Column values – prefer Gemini’s monday_fields, but enforce board schema
    mf = g.get("monday_fields", {}) or {}

    # Category & Priority normalization
    categoria_label = mf.get("Categoria") or categoria
    priorita_label  = norm_priority(mf.get("Priorità") or priorita)

    email_obj = mf.get("Email") or {"email": reporter_email, "text": reporter_email}
    drive_links = only_drive_links(mf.get("Allegati") if isinstance(mf.get("Allegati"), list) else attachments)

    descr = mf.get("Descrizione Dettagliata") or description
    link  = mf.get("Link_of_the_record") or link_of_record

    col_vals: Dict[str, Any] = {}
    if COL_DESCRIPTION:  col_vals[COL_DESCRIPTION] = descr
    if COL_EMAIL:        col_vals[COL_EMAIL]       = {"email": email_obj.get("email",""), "text": email_obj.get("text","")}
    if COL_CATEGORY and categoria_label: col_vals[COL_CATEGORY] = {"label": categoria_label}
    if COL_PRIORITY and priorita_label:  col_vals[COL_PRIORITY]  = {"label": priorita_label}
    if COL_ATTACHMENTS and drive_links:  col_vals[COL_ATTACHMENTS] = "\n".join(drive_links)
    if COL_LINK_LONGTXT and link:        col_vals[COL_LINK_LONGTXT] = link

    created = await monday_create(item_name=title, column_values=col_vals)
    return {"ok": True, "routed_to": "create", "created": created, "ai": g}
