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
"""
Aessefin Ticket Router (FastAPI)

- POST /webhook  (JSON body, header: X-Forms-Secret)
- Validates & routes Google Form submissions to Monday.com
- Uses Gemini (if GOOGLE_GEMINI_API_KEY set) or OpenAI (if OPENAI_API_KEY set)
  to decide accept/reject and to suggest a normalized title/summary.
- Guarantees category prefix appears AT MOST ONCE in the final title.

Env vars needed (Render → Environment):
  FORMS_SHARED_SECRET
  MONDAY_API_TOKEN
  MONDAY_BOARD_ID
  MONDAY_GROUP_NEW                   (e.g. "topics")

  # Column IDs from the Monday API Playground
  MONDAY_COLUMN_EMAIL
  MONDAY_COLUMN_CATEGORY
  MONDAY_COLUMN_PRIORITY
  MONDAY_COLUMN_DESCRIPTION
  MONDAY_COLUMN_ATTACHMENTS
  MONDAY_COLUMN_LINK_LONGTEXT

  # Optional mail if you ever want to email clarifications
  EMAIL_TO_DEFAULT
  SMTP_HOST
  SMTP_PORT
  SMTP_USER
  SMTP_PASS

  # LLM (use one or none)
  GOOGLE_GEMINI_API_KEY              # preferred
  OPENAI_API_KEY                     # fallback
"""

import os
import re
import json
from typing import Optional, Dict, Any, List

import httpx
from fastapi import FastAPI, Request, Header, HTTPException

# -------------------------
# Environment
# -------------------------
FORMS_SHARED_SECRET = os.getenv("FORMS_SHARED_SECRET", "forms-shared-secret-123")

MONDAY_API_TOKEN = os.getenv("MONDAY_API_TOKEN", "")
MONDAY_BOARD_ID  = os.getenv("MONDAY_BOARD_ID", "")
MONDAY_GROUP_NEW = os.getenv("MONDAY_GROUP_NEW", "topics")

COL_EMAIL        = os.getenv("MONDAY_COLUMN_EMAIL", "")
COL_CATEGORY     = os.getenv("MONDAY_COLUMN_CATEGORY", "")
COL_PRIORITY     = os.getenv("MONDAY_COLUMN_PRIORITY", "")
COL_DESCRIPTION  = os.getenv("MONDAY_COLUMN_DESCRIPTION", "")
COL_ATTACHMENTS  = os.getenv("MONDAY_COLUMN_ATTACHMENTS", "")
COL_LINK_LONGTXT = os.getenv("MONDAY_COLUMN_LINK_LONGTEXT", "")

# LLM (Gemini preferred, OpenAI fallback)
GEMINI_KEY       = os.getenv("GOOGLE_GEMINI_API_KEY", "")
GEMINI_MODEL     = os.getenv("GOOGLE_GEMINI_MODEL", "gemini-1.5-flash")
OPENAI_KEY       = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL     = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# -------------------------
# FastAPI
# -------------------------
app = FastAPI(title="Aessefin Ticket Router", version="2.0")


# -------------------------
# Helpers (validation)
# -------------------------
CATEGORIES_ALLOWED = {
    "CRM",
    "AMMINISTRAZIONE",
    "CONSOLE",
    "OPERATIONS (Cambi asseganzione, etc.)",
}

def valid_category(cat: str) -> bool:
    return cat in CATEGORIES_ALLOWED

def word_count(text: str) -> int:
    return len([w for w in (text or "").strip().split() if w])

def is_vague(text: str) -> bool:
    """Very simple vagueness detection."""
    t = (text or "").lower().strip()
    vague = [
        r"\bi have (an )?issue\b",
        r"\bsomething (is )?not working\b",
        r"\bplease fix\b",
        r"^help\b",
        r"\bhelp\b$",
    ]
    return any(re.search(p, t) for p in vague)

def is_drive_url(u: str) -> bool:
    u = (u or "").strip().lower()
    return u.startswith("https://drive.google.com/") or u.startswith("http://drive.google.com/")

def filter_drive_urls(urls: List[str]) -> List[str]:
    if not urls:
        return []
    return [u for u in urls if is_drive_url(u)]

# -------------------------
# Helpers (title handling)
# -------------------------
def strip_cat_prefix(text: str, categoria: str) -> str:
    """Remove a leading '<categoria>:' if already present (case-insensitive)."""
    if not text or not categoria:
        return (text or "").strip()
    pat = r'^\s*' + re.escape(categoria) + r'\s*:\s*'
    return re.sub(pat, '', text, flags=re.IGNORECASE).strip()

def ensure_single_cat_prefix(title: str, categoria: str) -> str:
    """
    Ensure the title has the category prefix at most once.
    """
    core = strip_cat_prefix(title, categoria)
    return f"{categoria}: {core}" if categoria and core else (title or categoria)

def clip30(s: str) -> str:
    s = (s or "").strip()
    return (s[:27] + "…") if len(s) > 30 else s

def normalize_title(desc: str, categoria: str) -> str:
    # Remove any existing category prefix
    base = strip_cat_prefix(desc or "", categoria)
    # Take the first sentence / few words for compactness
    first = re.split(r"[.!?\n]", base.strip())[0] or base.strip()
    words = first.split()
    first = " ".join(words[:6]) if len(words) > 6 else first
    if first:
        first = first[:1].upper() + first[1:]
    return clip30(first or "Ticket")


# -------------------------
# LLM prompt
# -------------------------
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

def build_user_payload(description: str, categoria: str, priorita: str,
                       reporter_email: str, link_of_record: str, attachments: list) -> str:
    inp = {
        "description": description,
        "categoria": categoria,
        "priorita": priorita,
        "email": reporter_email,
        "link_of_record": link_of_record,
        "attachments": attachments or []
    }
    return json.dumps(inp, ensure_ascii=False)


# -------------------------
# LLM calls
# -------------------------
async def llm_analyze_with_gemini(user_json: str) -> Optional[dict]:
    if not GEMINI_KEY:
        return None
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_KEY}"
    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": PROMPT_SYSTEM}]},
            {"role": "user", "parts": [{"text": user_json}]},
        ],
        "generationConfig": {"temperature": 0.2}
    }
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(text)
        except Exception:
            return None

async def llm_analyze_with_openai(user_json: str) -> Optional[dict]:
    if not OPENAI_KEY:
        return None
    headers = {"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": PROMPT_SYSTEM},
            {"role": "user", "content": user_json},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
    }
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        try:
            text = data["choices"][0]["message"]["content"]
            return json.loads(text)
        except Exception:
            return None

async def llm_decide(description: str, categoria: str, priorita: str,
                     reporter_email: str, link_of_record: str, attachments: list) -> Optional[dict]:
    user_json = build_user_payload(description, categoria, priorita, reporter_email, link_of_record, attachments)
    # Gemini first
    out = await llm_analyze_with_gemini(user_json)
    if out:
        return out
    # Fallback OpenAI
    out = await llm_analyze_with_openai(user_json)
    return out


# -------------------------
# Monday.com call
# -------------------------
async def monday_create(item_name: str, column_values: Dict[str, Any]) -> Dict[str, Any]:
    if not MONDAY_API_TOKEN or not MONDAY_BOARD_ID:
        raise HTTPException(status_code=500, detail="Monday API not configured.")
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
        "columnVals": json.dumps(column_values, ensure_ascii=False),
    }
    headers = {"Authorization": MONDAY_API_TOKEN, "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post("https://api.monday.com/v2", headers=headers, json={"query": query, "variables": variables})
        r.raise_for_status()
        data = r.json()
        if "errors" in data:
            raise HTTPException(status_code=502, detail=data["errors"])
        return data


# -------------------------
# Webhook
# -------------------------
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
    attachments_in = p.get("attachments") or []
    attachments    = filter_drive_urls(attachments_in)

    # Validate category
    if not valid_category(categoria):
        raise HTTPException(status_code=400, detail="Rejected: invalid category")

    # LLM FIRST if available
    llm = await llm_decide(description, categoria, priorita, reporter_email, link_of_record, attachments)

    if llm:
        next_action = (llm.get("next_action") or "").lower()
        router_dec  = (llm.get("router_decision") or "").lower()
        norm_title  = (llm.get("normalized_title") or "").strip()

        if next_action == "ask_clarify" or router_dec == "ask_clarify":
            raise HTTPException(status_code=400, detail="Rejected by LLM: ask_clarify")

        # Build item title with AT MOST ONE category prefix
        title_raw = norm_title if norm_title else normalize_title(description, categoria)
        item_name = clip30(ensure_single_cat_prefix(title_raw, categoria))

        # Map to columns (use llm['monday_fields'] if present; otherwise fallback)
        fields = llm.get("monday_fields") or {}
        # Ensure we still fill our known columns
        col_vals: Dict[str, Any] = {}
        if COL_DESCRIPTION:
            col_vals[COL_DESCRIPTION] = fields.get("Descrizione Dettagliata") or description

        if COL_EMAIL:
            email_obj = fields.get("Email") or ({"email": reporter_email, "text": reporter_email} if reporter_email else {"email": "", "text": ""})
            col_vals[COL_EMAIL] = email_obj

        if COL_CATEGORY and categoria:
            col_vals[COL_CATEGORY] = {"label": categoria}

        if COL_PRIORITY and priorita:
            col_vals[COL_PRIORITY] = {"label": priorita}

        if COL_ATTACHMENTS and attachments:
            col_vals[COL_ATTACHMENTS] = "\n".join(attachments)

        if COL_LINK_LONGTXT and link_of_record:
            col_vals[COL_LINK_LONGTXT] = link_of_record

        created = await monday_create(item_name=item_name, column_values=col_vals)
        return {"ok": True, "source": "llm", "created": created, "item_name": item_name}

    # ---- No LLM available: rule-based fallback ----
    is_ops = categoria.upper().startswith("OPERATIONS")
    if word_count(description) < 10 and not is_ops:
        raise HTTPException(status_code=400, detail="Rejected: description too short")
    if is_vague(description) and not is_ops:
        raise HTTPException(status_code=400, detail="Rejected: description too vague")

    title_raw = normalize_title(description, categoria)
    item_name = clip30(ensure_single_cat_prefix(title_raw, categoria))

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
        col_vals[COL_LINK_LONGTXT] = link_of_record

    created = await monday_create(item_name=item_name, column_values=col_vals)
    return {"ok": True, "source": "rules", "created": created, "item_name": item_name}


@app.get("/")
def health():
    return {"status": "ok"}

