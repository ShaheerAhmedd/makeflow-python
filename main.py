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

import os, json, re
from typing import Optional, Dict, Any, List, Tuple

import httpx
from fastapi import FastAPI, Request, Header, HTTPException

# -----------------------------
# Environment
# -----------------------------
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

# Gemini (optional but recommended)
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL     = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

if not MONDAY_API_TOKEN or not MONDAY_BOARD_ID:
    raise RuntimeError("Set MONDAY_API_TOKEN and MONDAY_BOARD_ID.")

# -----------------------------
# Constants / Prompt
# -----------------------------
VALID_CATEGORIES = {
    "CRM", "AMMINISTRAZIONE", "CONSOLE", "OPERATIONS (Cambi asseganzione, etc.)"
}
VALID_PRIORITIES = {"URGENTE", "MEDIA", "ALTA", "BASSA"}

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
    "   - Set next_action = \"create\" AND router_decision = \"create\".\n"
    "   - Populate ALL fields in monday_fields.\n"
    "4) If rejected (Ticket is REJECTED):\n"
    "   - Set next_action = \"ask_clarify\" AND router_decision = \"ask_clarify\".\n"
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

# -----------------------------
# App
# -----------------------------
app = FastAPI(title="Aessefin Ticket Router (Gemini)")

# -----------------------------
# Helpers
# -----------------------------
def strip_leading_category(desc: str) -> str:
    """Remove a leading known category like 'CRM:' from description to avoid doubling in titles."""
    if not desc:
        return desc
    lead = desc.strip()
    for cat in VALID_CATEGORIES:
        prefix = f"{cat}:"
        if lead.upper().startswith(prefix):
            return lead[len(prefix):].lstrip()
    return lead

def make_prefixed_title(category: str, core_title: str, max_len: int = 30) -> str:
    """
    Ensure the category appears once as 'CAT: Title'. If 'core_title' already starts with
    category + ':', don’t duplicate it.
    """
    core = core_title.strip()
    if core.upper().startswith(f"{category}:"):
        title = core  # already prefixed
    else:
        title = f"{category}: {core}"

    if len(title) > max_len:
        return title[: max_len - 1] + "…"
    return title

def is_drive_url(url: str) -> bool:
    return "drive.google.com" in (url or "")

def filter_drive_links(urls: List[str]) -> List[str]:
    return [u for u in urls if is_drive_url(u)]

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
        "columnVals": json.dumps(column_values),
    }
    headers = {"Authorization": MONDAY_API_TOKEN, "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post("https://api.monday.com/v2", headers=headers, json={"query": query, "variables": variables})
        r.raise_for_status()
        data = r.json()
        if "errors" in data:
            raise HTTPException(status_code=502, detail=data["errors"])
        return data

# -----------------------------
# LLM (Gemini)
# -----------------------------
async def call_gemini(form_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Call Gemini with the Ticket Verificator prompt. Return parsed JSON dict or None on failure.
    """
    if not GEMINI_API_KEY:
        return None

    # Clean & prep payload for the model
    payload_to_model = {
        "description": strip_leading_category(form_payload.get("description", "")),
        "categoria": form_payload.get("categoria", ""),
        "priorita": form_payload.get("priorita", ""),
        "email": form_payload.get("email", ""),
        "link_of_record": form_payload.get("link_of_record", ""),
        "attachments": form_payload.get("attachments", []),
    }

    gen_input = (
        PROMPT_SYSTEM
        + "\n\nINPUT:\n"
        + json.dumps(payload_to_model, ensure_ascii=False)
    )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    body = {
        "contents": [
            {"parts": [{"text": gen_input}]}
        ],
        "generationConfig": {
            "temperature": 0.2,
            "response_mime_type": "application/json"
        }
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, json=body)
            resp.raise_for_status()
            data = resp.json()
            # Gemini returns text in candidates[0].content.parts[].text
            candidates = data.get("candidates") or []
            if not candidates:
                return None
            txt = ""
            for part in candidates[0].get("content", {}).get("parts", []):
                if "text" in part:
                    txt += part["text"]
            txt = txt.strip()
            # try to parse JSON
            return json.loads(txt)
    except Exception:
        return None

def validate_ai_output(ai: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validate the AI JSON. Return (accepted, reason, normalized_dict).
    """
    if not isinstance(ai, dict):
        return False, "AI output not dict", {}

    na = ai.get("next_action", "")
    rd = ai.get("router_decision", "")

    accepted = (na == "create" and rd == "create")
    normalized_title = (ai.get("normalized_title") or "").strip()
    categoria = (ai.get("categoria") or "").strip()
    priorita = (ai.get("priorita") or "").strip()

    if categoria not in VALID_CATEGORIES:
        return False, "Invalid categoria from AI", {}

    if priorita not in VALID_PRIORITIES:
        return False, "Invalid priorita from AI", {}

    monday_fields = ai.get("monday_fields") or {}
    # required keys check
    mf_required = ["Item", "Categoria", "Priorità", "Descrizione Dettagliata", "Allegati", "Link_of_the_record", "Email"]
    if not all(k in monday_fields for k in mf_required):
        return False, "AI monday_fields missing keys", {}

    return accepted, "", {
        "title": normalized_title,
        "categoria": categoria,
        "priorita": priorita,
        "fields": monday_fields
    }

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def health():
    return {"status": "ok"}

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
    attachments    = p.get("attachments") or []

    # 1) Try Gemini to decide + format
    ai_decision = await call_gemini(p)

    if ai_decision:
        accepted, reason, norm = validate_ai_output(ai_decision)
        if accepted:
            # Use AI title and fields
            ai_title   = norm["title"] or "Ticket"
            ai_cat     = norm["categoria"]
            ai_prio    = norm["priorita"]
            fields     = norm["fields"]

            # Ensure category prefix exactly once in title
            clean_title = make_prefixed_title(ai_cat, ai_title)

            # Map AI fields to Monday columns
            col_vals: Dict[str, Any] = {}
            if COL_DESCRIPTION:
                col_vals[COL_DESCRIPTION] = fields.get("Descrizione Dettagliata", "")

            if COL_EMAIL:
                email_obj = fields.get("Email") or {}
                col_vals[COL_EMAIL] = {
                    "email": email_obj.get("email", ""),
                    "text":  email_obj.get("text",  "")
                }

            if COL_CATEGORY:
                col_vals[COL_CATEGORY] = {"label": ai_cat}

            if COL_PRIORITY:
                col_vals[COL_PRIORITY] = {"label": ai_prio}

            if COL_ATTACHMENTS:
                col_vals[COL_ATTACHMENTS] = "\n".join(filter_drive_links(fields.get("Allegati") or []))

            if COL_LINK_LONGTXT:
                col_vals[COL_LINK_LONGTXT] = fields.get("Link_of_the_record", "")

            created = await monday_create(clean_title, col_vals)
            return {"ok": True, "source": "gemini", "created": created}

        else:
            # AI says ask clarify or invalid => reject
            raise HTTPException(status_code=400, detail=f"Rejected by AI: {reason or 'ask_clarify'}")

    # 2) Fallback (no Gemini / failed): very light rules (minimum gate) + short title
    #    - Accept if >= 10 words or category starts with OPERATIONS
    words = len(strip_leading_category(description).split())
    is_ops = categoria.upper().startswith("OPERATIONS")
    if words < 10 and not is_ops:
        raise HTTPException(status_code=400, detail="Rejected: description too short/vague")

    # naive core title: first sentence trimmed
    core = re.split(r"[.!?\n]", strip_leading_category(description))[0].strip() or "Ticket"
    core = " ".join(core.split())  # normalize spaces

    # ensure single category prefix
    final_title = make_prefixed_title(categoria, core)

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
        col_vals[COL_ATTACHMENTS] = "\n".join(filter_drive_links(attachments))

    if COL_LINK_LONGTXT and link_of_record:
        col_vals[COL_LINK_LONGTXT] = link_of_record

    created = await monday_create(final_title, col_vals)
    return {"ok": True, "source": "rules", "created": created}
