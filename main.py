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

import os, re, json
from typing import Optional, Dict, Any, List
import httpx
from fastapi import FastAPI, Request, Header, HTTPException

# -----------------------------
# Env / Config
# -----------------------------
FORMS_SHARED_SECRET = os.getenv("FORMS_SHARED_SECRET", "forms-shared-secret-123")

GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL    = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

MONDAY_API_TOKEN = os.getenv("MONDAY_API_TOKEN", "")
MONDAY_BOARD_ID  = os.getenv("MONDAY_BOARD_ID", "")
MONDAY_GROUP_NEW = os.getenv("MONDAY_GROUP_NEW", "topics")

COL_EMAIL        = os.getenv("MONDAY_COLUMN_EMAIL", "")
COL_CATEGORY     = os.getenv("MONDAY_COLUMN_CATEGORY", "")
COL_PRIORITY     = os.getenv("MONDAY_COLUMN_PRIORITY", "")
COL_DESCRIPTION  = os.getenv("MONDAY_COLUMN_DESCRIPTION", "")
COL_ATTACHMENTS  = os.getenv("MONDAY_COLUMN_ATTACHMENTS", "")
COL_LINK_LONGTXT = os.getenv("MONDAY_COLUMN_LINK_LONGTEXT", "")

if not MONDAY_API_TOKEN or not MONDAY_BOARD_ID:
    raise RuntimeError("Set MONDAY_API_TOKEN and MONDAY_BOARD_ID")

VALID_CATEGORIES = {
    "CRM", "AMMINISTRAZIONE", "CONSOLE", "OPERATIONS (Cambi asseganzione, etc.)"
}
VALID_PRIORITIES = {"URGENTE", "MEDIA", "ALTA", "BASSA"}

# -----------------------------
# Prompt (your Make.com rubric)
# -----------------------------
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
# Helpers (title, filters, etc.)
# -----------------------------
def strip_any_leading_category(text: str) -> str:
    if not text:
        return text
    cats = [
        r"CRM",
        r"AMMINISTRAZIONE",
        r"CONSOLE",
        r"OPERATIONS \(Cambi asseganzione, etc\.\)",
    ]
    pattern = rf"^\s*(?:{'|'.join(cats)})\s*[:\-–—]?\s*"
    return re.sub(pattern, "", text.strip(), flags=re.IGNORECASE).strip()

def build_title_with_category(category: str, candidate: str, max_len: int = 30) -> str:
    core = strip_any_leading_category(candidate) or "Ticket"
    clean = f"{category}: {core}"
    return clean if len(clean) <= max_len else clean[:max_len - 1] + "…"

def first_sentence(text: str) -> str:
    if not text:
        return "Ticket"
    clean = strip_any_leading_category(text)
    sent = re.split(r"[.!?\n]", clean, maxsplit=1)[0].strip()
    return sent or "Ticket"

def filter_drive_links(urls: List[str]) -> List[str]:
    out = []
    for u in urls or []:
        if isinstance(u, str) and ("drive.google.com" in u or "docs.google.com" in u):
            out.append(u)
    return out

# -----------------------------
# Gemini
# -----------------------------
async def call_gemini(user_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not GEMINI_API_KEY:
        return None
    # Google Generative Language API (v1beta)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    contents = [
        {"role": "user", "parts": [{"text": PROMPT_SYSTEM}]},
        {"role": "user", "parts": [{"text": json.dumps(user_payload, ensure_ascii=False)}]},
    ]
    body = {"contents": contents, "generationConfig": {"temperature": 0.2}}
    try:
        async with httpx.AsyncClient(timeout=25) as c:
            r = await c.post(url, json=body)
            r.raise_for_status()
            data = r.json()
            text = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
                .strip()
            )
            if not text:
                return None
            # Try to parse strict JSON
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                # strip code fences if present
                text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
                return json.loads(text)
    except Exception:
        return None

def validate_ai_output(ai: Dict[str, Any], form_categoria: str) -> (bool, str, Dict[str, Any]):
    try:
        next_action = (ai.get("next_action") or "").lower().strip()
        router_dec = (ai.get("router_decision") or "").lower().strip()
        if next_action != "create" or router_dec != "create":
            return False, "ask_clarify", {}

        categoria = ai.get("categoria", "").strip()
        priorita  = ai.get("priorita", "").strip()
        title     = (ai.get("normalized_title") or "").strip()

        # Hard constraints: must echo the form category & be valid
        if categoria != form_categoria or categoria not in VALID_CATEGORIES:
            return False, "categoria_mismatch", {}

        if priorita not in VALID_PRIORITIES:
            return False, "priorita_invalid", {}

        fields = ai.get("monday_fields", {}) or {}
        # Minimal presence
        if not title:
            return False, "title_missing", {}

        norm = {
            "title": title,
            "categoria": categoria,
            "priorita": priorita,
            "fields": {
                "Descrizione Dettagliata": fields.get("Descrizione Dettagliata", ""),
                "Allegati": fields.get("Allegati", []) or [],
                "Link_of_the_record": fields.get("Link_of_the_record", ""),
                "Email": fields.get("Email", {"email": "", "text": ""}),
            },
        }
        return True, "", norm
    except Exception:
        return False, "invalid_ai_response", {}

# -----------------------------
# Monday.com
# -----------------------------
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

# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI(title="Aessefin Ticket Router", version="1.2.0")

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/webhook")
async def webhook(req: Request, x_forms_secret: Optional[str] = Header(None)):
    if x_forms_secret != FORMS_SHARED_SECRET:
        raise HTTPException(status_code=401, detail="bad secret")

    p = await req.json()
    # Google Form fields (from your form)
    description    = (p.get("description") or "").strip()
    categoria      = (p.get("categoria") or "").strip()     # always provided by the form
    priorita       = (p.get("priorita")  or "").strip()     # always provided by the form
    reporter_email = (p.get("email")     or "").strip()
    link_of_record = (p.get("link_of_record") or "").strip()
    attachments    = p.get("attachments") or []

    # ---------- Try Gemini ----------
    ai_decision = await call_gemini({
        "description": description,
        "categoria": categoria,
        "priorita": priorita,
        "email": reporter_email,
        "link_of_record": link_of_record,
        "attachments": attachments,
    })

    if ai_decision:
        accepted, reason, norm = validate_ai_output(ai_decision, categoria)
        if accepted:
            ai_title = norm["title"]
            final_title = build_title_with_category(categoria, ai_title)

            fields = norm["fields"]
            col_vals: Dict[str, Any] = {}
            if COL_DESCRIPTION:
                col_vals[COL_DESCRIPTION] = fields.get("Descrizione Dettagliata", "") or description

            if COL_EMAIL:
                email_obj = fields.get("Email") or {}
                col_vals[COL_EMAIL] = {
                    "email": email_obj.get("email", reporter_email),
                    "text":  email_obj.get("text",  reporter_email),
                }

            if COL_CATEGORY:
                col_vals[COL_CATEGORY] = {"label": categoria}

            if COL_PRIORITY:
                col_vals[COL_PRIORITY] = {"label": priorita}

            if COL_ATTACHMENTS:
                col_vals[COL_ATTACHMENTS] = "\n".join(filter_drive_links(fields.get("Allegati") or attachments))

            if COL_LINK_LONGTXT:
                col_vals[COL_LINK_LONGTXT] = fields.get("Link_of_the_record", link_of_record)

            created = await monday_create(final_title, col_vals)
            return {"ok": True, "source": "gemini", "used_title": final_title, "created": created}

        # AI rejected
        raise HTTPException(status_code=400, detail=f"Rejected by AI: {reason}")

    # ---------- Fallback (no Gemini / error) ----------
    words = len(strip_any_leading_category(description).split())
    is_ops = categoria.upper().startswith("OPERATIONS")
    if words < 10 and not is_ops:
        raise HTTPException(status_code=400, detail="Rejected: description too short/vague")

    core = first_sentence(description)
    final_title = build_title_with_category(categoria, core)

    col_vals: Dict[str, Any] = {}
    if COL_DESCRIPTION:
        col_vals[COL_DESCRIPTION] = description
    if COL_EMAIL:
        col_vals[COL_EMAIL] = {"email": reporter_email, "text": reporter_email}
    if COL_CATEGORY:
        col_vals[COL_CATEGORY] = {"label": categoria}
    if COL_PRIORITY:
        col_vals[COL_PRIORITY] = {"label": priorita}
    if COL_ATTACHMENTS and attachments:
        col_vals[COL_ATTACHMENTS] = "\n".join(filter_drive_links(attachments))
    if COL_LINK_LONGTXT and link_of_record:
        col_vals[COL_LINK_LONGTXT] = link_of_record

    created = await monday_create(final_title, col_vals)
    return {"ok": True, "source": "rules", "used_title": final_title, "created": created}
