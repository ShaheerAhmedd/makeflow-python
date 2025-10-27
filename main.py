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
import smtplib
from typing import Optional, Dict, Any, List, Tuple

import httpx
from email.mime.text import MIMEText
from fastapi import FastAPI, Request, Header, HTTPException


# =============================
# Environment / Config
# =============================
FORMS_SHARED_SECRET = os.getenv("FORMS_SHARED_SECRET", "forms-shared-secret-123")

# Gemini (Google Generative Language API)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# Monday.com
MONDAY_API_TOKEN = os.getenv("MONDAY_API_TOKEN", "")
MONDAY_BOARD_ID = os.getenv("MONDAY_BOARD_ID", "")
MONDAY_GROUP_NEW = os.getenv("MONDAY_GROUP_NEW", "topics")

# Column IDs
COL_EMAIL = os.getenv("MONDAY_COLUMN_EMAIL", "")
COL_CATEGORY = os.getenv("MONDAY_COLUMN_CATEGORY", "")
COL_PRIORITY = os.getenv("MONDAY_COLUMN_PRIORITY", "")
COL_DESCRIPTION = os.getenv("MONDAY_COLUMN_DESCRIPTION", "")
COL_ATTACHMENTS = os.getenv("MONDAY_COLUMN_ATTACHMENTS", "")
COL_LINK_LONGTXT = os.getenv("MONDAY_COLUMN_LINK_LONGTEXT", "")

if not MONDAY_API_TOKEN or not MONDAY_BOARD_ID:
    raise RuntimeError("Set MONDAY_API_TOKEN and MONDAY_BOARD_ID")

VALID_CATEGORIES = {
    "CRM", "AMMINISTRAZIONE", "CONSOLE", "OPERATIONS (Cambi asseganzione, etc.)"
}
VALID_PRIORITIES = {"URGENTE", "MEDIA", "ALTA", "BASSA"}

# Rejection email (separate mailbox)
SMTP_HOST_REJECT = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT_REJECT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER_REJECT = os.getenv("SMTP_USER_REJECT", "")
SMTP_PASS_REJECT = os.getenv("SMTP_PASS_REJECT", "")
REJECT_FROM_NAME = os.getenv("REJECT_FROM_NAME", "Team IT")
REJECT_SUBJECT = os.getenv(
    "REJECT_SUBJECT",
    "Ticket non aperto - Informazioni insufficienti",
)
FORM_URL = os.getenv("FORM_URL", "https://shorturl.at/2Mqxc")
REJECT_LOGO_URL = os.getenv(
    "REJECT_LOGO_URL",
    "https://prestiti.aessefin.it/wp-content/uploads/2024/11/aessefin_nuovo_logo2-removebg-preview.png",
)


# =============================
# Make.com Prompt (system)
# =============================
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


# =============================
# Helper functions
# =============================
def strip_any_leading_category(text: str) -> str:
    """Remove any leading category prefix like 'CRM:' from a string."""
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


def first_sentence(text: str) -> str:
    """First sentence (or whole string) without a leading category; fallback to 'Ticket'."""
    if not text:
        return "Ticket"
    clean = strip_any_leading_category(text)
    sent = re.split(r"[.!?\n]", clean, maxsplit=1)[0].strip()
    return sent or "Ticket"


def filter_drive_links(urls: List[str]) -> List[str]:
    """Only Google Drive/Docs links for the Allegati field."""
    out = []
    for u in urls or []:
        if isinstance(u, str) and ("drive.google.com" in u or "docs.google.com" in u):
            out.append(u)
    return out


# =============================
# Rejection email (HTML)
# =============================
def send_reject_email(to_email: str) -> bool:
    """
    Send the exact rejection email using the dedicated mailbox.
    """
    if not (SMTP_USER_REJECT and SMTP_PASS_REJECT and to_email):
        return False

    html = f"""<!doctype html>
<html>
  <body style="font-family:Arial,Helvetica,sans-serif; color:#222; line-height:1.5;">
    <p>Ciao,</p>

    <p>abbiamo ricevuto la tua richiesta tramite il modulo di apertura 📥</p>

    <p>Purtroppo, non è stato possibile procedere con l’apertura del ticket in quanto
       non sono state fornite informazioni sufficienti nei seguenti campi:<br/>
       Ti invitiamo a compilare nuovamente il modulo in modo completo, specificando nel
       dettaglio il problema e la priorità.</p>

    <p>Puoi accedere al modulo tramite questo link:
       <a href="{FORM_URL}" target="_blank">{FORM_URL}</a> ✍️</p>

    <p>Cordiali saluti,<br/>
       {REJECT_FROM_NAME}</p>

    <img src="{REJECT_LOGO_URL}" alt="Company Logo"
         style="width:150px; height:auto; display:block; margin-top:20px;" />
  </body>
</html>"""

    msg = MIMEText(html, "html", "utf-8")
    msg["Subject"] = REJECT_SUBJECT
    msg["From"] = f"{REJECT_FROM_NAME} <{SMTP_USER_REJECT}>"
    msg["To"] = to_email

    try:
        with smtplib.SMTP(SMTP_HOST_REJECT, SMTP_PORT_REJECT) as s:
            s.starttls()
            s.login(SMTP_USER_REJECT, SMTP_PASS_REJECT)
            s.sendmail(SMTP_USER_REJECT, [to_email], msg.as_string())
        return True
    except Exception:
        return False


# =============================
# Gemini call + validation
# =============================
async def call_gemini(user_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Call Gemini; return parsed JSON dict or None."""
    if not GEMINI_API_KEY:
        return None

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
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
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                # Remove possible ```json fences
                text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
                return json.loads(text)
    except Exception:
        return None


def validate_ai_output(ai: Dict[str, Any], form_categoria: str) -> Tuple[bool, str, Dict[str, Any]]:
    """Validate Gemini output against rules and return (accepted, reason, normalized)."""
    try:
        next_action = (ai.get("next_action") or "").lower().strip()
        router_dec = (ai.get("router_decision") or "").lower().strip()
        if next_action != "create" or router_dec != "create":
            return False, "ask_clarify", {}

        categoria = ai.get("categoria", "").strip()
        priorita = ai.get("priorita", "").strip()
        title = (ai.get("normalized_title") or "").strip()

        # Category must match exactly what the user selected
        if categoria != form_categoria or categoria not in VALID_CATEGORIES:
            return False, "categoria_mismatch", {}
        if priorita not in VALID_PRIORITIES:
            return False, "priorita_invalid", {}
        if not title:
            return False, "title_missing", {}

        fields = ai.get("monday_fields", {}) or {}
        norm = {
            "title": title,  # <-- use exactly the Gemini title as item_name
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


# =============================
# Monday.com
# =============================
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


# =============================
# FastAPI app
# =============================
app = FastAPI(title="Aessefin Ticket Router", version="1.4.0")


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/webhook")
async def webhook(req: Request, x_forms_secret: Optional[str] = Header(None)):
    # Authentication shared secret
    if x_forms_secret != FORMS_SHARED_SECRET:
        raise HTTPException(status_code=401, detail="bad secret")

    p = await req.json()
    description = (p.get("description") or "").strip()
    categoria = (p.get("categoria") or "").strip()
    priorita = (p.get("priorita") or "").strip()
    reporter_email = (p.get("email") or "").strip()
    link_of_record = (p.get("link_of_record") or "").strip()
    attachments = p.get("attachments") or []

    # --------------------------
    # 1) Try Gemini
    # --------------------------
    ai_decision = await call_gemini(
        {
            "description": description,
            "categoria": categoria,
            "priorita": priorita,
            "email": reporter_email,
            "link_of_record": link_of_record,
            "attachments": attachments,
        }
    )

    if ai_decision:
        accepted, reason, norm = validate_ai_output(ai_decision, categoria)
        if accepted:
            # Use Gemini's title EXACTLY for item_name
            final_title = norm["title"]

            fields = norm["fields"]
            col_vals: Dict[str, Any] = {}

            # Description
            if COL_DESCRIPTION:
                col_vals[COL_DESCRIPTION] = fields.get("Descrizione Dettagliata", "") or description

            # Email column object
            if COL_EMAIL:
                email_obj = fields.get("Email") or {}
                col_vals[COL_EMAIL] = {
                    "email": email_obj.get("email", reporter_email),
                    "text": email_obj.get("text", reporter_email),
                }

            # Category + Priority from the user (they must match)
            if COL_CATEGORY:
                col_vals[COL_CATEGORY] = {"label": categoria}
            if COL_PRIORITY:
                col_vals[COL_PRIORITY] = {"label": priorita}

            # Attachments (only Google Drive/Docs links)
            if COL_ATTACHMENTS:
                col_vals[COL_ATTACHMENTS] = "\n".join(
                    filter_drive_links(fields.get("Allegati") or attachments)
                )

            # Link of the record (long text)
            if COL_LINK_LONGTXT:
                col_vals[COL_LINK_LONGTXT] = fields.get("Link_of_the_record", link_of_record)

            created = await monday_create(final_title, col_vals)
            return {"ok": True, "source": "gemini", "used_title": final_title, "created": created}

        # Rejected by AI → email the user
        emailed = send_reject_email(reporter_email)
        return {"ok": False, "reason": f"Rejected by AI: {reason}", "email_sent": emailed}

    # --------------------------
    # 2) Fallback if Gemini not available
    # --------------------------
    # Simple rule: >= 10 words unless OPERATIONS..., otherwise reject.
    words = len(strip_any_leading_category(description).split())
    is_ops = categoria.upper().startswith("OPERATIONS")
    if words < 10 and not is_ops:
        emailed = send_reject_email(reporter_email)
        return {"ok": False, "reason": "Rejected: description too short/vague", "email_sent": emailed}

    # Fallback title: first sentence only (no category prefix) to avoid "CRM: CRM: ..."
    final_title = first_sentence(description)

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

