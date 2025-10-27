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

import os, json, re
from typing import Optional, Dict, Any, List
import httpx
from fastapi import FastAPI, Request, Header, HTTPException

# ---------------- Env ----------------
MONDAY_API_TOKEN = os.getenv("MONDAY_API_TOKEN", "")
MONDAY_BOARD_ID  = os.getenv("MONDAY_BOARD_ID", "")
MONDAY_GROUP_NEW = os.getenv("MONDAY_GROUP_NEW", "topics")

# Column IDs (set these to your board's column IDs)
COL_EMAIL        = os.getenv("MONDAY_COLUMN_EMAIL", "")          # email_mkw892yz
COL_CATEGORY     = os.getenv("MONDAY_COLUMN_CATEGORY", "")       # color_mkvx44c
COL_PRIORITY     = os.getenv("MONDAY_COLUMN_PRIORITY", "")       # color_mkvten2j
COL_DESCRIPTION  = os.getenv("MONDAY_COLUMN_DESCRIPTION", "")    # text_mkvtqe4a
COL_ATTACHMENTS  = os.getenv("MONDAY_COLUMN_ATTACHMENTS", "")    # text_mkw4tamw
COL_LINK_LONGTXT = os.getenv("MONDAY_COLUMN_LINK_LONGTEXT", "")  # long_text_mkw5c6jg

FORMS_SHARED_SECRET = os.getenv("FORMS_SHARED_SECRET", "forms-shared-secret-123")
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY", "")   # optional
OPENAI_MODEL        = "gpt-4o-mini"

if not MONDAY_API_TOKEN or not MONDAY_BOARD_ID:
    raise RuntimeError("Set MONDAY_API_TOKEN and MONDAY_BOARD_ID.")

# ---------------- Constants ----------------
ALLOWED_CATEGORIES = {
    "CRM",
    "AMMINISTRAZIONE",
    "CONSOLE",
    "OPERATIONS (Cambi asseganzione, etc.)",
}
# normalize allowed priorities
PRIO_MAP = {
    "URGENTE": "URGENTE",
    "ALTA": "ALTA",
    "HIGH": "ALTA",
    "MEDIA": "MEDIA",
    "MEDIUM": "MEDIA",
    "BASSA": "BASSA",
    "LOW": "BASSA",
}

VAGUE_PATTERNS = [
    r"\bi have (an )?issue\b",
    r"\bsomething (is )?not working\b",
    r"\bplease fix\b",
    r"^help\b",
    r"\bhelp\b$",
    r"\bissue\b$",
    r"\bproblem\b$",
    r"\bnot working\b",
    r"\bfix this\b",
    r"\bnon funziona\b",
    r"\bqualcosa non funziona\b",
]

# ---------------- App ----------------
app = FastAPI(title="Aessefin Ticket Router")


# ---------------- Helpers ----------------
def word_count(text: str) -> int:
    return len([w for w in re.findall(r"\b\w+\b", text)])

def is_vague(text: str) -> bool:
    t = (text or "").lower().strip()
    if len(t.split()) < 10:
        return True
    return any(re.search(p, t) for p in VAGUE_PATTERNS)

def normalize_title(desc: str, categoria: str, limit: int = 30) -> str:
    base = (re.split(r"[.!?\n]", (desc or "").strip())[0] or desc or "").strip()
    words = base.split()
    if len(words) > 6:
        base = " ".join(words[:6])
    base = base[:1].upper() + base[1:] if base else "Ticket"
    if categoria:
        base = f"{categoria}: {base}"
    return (base[:limit-1] + "…") if len(base) > limit else base

def normalize_priority(prio: str) -> str:
    if not prio:
        return ""
    return PRIO_MAP.get(prio.strip().upper(), prio.strip().upper())

def filter_drive_links(urls: Optional[List[str]]) -> List[str]:
    return [u for u in (urls or []) if isinstance(u, str) and "drive.google.com" in u]

def is_operations(categoria: str) -> bool:
    return (categoria or "").upper().startswith("OPERATIONS")

async def openai_title(desc: str, categoria: str) -> Optional[str]:
    if not OPENAI_API_KEY:
        return None
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": "Generate a concise Italian IT ticket title <= 30 chars. No emojis."},
            {"role": "user", "content": f"Categoria: {categoria}\nDescrizione: {desc}"},
        ],
        "temperature": 0.2,
    }
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            r.raise_for_status()
            title = r.json()["choices"][0]["message"]["content"].strip()
            return (title[:29] + "…") if len(title) > 30 else title
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


# ---------------- Webhook ----------------
@app.post("/webhook")
async def webhook(req: Request, x_forms_secret: Optional[str] = Header(None)):
    if x_forms_secret != FORMS_SHARED_SECRET:
        raise HTTPException(status_code=401, detail="bad secret")

    p = await req.json()

    description    = (p.get("description") or "").strip()
    categoria      = (p.get("categoria") or "").strip()
    priorita_in    = (p.get("priorita")  or "").strip()
    reporter_email = (p.get("email")     or "").strip()
    link_of_record = (p.get("link_of_record") or "").strip()
    attachments_in = p.get("attachments") or []

    # 1) Category must be one of the allowed labels
    if categoria not in ALLOWED_CATEGORIES:
        return {"ok": False, "routed_to": "ask_clarify", "reason": f"Categoria non valida: {categoria}"}

    # 2) Gatekeeping
    ops = is_operations(categoria)
    if ops:
        # OPERATIONS: allow shorter, but still require *some* substance (>=4 words) and not purely vague
        if word_count(description) < 4 or any(re.search(p, description.lower()) for p in VAGUE_PATTERNS):
            return {"ok": False, "routed_to": "ask_clarify", "reason": "Descrizione troppo vaga per OPERATIONS"}
    else:
        if is_vague(description):
            return {"ok": False, "routed_to": "ask_clarify", "reason": "Descrizione troppo breve/vaga (<10 parole o frasi generiche)"}

    # 3) Title (LLM -> fallback)
    title = await openai_title(description, categoria) or normalize_title(description, categoria)

    # 4) Normalize priority to board labels
    priorita = normalize_priority(priorita_in)  # URGENTE | ALTA | MEDIA | BASSA (else passes as-is)

    # 5) Filter attachments (Google Drive only)
    attachments = filter_drive_links(attachments_in)

    # 6) Build Monday column values
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
        col_vals[COL_ATTACHMENTS] = ", ".join(attachments)

    if COL_LINK_LONGTXT and link_of_record:
        col_vals[COL_LINK_LONGTXT] = link_of_record

    # 7) Create the item
    created = await monday_create(item_name=title, column_values=col_vals)
    return {"ok": True, "routed_to": "create", "created": created}
