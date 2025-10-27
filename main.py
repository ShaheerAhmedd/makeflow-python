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

import os
import json
import smtplib
import httpx
from email.mime.text import MIMEText
from typing import Optional, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# -----------------------------
# Config
# -----------------------------
class Settings(BaseSettings):
    # Email (Gmail via SMTP with App Password)
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: str = ""            # your Gmail address
    SMTP_PASS: str = ""            # app password (NOT your login password)
    EMAIL_TO_DEFAULT: str = ""     # fallback recipient for clarify path

    # Monday.com
    MONDAY_API_TOKEN: str = ""
    MONDAY_BOARD_ID: Optional[int] = None
    MONDAY_GROUP_ID: Optional[str] = None  # e.g., "topics"

    # Optional OpenAI for AI-assisted processing
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

# -----------------------------
# Data models
# -----------------------------
class Incoming(BaseModel):
    sender: Optional[str] = None
    subject: Optional[str] = None
    message: str = Field(..., description="Free text from the webhook")
    metadata: Optional[dict] = None

class AIResult(BaseModel):
    intent: Literal["clarify", "create"]
    summary: str
    priority: Literal["low", "medium", "high"] = "medium"
    reason: Optional[str] = None

# -----------------------------
# Classifier (rule-based + optional LLM)
# -----------------------------
KEYWORDS_CREATE = [
    "create", "open ticket", "raise ticket", "new item",
    "add task", "setup", "provision", "access", "request",
]

KEYWORDS_CLARIFY = [
    "clarify", "question", "unsure", "help", "details", "explain",
]

def classify_rule_based(text: str) -> AIResult:
    t = text.lower()
    if any(k in t for k in KEYWORDS_CREATE):
        return AIResult(
            intent="create",
            summary=text.strip()[:160],
            priority="medium",
            reason="Matched create keywords"
        )
    if "urgent" in t or "asap" in t:
        return AIResult(
            intent="create",
            summary=text.strip()[:160],
            priority="high",
            reason="Urgency markers detected"
        )
    if any(k in t for k in KEYWORDS_CLARIFY):
        return AIResult(
            intent="clarify",
            summary=text.strip()[:160],
            priority="low",
            reason="Matched clarify keywords"
        )
    # default: ask for clarification
    return AIResult(
        intent="clarify",
        summary=text.strip()[:160],
        priority="low",
        reason="No strong pattern; defaulting to clarify"
    )

async def classify_with_llm(text: str) -> Optional[AIResult]:
    """Optional: use OpenAI if OPENAI_API_KEY is present. Returns None if not configured."""
    if not settings.OPENAI_API_KEY:
        return None
    # Minimal, dependency-free OpenAI call via httpx
    system = (
        "You are a routing assistant. "
        "Read the user's message and decide whether we should 'create' an item in monday.com "
        "or 'clarify' via email first. Return strict JSON with keys: intent, summary, priority, reason. "
        "intent must be 'create' or 'clarify'. priority must be 'low'|'medium'|'high'."
    )
    user = text
    payload = {
        "model": settings.OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
    }
    headers = {
        "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post("https://api.openai.com/v1/chat/completions",
                                     headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            return AIResult(**parsed)
    except Exception as e:
        # Fall back silently to rule-based
        return None

# -----------------------------
# Actions
# -----------------------------
def send_email(subject: str, body: str, to: Optional[str] = None):
    to_addr = to or settings.EMAIL_TO_DEFAULT
    if not to_addr:
        raise RuntimeError("EMAIL_TO_DEFAULT not set and no 'to' provided.")

    msg = MIMEText(body, "plain")
    msg["Subject"] = subject
    msg["From"] = settings.SMTP_USER
    msg["To"] = to_addr

    with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
        server.starttls()
        server.login(settings.SMTP_USER, settings.SMTP_PASS)
        server.sendmail(settings.SMTP_USER, [to_addr], msg.as_string())

async def create_monday_item(item_name: str, column_values: dict):
    if not settings.MONDAY_API_TOKEN or not settings.MONDAY_BOARD_ID:
        raise RuntimeError("Missing MONDAY_API_TOKEN or MONDAY_BOARD_ID")
    query = """
    mutation ($board_id: Int!, $group_id: String, $item_name: String!, $column_values: JSON) {
      create_item (board_id: $board_id, group_id: $group_id, item_name: $item_name, column_values: $column_values) {
        id
        name
      }
    }
    """
    variables = {
        "board_id": settings.MONDAY_BOARD_ID,
        "group_id": settings.MONDAY_GROUP_ID,
        "item_name": item_name,
        "column_values": json.dumps(column_values) if column_values else None,
    }
    headers = {
        "Authorization": settings.MONDAY_API_TOKEN,
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post("https://api.monday.com/v2",
                              headers=headers, json={"query": query, "variables": variables})
        r.raise_for_status()
        return r.json()

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Makeflow-Python", version="1.0.0")

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/webhook")
async def webhook(payload: Incoming):
    text = payload.message or ""
    # Try LLM first (if configured), else rule-based
    ai = await classify_with_llm(text)
    if ai is None:
        ai = classify_rule_based(text)

    if ai.intent == "clarify":
        subject = payload.subject or f"[Clarify Needed] {ai.summary[:60]}"
        body = f"""We need clarification on the request.

Reason: {ai.reason}
Priority: {ai.priority}

Sender: {payload.sender or 'unknown'}
Subject: {payload.subject or '(none)'}
Message:
{payload.message}
"""
        try:
            send_email(subject, body)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Email failed: {e}")
        return {"routed_to": "email", "ai": ai.model_dump()}

    else:  # create
        item_name = payload.subject or ai.summary[:80]
        column_values = {
            "text": ai.summary,
            "status": {"label": ai.priority.capitalize()},
            "email": {"email": payload.sender or "", "text": payload.sender or ""},
        }
        try:
            result = await create_monday_item(item_name, column_values)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Monday.com failed: {e}")
        return {"routed_to": "monday.com", "api_result": result, "ai": ai.model_dump()}
