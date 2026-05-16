"""Email notification tool for tradingo-mcp."""

from __future__ import annotations

import os

from tradingo.notifications.email import send_email as _send_email
from tradingo.settings import SMTPConfig


def send_email(subject: str, body: str) -> dict:
    """Send an email to TP_MCP_EMAIL_RECIPIENTS (semicolon-separated list).

    Initialises SMTP env vars via SMTPConfig.from_env().to_env() before sending.
    """
    SMTPConfig.from_env().to_env()

    raw = os.environ.get("TP_MCP_EMAIL_RECIPIENTS", "")
    recipients = [r.strip() for r in raw.split(";") if r.strip()]
    if not recipients:
        raise ValueError("TP_MCP_EMAIL_RECIPIENTS is not set or empty")

    for recipient in recipients:
        _send_email(body=body, subject=subject, recipient=recipient)

    return {"sent_to": recipients, "subject": subject}
