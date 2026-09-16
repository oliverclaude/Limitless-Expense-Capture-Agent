"""Done/Skipped reply to the configured work address only."""

from __future__ import annotations

import smtplib
import ssl
from email.message import EmailMessage
from typing import Any

from sheet import HEADERS

_FIELD_KEYS = {
    "claim date": "claim_date",
    "personal": "personal_amount",
    "company": "company_amount",
    "vat": "vat",
    "comment": "comment",
    "journal": "journal",
    "proof file": "proof_file",
}


def reply_subject(subject: str) -> str:
    text = subject.strip() or "CLAIM:"
    if text.lower().startswith("re:"):
        return text
    return f"Re: {text}"


def done_body(fields: dict[str, Any], *, skipped: bool) -> str:
    heading = "Skipped." if skipped else "Done."
    lines = [heading, ""]
    for header in HEADERS:
        key = _FIELD_KEYS.get(header, header)
        value = fields.get(key)
        if value is None or value == "":
            shown = "(none)"
        else:
            shown = str(value)
        lines.append(f"{header}: {shown}")
    return "\n".join(lines) + "\n"


def send_done_mail(
    *,
    host: str,
    port: int,
    user: str,
    password: str,
    work_email: str,
    subject: str,
    fields: dict[str, Any],
    skipped: bool,
    in_reply_to: str = "",
) -> None:
    if not work_email.strip():
        raise RuntimeError("missing env: WORK_EMAIL")
    msg = EmailMessage()
    msg["From"] = user
    msg["To"] = work_email.strip()
    msg["Subject"] = reply_subject(subject)
    if in_reply_to.strip():
        msg["In-Reply-To"] = in_reply_to.strip()
        msg["References"] = in_reply_to.strip()
    msg.set_content(done_body(fields, skipped=skipped))
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(host, port, timeout=30, context=context) as smtp:
        smtp.login(user, password)
        smtp.send_message(msg)
