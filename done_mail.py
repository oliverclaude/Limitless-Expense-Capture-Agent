"""Done/Skipped reply to the configured work address only."""

from __future__ import annotations

import smtplib
import ssl
from email.message import EmailMessage
from typing import Any


def done_body(fields: dict[str, Any], *, skipped: bool) -> str:
    heading = "Skipped." if skipped else "Done."
    bits: list[str] = []
    amount = fields.get("personal_amount")
    company = fields.get("company_amount")
    currency = str(fields.get("currency") or "ZAR").upper()
    if amount is not None:
        bits.append(_money(amount, currency))
    if company not in (None, 0, 0.0):
        bits.append(f"company {_money(company, currency)}")
    merchant = fields.get("merchant")
    if merchant:
        bits.append(str(merchant))
    comment = fields.get("comment")
    if comment:
        bits.append(str(comment))
    proof = fields.get("proof_file")
    if proof:
        bits.append(str(proof))
    if not bits:
        return heading + "\n"
    return heading + "\n\n" + " · ".join(bits) + "\n"


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
    msg["Subject"] = subject
    if in_reply_to.strip():
        msg["In-Reply-To"] = in_reply_to.strip()
        msg["References"] = in_reply_to.strip()
    msg.set_content(done_body(fields, skipped=skipped))
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(host, port, timeout=30, context=context) as smtp:
        smtp.login(user, password)
        smtp.send_message(msg)


def _money(amount: float, currency: str) -> str:
    value = f"{amount:.2f}".rstrip("0").rstrip(".")
    if currency == "ZAR":
        return f"R{value}"
    return f"{value} {currency}"
