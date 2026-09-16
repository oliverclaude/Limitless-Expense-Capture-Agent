"""Extract claim fields from subject, body, and the cropped slip via xAI."""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

TICKS_PER_USD = 10_000_000_000
DEFAULT_MODEL = "grok-4.6"
INFERENCE_URL = "https://api.x.ai/v1/responses"
MANAGEMENT_BASE = "https://management-api.x.ai"


def journals_from_env() -> list[str]:
    raw = os.environ.get("JOURNALS", "Entertainment")
    return [item.strip() for item in raw.split(",") if item.strip()]


def ticks_to_usd(ticks: int | float | None) -> float | None:
    if ticks is None:
        return None
    return float(ticks) / TICKS_PER_USD


def extract_claim(
    *,
    subject: str,
    body: str,
    image_path: Path,
    proof_name: str,
    journals: list[str],
) -> dict[str, Any]:
    api_key = os.environ.get("XAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("missing env: XAI_API_KEY")
    model = os.environ.get("XAI_MODEL", "").strip() or DEFAULT_MODEL
    image_b64, mime = _image_data(image_path)
    prompt = _prompt(subject, body, journals)
    payload = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": f"data:{mime};base64,{image_b64}",
                        "detail": "high",
                    },
                    {"type": "input_text", "text": prompt},
                ],
            }
        ],
    }
    data = _post_json(INFERENCE_URL, payload, api_key)
    text = _output_text(data)
    fields = _parse_fields(text, journals)
    fields["proof_file"] = proof_name
    usage = data.get("usage") or {}
    fields["credits_used_usd"] = ticks_to_usd(usage.get("cost_in_usd_ticks"))
    fields["credits_left_usd"] = remaining_credits(fields["credits_used_usd"])
    return fields


def remaining_credits(used_usd: float | None) -> float | None:
    remote = _prepaid_balance_usd()
    if remote is not None:
        return remote
    return _local_remaining(used_usd)


def format_money(value: float | None) -> str:
    if value is None:
        return "(unknown)"
    return f"${value:.4f}"


def _image_data(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return base64.b64encode(raw).decode("ascii"), mime


def _prompt(subject: str, body: str, journals: list[str]) -> str:
    journal_list = ", ".join(journals) if journals else "(none configured)"
    return f"""You extract a South African expense claim. Return JSON only, no markdown.

Use ALL of: the slip image, the email subject, and the email body (a free-text note to a clerk).
Never invent amount, date, or VAT. If a value is not on the slip or in the note, use null.

Rules:
- claim_date: calendar date on the slip/invoice (YYYY-MM-DD). Not the email date unless the slip date is unreadable — then null.
- personal_amount: money spent from the operator's personal account.
- company_amount: money spent from the company account.
- If the note does not say whose account paid, the WHOLE amount is personal_amount and company_amount is 0.
- "My half" (or similar) means a third party paid the other half. personal_amount is the operator's half. company_amount is 0. Do not put the other half in either amount.
- vat: VAT amount only if printed on the slip or stated in the note. Never calculate VAT from a rate unless the slip already shows the VAT figure.
- currency: ZAR unless another currency is explicit.
- comment: short clerk note combining useful subject text, body, merchant name, and anything to ignore.
- journal: MUST be exactly one of these, or null if none fit: {journal_list}
- needs_review: true if amount, date, or merchant is missing, journal is null, the note conflicts with the slip, or you are unsure.
- confidence: high, medium, or low.

Email subject:
{subject}

Email body:
{body if body.strip() else "(empty)"}

JSON shape:
{{
  "claim_date": "YYYY-MM-DD or null",
  "personal_amount": number or null,
  "company_amount": number or null,
  "vat": number or null,
  "currency": "ZAR",
  "merchant": "string or null",
  "comment": "string",
  "journal": "string or null",
  "confidence": "high|medium|low",
  "needs_review": true,
  "review_reasons": ["..."]
}}
"""


def _post_json(url: str, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
    raw = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=raw,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:800]
        raise RuntimeError(f"xAI HTTP {exc.code}: {detail}") from exc


def _get_json(url: str, api_key: str) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        method="GET",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        raise RuntimeError(f"xAI HTTP {exc.code}: {detail}") from exc


def _output_text(data: dict[str, Any]) -> str:
    if isinstance(data.get("output_text"), str) and data["output_text"].strip():
        return data["output_text"]
    chunks: list[str] = []
    for item in data.get("output") or []:
        if not isinstance(item, dict):
            continue
        for part in item.get("content") or []:
            if isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                text = part.get("text")
                if isinstance(text, str):
                    chunks.append(text)
    return "\n".join(chunks).strip()


def _parse_fields(text: str, journals: list[str]) -> dict[str, Any]:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"model did not return JSON: {text[:400]}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("model JSON was not an object")

    allowed = {j.lower(): j for j in journals}
    journal = parsed.get("journal")
    if isinstance(journal, str) and journal.strip():
        match = allowed.get(journal.strip().lower())
        journal = match
    else:
        journal = None

    personal = _as_number(parsed.get("personal_amount"))
    company = _as_number(parsed.get("company_amount"))
    if personal is None and company is None:
        whole = _as_number(parsed.get("amount"))
        if whole is not None:
            personal, company = whole, 0.0

    reasons = parsed.get("review_reasons") or []
    if not isinstance(reasons, list):
        reasons = [str(reasons)]
    reasons = [str(r) for r in reasons]

    needs = bool(parsed.get("needs_review"))
    claim_date = parsed.get("claim_date")
    if not isinstance(claim_date, str) or not claim_date.strip():
        claim_date = None
        needs = True
        reasons.append("missing claim date")
    if personal is None:
        needs = True
        reasons.append("missing amount")
    if journal is None:
        needs = True
        reasons.append("journal not in list")

    comment = parsed.get("comment")
    if not isinstance(comment, str):
        comment = ""
    merchant = parsed.get("merchant")
    if not isinstance(merchant, str) or not merchant.strip():
        merchant = None

    confidence = str(parsed.get("confidence") or "low").lower()
    if confidence not in ("high", "medium", "low"):
        confidence = "low"

    return {
        "claim_date": claim_date,
        "personal_amount": personal,
        "company_amount": 0.0 if company is None and personal is not None else company,
        "vat": _as_number(parsed.get("vat")),
        "currency": str(parsed.get("currency") or "ZAR"),
        "merchant": merchant,
        "comment": comment.strip(),
        "journal": journal,
        "confidence": confidence,
        "needs_review": needs,
        "review_reasons": reasons,
    }


def _as_number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace("R", "").replace(",", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _prepaid_balance_usd() -> float | None:
    mgmt = os.environ.get("XAI_MANAGEMENT_KEY", "").strip()
    if not mgmt:
        return None
    team_id = os.environ.get("XAI_TEAM_ID", "").strip()
    if not team_id:
        try:
            info = _get_json(f"{MANAGEMENT_BASE}/auth/management-keys/validation", mgmt)
        except RuntimeError:
            return None
        team_id = str(info.get("teamId") or info.get("scopeId") or "").strip()
    if not team_id:
        return None
    try:
        data = _get_json(f"{MANAGEMENT_BASE}/v1/billing/teams/{team_id}/prepaid/balance", mgmt)
    except RuntimeError:
        return None
    total = data.get("total") or {}
    val = total.get("val")
    if val is None:
        return None
    try:
        cents = float(str(val))
    except ValueError:
        return None
    return abs(cents) / 100.0


def _local_remaining(used_usd: float | None) -> float | None:
    start_raw = os.environ.get("XAI_PREPAID_USD", "").strip()
    path = _ledger_path()
    stored: float | None = None
    if path.is_file():
        try:
            stored = float(json.loads(path.read_text(encoding="utf-8")).get("remaining_usd"))
        except (OSError, ValueError, json.JSONDecodeError, TypeError):
            stored = None
    if stored is None:
        if not start_raw:
            return None
        try:
            stored = float(start_raw)
        except ValueError:
            return None
    if used_usd is None:
        return stored
    remaining = max(0.0, stored - used_usd)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"remaining_usd": remaining}), encoding="utf-8")
    except OSError:
        pass
    return remaining


def _ledger_path() -> Path:
    env_file = os.environ.get("ENV_FILE", "").strip()
    if env_file:
        return Path(env_file).resolve().parent / "prepaid.json"
    return Path(".prepaid.json")
