"""Extract claim fields from subject, body, and the cropped slip via xAI."""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

TICKS_PER_USD = 10_000_000_000
DEFAULT_MODEL = "grok-4.6"
INFERENCE_URL = "https://api.x.ai/v1/responses"
MANAGEMENT_BASE = "https://management-api.x.ai"


def _csv_env(name: str, default: str = "") -> list[str]:
    raw = os.environ.get(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def journals_from_env() -> list[str]:
    return _csv_env("JOURNALS", "Entertainment")


def vat_journals_from_env() -> list[str]:
    return _csv_env("VAT_JOURNALS", "")


def apply_vat_policy(fields: dict[str, Any], vat_journals: list[str] | None = None) -> dict[str, Any]:
    """Only journals on VAT_JOURNALS may claim VAT. Others get 0 (SA entertainment, etc.)."""
    allowed = {name.lower() for name in (vat_journals if vat_journals is not None else vat_journals_from_env())}
    journal = fields.get("journal")
    if not isinstance(journal, str) or journal.lower() not in allowed:
        fields["vat"] = 0.0
    return fields


def ticks_to_usd(ticks: int | float | None) -> float | None:
    if ticks is None:
        return None
    return float(ticks) / TICKS_PER_USD


def slip_corners(path: Path) -> list[tuple[float, float]] | None:
    """Ask xAI for the slip's four corners as fractions 0-1 of the EXIF-upright image."""
    api_key = os.environ.get("XAI_API_KEY", "").strip()
    if not api_key:
        return None
    model = os.environ.get("XAI_MODEL", "").strip() or DEFAULT_MODEL
    preview, mime = _preview_jpeg(path)
    prompt = """This photo contains a paper till slip or invoice. It may be rotated or shot at an angle.
Return JSON only, no markdown.
Give the four corners of the PAPER (include amounts and header; do not cut off text; exclude table/background).
Coordinates are fractions of image width and height, 0 to 1.
Order: top-left, top-right, bottom-right, bottom-left of the paper rectangle in the photo.

{"tl": [x, y], "tr": [x, y], "br": [x, y], "bl": [x, y]}
"""
    payload = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": f"data:{mime};base64,{base64.b64encode(preview).decode('ascii')}",
                        "detail": "high",
                    },
                    {"type": "input_text", "text": prompt},
                ],
            }
        ],
    }
    data = _post_json(INFERENCE_URL, payload, api_key)
    text = _output_text(data)
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        parsed = json.loads(match.group(0) if match else text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    points: list[tuple[float, float]] = []
    for key in ("tl", "tr", "br", "bl"):
        pair = parsed.get(key)
        if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
            return None
        try:
            x, y = float(pair[0]), float(pair[1])
        except (TypeError, ValueError):
            return None
        if not (0.0 - 0.05 <= x <= 1.05 and 0.0 - 0.05 <= y <= 1.05):
            return None
        points.append((min(1.0, max(0.0, x)), min(1.0, max(0.0, y))))
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    area = (max(xs) - min(xs)) * (max(ys) - min(ys))
    if area < 0.08:
        return None
    return points


def _preview_jpeg(path: Path, max_side: int = 1280) -> tuple[bytes, str]:
    from io import BytesIO

    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((max_side, max_side))
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=85)
    return buf.getvalue(), "image/jpeg"


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
    vat_journals = vat_journals_from_env()
    prompt = _prompt(subject, body, journals, vat_journals)
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
    fields = _parse_fields(text, journals, subject)
    apply_vat_policy(fields, vat_journals)
    fields["proof_file"] = proof_name
    usage = data.get("usage") or {}
    fields["credits_used_usd"] = ticks_to_usd(usage.get("cost_in_usd_ticks"))
    fields["credits_left_usd"] = remaining_credits(fields["credits_used_usd"])
    return fields


CONFIRM_REPLIES = {"ok", "okay", "yes", "y", "proceed", "log", "logged", "confirm", "confirmed"}
SKIP_REPLIES = {"skip", "skipped", "ignore", "discard"}


def interpret_reply(reply: str, fields: dict[str, Any], journals: list[str]) -> dict[str, Any]:
    """Map an English ntfy reply onto confirm / skip / field updates."""
    compact = re.sub(r"[.!?]+$", "", reply.strip().lower()).strip()
    if compact in SKIP_REPLIES:
        return {"action": "skip", "fields": fields, "credits_used_usd": 0.0}
    if compact in CONFIRM_REPLIES:
        updated = dict(fields)
        refresh_review(updated, journals)
        updated["needs_review"] = False
        updated["review_reasons"] = []
        return {"action": "confirm", "fields": updated, "credits_used_usd": 0.0}

    api_key = os.environ.get("XAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("missing env: XAI_API_KEY")
    model = os.environ.get("XAI_MODEL", "").strip() or DEFAULT_MODEL
    snapshot = {
        "claim_date": fields.get("claim_date"),
        "personal_amount": fields.get("personal_amount"),
        "company_amount": fields.get("company_amount"),
        "vat": fields.get("vat"),
        "merchant": fields.get("merchant"),
        "comment": fields.get("comment"),
        "journal": fields.get("journal"),
        "review_reasons": fields.get("review_reasons"),
    }
    prompt = f"""The operator replied in English about this expense claim. Return JSON only.

Current claim:
{json.dumps(snapshot, default=str)}

Journals they may use: {", ".join(journals) or "(none)"}

Operator reply:
{reply}

Decide action:
- skip: they do not want this logged
- confirm: log as-is (ok, yes, looks good)
- update: they changed a field (merchant, half, journal, amounts, date)

JSON:
{{
  "action": "skip|confirm|update",
  "claim_date": "YYYY-MM-DD or null",
  "personal_amount": number or null,
  "company_amount": number or null,
  "vat": number or null,
  "merchant": "string or null",
  "comment": "string or null",
  "journal": "string or null"
}}
Keep fields they did not change. "My half" means personal is half of the spend and company is 0.
"""
    payload = {"model": model, "input": prompt}
    data = _post_json(INFERENCE_URL, payload, api_key)
    text = _output_text(data)
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        action_obj = json.loads(match.group(0) if match else text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"model did not return JSON: {text[:400]}") from exc
    if not isinstance(action_obj, dict):
        action_obj = {}
    action = str(action_obj.get("action") or "update").lower()
    if action not in ("skip", "confirm", "update"):
        action = "update"
    merged = dict(fields)
    if action_obj.get("claim_date"):
        merged["claim_date"] = str(action_obj["claim_date"]).strip()
    for key in ("personal_amount", "company_amount", "vat"):
        number = _as_number(action_obj.get(key))
        if number is not None:
            merged[key] = number
    if isinstance(action_obj.get("merchant"), str) and action_obj["merchant"].strip():
        merged["merchant"] = action_obj["merchant"].strip()
    if isinstance(action_obj.get("journal"), str) and action_obj["journal"].strip():
        merged["journal"] = action_obj["journal"].strip()
    refresh_review(merged, journals)
    if action == "confirm":
        merged["needs_review"] = False
        merged["review_reasons"] = []
    used = ticks_to_usd((data.get("usage") or {}).get("cost_in_usd_ticks"))
    merged["credits_used_usd"] = used
    merged["credits_left_usd"] = remaining_credits(used)
    return {"action": action, "fields": merged, "credits_used_usd": used}


def refresh_review(fields: dict[str, Any], journals: list[str]) -> dict[str, Any]:
    allowed = {item.lower(): item for item in journals}
    journal = fields.get("journal")
    if isinstance(journal, str) and journal.strip():
        fields["journal"] = allowed.get(journal.strip().lower())
    else:
        fields["journal"] = None
    reasons: list[str] = []
    if not fields.get("claim_date"):
        reasons.append("missing claim date")
    if fields.get("personal_amount") is None:
        reasons.append("missing amount")
    if not fields.get("merchant"):
        reasons.append("missing merchant")
    if not fields.get("journal"):
        reasons.append("journal not in list")
    apply_vat_policy(fields)
    fields["review_reasons"] = reasons
    fields["needs_review"] = bool(reasons)
    return fields


def notify_reasons(fields: dict[str, Any], crop_status: str) -> list[str]:
    reasons: list[str] = []
    for item in fields.get("review_reasons") or []:
        text = str(item)
        if text not in reasons:
            reasons.append(text)
    if not fields.get("merchant") and "missing merchant" not in reasons:
        reasons.append("missing merchant")
    if crop_status.startswith("failed") and crop_status not in reasons:
        reasons.append(crop_status)
    return reasons


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


def comment_from_subject(subject: str) -> str:
    text = subject.strip()
    if text.lower().startswith("claim:"):
        text = text[6:].strip()
    text = re.sub(r"^(slip|invoice|receipt|bill)\s+for\s+", "", text, flags=re.IGNORECASE).strip()
    if not text:
        return ""
    return " ".join(word.capitalize() if word.islower() else word for word in text.split())


def _is_noise_reason(reason: str) -> bool:
    text = reason.lower()
    if "future" in text:
        return True
    if "tip" in text:
        return True
    if "handwrit" in text and "printed" in text:
        return True
    if "printed" in text and ("claimed" in text or " vs " in text):
        return True
    return False


def _prompt(subject: str, body: str, journals: list[str], vat_journals: list[str] | None = None) -> str:
    journal_list = ", ".join(journals) if journals else "(none configured)"
    if vat_journals is None:
        vat_journals = vat_journals_from_env()
    vat_list = ", ".join(vat_journals) if vat_journals else "(none — no journal may claim VAT)"
    today = date.today().isoformat()
    return f"""You extract a South African expense claim. Return JSON only, no markdown.

Use ALL of: the slip image, the email subject, and the email body (a free-text note to a clerk).
Never invent amount, date, or VAT. If a value is not on the slip or in the note, use null.
Today's date is {today}. A slip dated 2026 is not "in the future".

Rules:
- claim_date: calendar date on the slip/invoice (YYYY-MM-DD). Not the email date unless the slip date is unreadable — then null.
- personal_amount: money spent from the operator's personal account.
- company_amount: money spent from the company account.
- If the note does not say whose account paid, the WHOLE amount is personal_amount and company_amount is 0.
- "My half" (or similar) means a third party paid the other half. personal_amount is the operator's half. company_amount is 0. Do not put the other half in either amount.
- If the slip has a handwritten total or tip (for example Total: R130 next to a printed Total Incl. R118), the handwritten figure is the amount spent. Use it. Do not set needs_review for printed total vs handwritten tip.
- vat: claimable VAT only. If the chosen journal is not in this list, vat MUST be 0 even if the slip prints VAT (South Africa: entertainment is not VAT-claimable): {vat_list}
- If the journal IS on that list, copy VAT only if printed on the slip or stated in the note. Keep that printed VAT even when a tip is added. Never calculate VAT from a rate unless the slip already shows the VAT figure.
- currency: ZAR unless another currency is explicit.
- comment: 2–5 words, the purpose from the subject after CLAIM: (drop leading "Slip for" / "Invoice for"). Example: "Client Drinks". Do not summarise the merchant, mall, or amounts.
- journal: MUST be exactly one of these, or null if none fit: {journal_list}
- needs_review: true only if amount, date, or merchant is missing, journal is null, or the note truly conflicts with the slip (not tip vs printed total).
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


def _parse_fields(text: str, journals: list[str], subject: str = "") -> dict[str, Any]:
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
    reasons = [str(r) for r in reasons if not _is_noise_reason(str(r))]

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

    if not reasons:
        needs = False

    comment = comment_from_subject(subject)
    if not comment:
        raw_comment = parsed.get("comment")
        comment = raw_comment.strip() if isinstance(raw_comment, str) else ""
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
