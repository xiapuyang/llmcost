"""Fetch and apply LMSYS Chatbot Arena ELO scores from arena.ai."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import httpx

from llmcost.pricing.config import ARENA_IMAGE_WEIGHTS, ARENA_TEXT_WEIGHTS, ARENA_VISION_WEIGHTS, CONTEXT_DRIFT_BUFFER
from llmcost.pricing.models import ModelRecord

logger = logging.getLogger(__name__)

_CACHE_PATH = Path(__file__).parent.parent / "data" / "arena_scores.json"

_ARENA_URLS: dict[str, str] = {
    "text": "https://arena.ai/leaderboard/text",
    "coding": "https://arena.ai/leaderboard/code",
    "vision": "https://arena.ai/leaderboard/vision",
    "text_to_image": "https://arena.ai/leaderboard/text-to-image",
    "image_edit": "https://arena.ai/leaderboard/image-edit",
}

# Keys in arena_scores.json that are not category score dicts
_NON_SCORE_KEYS = {"arena_prices"}

# Regex patterns for parsing arena.ai leaderboard HTML
_SCORE_RE = re.compile(r'text-sm">(\d{3,4})</span><span class="text-tertiary')
_TITLE_RE = re.compile(r'title="([^"]{3,80})"')
# Price: "$5<!-- --> / <!-- -->$25"
_PRICE_RE = re.compile(r'\$(\d+(?:\.\d+)?)<!-- --> / <!-- -->\$(\d+(?:\.\d+)?)')
# Context: last text-sm span in a row, e.g. "1M", "128K", "N/A"
_CTX_RE = re.compile(r'<span class="text-sm">(\d+(?:\.\d+)?[KM]|N/A)</span></td></tr>')

# Stable aliases → Arena name, for model IDs that don't match via substring
_MANUAL_MAPPINGS: dict[str, str] = {
    "deepseek-chat": "deepseek-v3.2",
    "deepseek-reasoner": "deepseek-r1",
}


def _parse_context_str(ctx: str | None) -> str | None:
    """Normalize a context string like '1M' or '128K', returning None for N/A."""
    if not ctx or ctx == "N/A":
        return None
    return ctx


def _ctx_str_to_tokens(ctx: str) -> float | None:
    """Convert arena context string to approximate token count.

    Uses 1000-based multipliers matching arena.ai's display convention.
    """
    ctx = ctx.strip()
    if ctx.endswith("M"):
        return float(ctx[:-1]) * 1_000_000
    if ctx.endswith("K"):
        return float(ctx[:-1]) * 1_000
    return None


def _fmt_ctx(tokens: int | None) -> str | None:
    """Format OpenRouter token count the same way arena.ai displays it."""
    if tokens is None:
        return None
    if tokens >= 1_000_000:
        return f"{tokens // 1_000_000}M"
    return f"{tokens // 1_024}K"


def _fetch_category(
    category: str, url: str, timeout: int
) -> tuple[dict[str, int], dict[str, dict]]:
    """Fetch ELO scores and prices for one Arena leaderboard category.

    Args:
        category: Category label (for logging).
        url: Leaderboard URL to scrape.
        timeout: HTTP request timeout in seconds.

    Returns:
        Tuple of (scores, prices) where scores maps Arena model name → ELO
        score and prices maps Arena model name → {input, output, context}.
    """
    try:
        resp = httpx.get(url, timeout=timeout, follow_redirects=True)
        resp.raise_for_status()
        html = resp.text
    except Exception as exc:
        logger.warning("arena.ai %s fetch failed: %s", category, exc)
        return {}, {}

    scores: dict[str, int] = {}
    prices: dict[str, dict] = {}

    for score_match in _SCORE_RE.finditer(html):
        pos = score_match.start()
        window_back = html[max(0, pos - 1200) : pos]
        titles = _TITLE_RE.findall(window_back)
        if not titles:
            continue
        name = titles[-1]
        score = int(score_match.group(1))
        if name in scores:
            continue
        scores[name] = score

        # Look forward ~900 chars for price and context in the same table row
        window_fwd = html[pos : pos + 900]
        price_m = _PRICE_RE.search(window_fwd)
        ctx_m = _CTX_RE.search(window_fwd)
        prices[name] = {
            "input": float(price_m.group(1)) if price_m else None,
            "output": float(price_m.group(2)) if price_m else None,
            "context": _parse_context_str(ctx_m.group(1) if ctx_m else None),
        }

    if not scores:
        logger.warning("arena.ai %s: no scores parsed — check scraper", category)
    else:
        logger.info("arena.ai %s: fetched %d scores", category, len(scores))
    return scores, prices


def fetch_arena_scores(
    timeout: int = 30,
) -> tuple[dict[str, dict[str, int]], dict[str, dict]]:
    """Fetch ELO scores and prices for all Arena leaderboard categories.

    Args:
        timeout: HTTP request timeout in seconds.

    Returns:
        Tuple of (scores_by_category, arena_prices) where arena_prices maps
        Arena model name → {input, output, context} merged across all categories.
        Categories with no scores are omitted from scores_by_category.
    """
    scores_by_cat: dict[str, dict[str, int]] = {}
    arena_prices: dict[str, dict] = {}

    for category, url in _ARENA_URLS.items():
        scores, prices = _fetch_category(category, url, timeout)
        if scores:
            scores_by_cat[category] = scores
            # Merge prices: fill None fields from later categories, skip all-None entries
            for name, price in prices.items():
                if price["input"] is None and price["output"] is None and price["context"] is None:
                    continue
                if name not in arena_prices:
                    arena_prices[name] = price
                else:
                    existing = arena_prices[name]
                    arena_prices[name] = {
                        "input": existing["input"] if existing["input"] is not None else price["input"],
                        "output": existing["output"] if existing["output"] is not None else price["output"],
                        "context": existing["context"] if existing["context"] is not None else price["context"],
                    }

    return scores_by_cat, arena_prices


def save_arena_scores(
    scores_by_category: dict[str, dict[str, int]],
    arena_prices: dict[str, dict] | None = None,
) -> None:
    """Merge new scores (and optionally prices) into arena_scores.json incrementally.

    Per category: new names are inserted, existing names are updated, names
    absent from the fresh fetch are left unchanged. Refuses to write if scores
    is empty.

    Args:
        scores_by_category: category → (Arena model name → ELO score).
        arena_prices: Arena model name → {input, output, context} to persist.
    """
    if not scores_by_category:
        logger.warning("arena: refusing to save empty scores — keeping existing cache")
        return
    existing = load_arena_scores()
    added = updated = 0
    for category, new_scores in scores_by_category.items():
        old_cat = existing.get(category, {})
        added += sum(1 for k in new_scores if k not in old_cat)
        updated += sum(1 for k, v in new_scores.items() if k in old_cat and old_cat[k] != v)
        existing[category] = {**old_cat, **new_scores}

    # Persist arena prices: merge field-level (fill None from old, skip all-None entries)
    if arena_prices:
        old_prices = _load_raw().get("arena_prices", {})
        merged_prices: dict[str, dict] = dict(old_prices)
        for name, price in arena_prices.items():
            if price["input"] is None and price["output"] is None and price["context"] is None:
                continue
            old = merged_prices.get(name, {})
            merged_prices[name] = {
                "input": price["input"] if price["input"] is not None else old.get("input"),
                "output": price["output"] if price["output"] is not None else old.get("output"),
                "context": price["context"] if price["context"] is not None else old.get("context"),
            }
        existing["arena_prices"] = merged_prices

    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(json.dumps(existing, indent=2, sort_keys=True))
    total = sum(len(v) for k, v in existing.items() if k not in _NON_SCORE_KEYS)
    logger.info("arena: saved %d total (%d new, %d updated)", total, added, updated)


def _load_raw() -> dict:
    """Return the raw JSON dict from the cache file, or empty dict."""
    if not _CACHE_PATH.exists():
        return {}
    try:
        return json.loads(_CACHE_PATH.read_text())
    except Exception:
        return {}


def load_arena_scores() -> dict[str, dict[str, int]]:
    """Load cached arena scores from arena_scores.json.

    Transparently migrates the legacy flat format {name: score} to the
    current nested format {category: {name: score}}.

    Returns:
        category → (Arena model name → ELO score), or empty dict if missing.
    """
    data = _load_raw()
    if not data:
        return {}
    score_data = {k: v for k, v in data.items() if k not in _NON_SCORE_KEYS}
    if not score_data:
        return {}
    # Legacy flat format: values are ints, not dicts
    first_val = next(iter(score_data.values()))
    if isinstance(first_val, int):
        logger.info("arena: migrating legacy flat cache to nested format")
        return {"text": score_data}
    return score_data


def _match_arena_name(model_id: str, arena_dict: dict) -> str | None:
    """Find the Arena leaderboard name that best matches an OpenRouter model ID.

    Uses the same strategy as _match_score but returns the name key instead
    of the value, so callers can look up any field (score, price, context).
    """
    slug = model_id.split("/")[-1] if "/" in model_id else model_id
    slug_normalized = slug.replace(".", "-")

    if slug in _MANUAL_MAPPINGS:
        mapped = _MANUAL_MAPPINGS[slug]
        if mapped in arena_dict:
            return mapped

    if slug in arena_dict:
        return slug
    if slug_normalized != slug and slug_normalized in arena_dict:
        return slug_normalized

    candidates = [
        name for name in arena_dict
        if (slug in name or name in slug
            or slug_normalized in name or name in slug_normalized)
    ]
    # Prefer shortest (most specific) match
    return min(candidates, key=len) if candidates else None


def detect_price_drift(
    records: list[ModelRecord],
    threshold: float,
) -> list[dict]:
    """Compare current OpenRouter prices against Arena's listed prices.

    Args:
        records: Current OpenRouter model records.
        threshold: Fractional price change that triggers a flag (e.g. 0.01 = 1%).
                   Context length is always compared with strict equality.

    Returns:
        List of dicts with keys 'id', 'name', 'provider', 'issues'.
    """
    arena_prices: dict[str, dict] = _load_raw().get("arena_prices", {})
    if not arena_prices:
        return []

    drifted = []
    for r in records:
        arena_name = _match_arena_name(r.id, arena_prices)
        if not arena_name:
            continue
        saved = arena_prices[arena_name]
        overridden = set(r.overridden_fields or [])
        issues: list[str] = []

        for label, field, arena_val, or_val in [
            ("Input", "input_per_mtok", saved.get("input"), r.input_per_mtok),
            ("Output", "output_per_mtok", saved.get("output"), r.output_per_mtok),
        ]:
            if field in overridden:
                continue
            if arena_val is not None and or_val is not None and arena_val > 0:
                diff = abs(or_val - arena_val) / arena_val
                if diff > threshold:
                    arrow = "↑" if or_val > arena_val else "↓"
                    issues.append(
                        f"{label}: arena ${arena_val:.3f} vs OR ${or_val:.3f} {arrow}{diff:.0%}"
                    )

        if "context_length" not in overridden:
            arena_ctx = saved.get("context")
            if arena_ctx is not None and r.context_length is not None:
                arena_tokens = _ctx_str_to_tokens(arena_ctx)
                if arena_tokens and arena_tokens > 0:
                    diff = abs(r.context_length - arena_tokens) / arena_tokens
                    if diff > CONTEXT_DRIFT_BUFFER:
                        or_ctx = _fmt_ctx(r.context_length)
                        issues.append(f"Context: arena {arena_ctx} vs OR {or_ctx} {diff:.0%}")

        if issues:
            drifted.append({
                "id": r.id,
                "name": r.direct_id or r.id.split("/")[-1],
                "provider": r.provider_name,
                "arena_name": arena_name,
                "issues": issues,
            })

    return drifted


def _match_score(model_id: str, scores: dict[str, int]) -> int | None:
    """Match an OpenRouter model ID to an Arena ELO score."""
    name = _match_arena_name(model_id, scores)
    return scores[name] if name else None


def _weighted_avg(
    model_id: str,
    scores_by_category: dict[str, dict[str, int]],
    weights: dict[str, float],
) -> tuple[int | None, dict[str, int]]:
    """Compute a weighted average Arena score across categories.

    Missing categories are dropped and remaining weights renormalized.

    Returns:
        (rounded weighted-average score or None, detail dict of category → score).
    """
    weighted_sum = 0.0
    total_weight = 0.0
    detail: dict[str, int] = {}
    for category, weight in weights.items():
        score = _match_score(model_id, scores_by_category.get(category, {}))
        if score is not None:
            weighted_sum += score * weight
            total_weight += weight
            detail[category] = score
    avg = round(weighted_sum / total_weight) if total_weight else None
    return avg, detail


def apply_arena_scores(
    records: list[ModelRecord],
    scores_by_category: dict[str, dict[str, int]],
    text_weights: dict[str, float] | None = None,
    vision_weights: dict[str, float] | None = None,
    image_weights: dict[str, float] | None = None,
) -> list[ModelRecord]:
    """Populate arena_score on each record using per-category Arena scores.

    - Image-output models (category == "image"): text_to_image + image_edit.
    - Vision-in text models ("image" in input_modalities): text + coding + vision.
    - Other text models: text + coding.

    Missing categories are ignored and weights renormalized.
    """
    if not scores_by_category:
        return records

    tw = text_weights if text_weights is not None else ARENA_TEXT_WEIGHTS
    vw = vision_weights if vision_weights is not None else ARENA_VISION_WEIGHTS
    iw = image_weights if image_weights is not None else ARENA_IMAGE_WEIGHTS

    for r in records:
        if r.category == "image":
            matched, detail = _weighted_avg(r.id, scores_by_category, iw)
        elif "image" in r.input_modalities:
            matched, detail = _weighted_avg(r.id, scores_by_category, vw)
        else:
            matched, detail = _weighted_avg(r.id, scores_by_category, tw)
        if matched is not None:
            r.arena_score = matched
            r.arena_scores_detail = detail

    return records
