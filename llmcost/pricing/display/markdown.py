"""Markdown export of price table."""

from __future__ import annotations

from datetime import datetime, timezone

from llmcost.pricing.display.table import (
    compute_weighted,
    fmt_price,
    format_context,
    _weighted_sort_key,
)
from llmcost.pricing.models import ModelRecord


def render_markdown(
    records: list[ModelRecord],
    *,
    input_ratio: float,
    cache_hit_ratio: float = 0.0,
) -> str:
    """Render model records as a Markdown table sorted by weighted price.

    Args:
        records: Model records to render.
        input_ratio: Input token cost weight for weighted price calculation.
        cache_hit_ratio: Fraction of input tokens assumed to be cache hits (0.0–1.0).

    Returns:
        Markdown string with a header and table of model prices.
    """
    output_ratio = 1 - input_ratio
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    weight_desc = f"Weighted: {int(input_ratio*100)}% input / {int(output_ratio*100)}% output"
    if cache_hit_ratio > 0:
        weight_desc += f" / {int(cache_hit_ratio*100)}% cache hit"
    lines = [
        "# LLM Price Comparison",
        "",
        f"Generated: {now} | {weight_desc}",
        "",
        "| Model | Provider | Input $/M | Output $/M | Context | Weighted $/M | Vision In | Arena | Source |",
        "|-------|----------|-----------|------------|---------|--------------|-----------|-------|--------|",
    ]

    sorted_records = sorted(
        records,
        key=lambda r: _weighted_sort_key(r, input_ratio, cache_hit_ratio),
    )

    for r in sorted_records:
        weighted = compute_weighted(r, input_ratio=input_ratio, cache_hit_ratio=cache_hit_ratio)
        vision = "✓" if "image" in r.input_modalities else "✗"
        flag = "⚠ " if r.blacklisted else ""
        inp = fmt_price(r.input_per_mtok)
        out = fmt_price(r.output_per_mtok)
        w = fmt_price(weighted)
        arena = str(r.arena_score) if r.arena_score is not None else "—"
        lines.append(
            f"| {flag}{r.name} | [{r.provider_name}]({r.pricing_url}) "
            f"| {inp} | {out} | {format_context(r.context_length)} | {w} | {vision} | {arena} | {r.source} |"
        )

    return "\n".join(lines) + "\n"
