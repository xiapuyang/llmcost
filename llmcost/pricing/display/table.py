"""Rich table display with on-the-fly weighted price calculation."""

from __future__ import annotations

from itertools import groupby

from rich.console import Console
from rich.table import Table
from rich import box

from llmcost.pricing.models import ModelRecord

console = Console()


def compute_weighted(
    record: ModelRecord,
    *,
    input_ratio: float,
    cache_hit_ratio: float = 0.0,
) -> float | None:
    """Compute weighted price at display time. Never stored in ModelRecord.

    Token pricing takes priority over flat image_per_unit when both are present.
    image_per_unit is used only when no token prices exist (e.g. legacy flat-rate models).

    When cache_hit_ratio > 0 and the record has cache_read_per_mtok, the effective
    input cost is blended: cache_hit_ratio * cache_read + (1 - cache_hit_ratio) * input.
    Models without cache pricing are unaffected (full input price used).

    Args:
        record: The model record to compute for.
        input_ratio: Fraction of cost attributed to input tokens (0.0–1.0).
        cache_hit_ratio: Fraction of input tokens assumed to be cache hits (0.0–1.0).

    Returns:
        Weighted price per MTok, or the flat image_per_unit if applicable, or None.
    """
    if record.input_per_mtok is not None and record.output_per_mtok is not None:
        if cache_hit_ratio > 0 and record.cache_read_per_mtok is not None:
            effective_input = (
                cache_hit_ratio * record.cache_read_per_mtok
                + (1 - cache_hit_ratio) * record.input_per_mtok
            )
        else:
            effective_input = record.input_per_mtok
        return input_ratio * effective_input + (1 - input_ratio) * record.output_per_mtok
    if record.image_per_unit is not None:
        return record.image_per_unit
    return None


def compute_value_ratio(
    record: ModelRecord,
    *,
    input_ratio: float,
    cache_hit_ratio: float = 0.0,
) -> float | None:
    """Compute cost-effectiveness: Weighted$/M divided by (Arena/1000).

    Lower is better — fewer dollars per unit of Arena quality.

    Args:
        record: The model record to compute for.
        input_ratio: Fraction of cost attributed to input tokens (0.0–1.0).
        cache_hit_ratio: Fraction of input tokens assumed to be cache hits (0.0–1.0).

    Returns:
        Value ratio, or None if Arena score or weighted price is unavailable.
    """
    if record.arena_score is None:
        return None
    weighted = compute_weighted(record, input_ratio=input_ratio, cache_hit_ratio=cache_hit_ratio)
    if weighted is None:
        return None
    return weighted / (record.arena_score / 1000)


def format_context(ctx: int | None) -> str:
    """Format a context length integer as a human-readable string.

    Args:
        ctx: Context length in tokens, or None.

    Returns:
        Formatted string like '64K', '1M', or '—'.
    """
    if ctx is None:
        return "—"
    if ctx >= 1_000_000:
        return f"{ctx // 1_000_000}M"
    return f"{ctx // 1_024}K"


def fmt_price(value: float | None) -> str:
    """Format a price value as a dollar string or an em-dash placeholder.

    Args:
        value: Price in dollars per million tokens, or None if unavailable.

    Returns:
        Formatted string like '$0.003' or '—'.
    """
    if value is None:
        return "—"
    return f"${value:.3f}"


_ARENA_CATEGORY_ABBREV: dict[str, str] = {
    "text": "T",
    "coding": "C",
    "vision": "V",
    "text_to_image": "TI",
    "image_edit": "IE",
}


def _fmt_arena_breakdown(detail: dict[str, int] | None) -> str:
    """Format per-category Arena scores as a compact string, e.g. 'T:1400 C:1380'.

    Args:
        detail: category → score mapping from arena_scores_detail, or None.

    Returns:
        Formatted string or '—' if no detail available.
    """
    if not detail:
        return "—"
    parts = [f"{_ARENA_CATEGORY_ABBREV.get(cat, cat)}:{score}" for cat, score in detail.items()]
    return " ".join(parts)


def _weighted_sort_key(record: ModelRecord, input_ratio: float, cache_hit_ratio: float = 0.0) -> float:
    """Return the weighted price for sorting, mapping None to positive infinity.

    Args:
        record: The model record to evaluate.
        input_ratio: Fraction of cost attributed to input tokens (0.0–1.0).
        cache_hit_ratio: Fraction of input tokens assumed to be cache hits (0.0–1.0).

    Returns:
        Weighted price, or float('inf') when price data is unavailable.
    """
    w = compute_weighted(record, input_ratio=input_ratio, cache_hit_ratio=cache_hit_ratio)
    return w if w is not None else float("inf")


def _group_key(record: ModelRecord) -> tuple:
    """Return the group key for a record: (category, has_vision_input).

    Records without an Arena score always sort into a separate group (key None).
    """
    if record.arena_score is None:
        return (None, None)
    return (record.category or "text", "image" in record.input_modalities)


_GROUP_ORDER = {
    ("text", False): 0,
    ("text", True): 1,
    ("image", False): 2,
    ("image", True): 3,
    (None, None): 99,
}


def _collect_groups(
    records: list[ModelRecord],
    input_ratio: float,
    cache_hit_ratio: float,
    *,
    group_by_vision: bool = True,
) -> list[tuple[tuple, list[ModelRecord]]]:
    """Split records into display groups, each sorted appropriately.

    Groups with Arena scores are sorted by value ratio (lower = better).
    The no-Arena group is sorted by weighted price.
    """
    buckets: dict[tuple, list[ModelRecord]] = {}
    for r in records:
        key = _group_key(r)
        if not group_by_vision and key != (None, None):
            key = (key[0], None)
        buckets.setdefault(key, []).append(r)

    result = []
    for key in sorted(buckets, key=lambda k: _GROUP_ORDER.get(k, 50)):
        group = buckets[key]
        if key == (None, None):
            group.sort(key=lambda r: _weighted_sort_key(r, input_ratio, cache_hit_ratio))
        else:
            group.sort(
                key=lambda r: (
                    compute_value_ratio(r, input_ratio=input_ratio, cache_hit_ratio=cache_hit_ratio)
                    or float("inf")
                )
            )
        result.append((key, group))
    return result


def render_table(
    records: list[ModelRecord],
    *,
    input_ratio: float,
    cache_hit_ratio: float = 0.0,
    category: str | None = None,
    group_by_vision: bool = True,
) -> None:
    """Render a rich table of model records grouped by (output, vision) and sorted by value ratio.

    Args:
        records: Model records to display.
        input_ratio: Input token cost weight for weighted price calculation.
        cache_hit_ratio: Fraction of input tokens assumed to be cache hits (0.0–1.0).
        category: Category label for the table title (e.g. 'text', 'image').
        group_by_vision: Whether to split groups by vision input capability.
    """
    output_ratio = 1 - input_ratio
    title = f"[bold]{category.title() if category else 'All'} Models[/bold] — "
    title += f"weighted price (input {int(input_ratio*100)}% / output {int(output_ratio*100)}%"
    if cache_hit_ratio > 0:
        title += f", cache {int(cache_hit_ratio*100)}%"
    title += ")"

    table = Table(title=title, box=box.ROUNDED, show_lines=False)
    table.add_column("Model", style="cyan", min_width=20)
    table.add_column("Provider", style="white")
    table.add_column("Input/M", justify="right", style="green")
    table.add_column("Output/M", justify="right", style="red")
    table.add_column("Context", justify="right")
    table.add_column("Max Out", justify="right")
    table.add_column("Weighted$/M", justify="right", style="bold yellow")
    table.add_column("Vision In", justify="center")
    table.add_column("Arena Breakdown", justify="right", style="dim magenta")
    table.add_column("Arena", justify="right", style="magenta")
    table.add_column("$/kArena", justify="right", style="blue")

    groups = _collect_groups(records, input_ratio, cache_hit_ratio, group_by_vision=group_by_vision)
    n_groups = len(groups)

    for g_idx, (key, group) in enumerate(groups):
        cat, has_vision = key
        if cat is None:
            header = "── no Arena score ──"
        elif has_vision is None:
            header = f"── {cat} ──"
        else:
            vision_label = "vision ✓" if has_vision else "vision ✗"
            header = f"── {cat} / {vision_label} ──"

        table.add_row(
            f"[dim]{header}[/dim]", "", "", "", "", "", "", "", "", "", "",
        )

        for r_idx, r in enumerate(group):
            weighted = compute_weighted(r, input_ratio=input_ratio, cache_hit_ratio=cache_hit_ratio)
            vr = compute_value_ratio(r, input_ratio=input_ratio, cache_hit_ratio=cache_hit_ratio)
            vision = "✓" if "image" in r.input_modalities else "✗"
            flag = "[dim]⚠ [/dim]" if r.blacklisted else ""
            display_id = r.direct_id if r.direct_id else r.id.split("/")[-1]
            arena = str(r.arena_score) if r.arena_score is not None else "—"
            breakdown = _fmt_arena_breakdown(r.arena_scores_detail)
            vr_str = f"{vr:.3f}" if vr is not None else "—"
            is_last_in_group = r_idx == len(group) - 1
            table.add_row(
                f"{flag}{display_id}",
                r.provider_name,
                fmt_price(r.input_per_mtok),
                fmt_price(r.output_per_mtok),
                format_context(r.context_length),
                format_context(r.max_output_tokens),
                fmt_price(weighted),
                vision,
                breakdown,
                arena,
                vr_str,
                end_section=is_last_in_group and g_idx < n_groups - 1,
            )

    console.print(table)
    console.print(
        "[dim]Source: openrouter.ai/api/v1/models | "
        "Tip: use --input-ratio to adjust weighting | "
        "OR prices may differ from direct API rates[/dim]"
    )
    console.print(
        "[dim]Arena Breakdown: T=text  C=coding  V=vision  TI=text-to-image  IE=image-edit[/dim]"
    )


def render_drift_warnings(drifted: list[dict]) -> None:
    """Print a compact warning table for models whose price or context drifted.

    Args:
        drifted: Output of detect_price_drift — list of dicts with
                 'name', 'provider', 'issues' keys.
    """
    if not drifted:
        return
    table = Table(
        title="[bold yellow]⚠  Price / Context drift detected — please check[/bold yellow]",
        box=box.SIMPLE_HEAD,
        show_lines=False,
    )
    table.add_column("Model", style="cyan")
    table.add_column("Provider", style="white")
    table.add_column("Changes", style="yellow")
    for item in drifted:
        table.add_row(item["name"], item["provider"], "  |  ".join(item["issues"]))
    console.print()
    console.print(table)
