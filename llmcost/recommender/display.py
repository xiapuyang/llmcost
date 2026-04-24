"""Render model recommendations to the terminal using Rich."""

from __future__ import annotations

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from rich import box

from llmcost.pricing.filters.pipeline import CN_PROVIDERS
from llmcost.recommender.engine import Recommendation, ScoredCandidate
from llmcost.recommender.wizard import IMAGE_USE_CASES, UserPreferences


def _format_price_command(prefs: UserPreferences) -> str:
    """Build the equivalent `llmcost price` command for these preferences."""
    category = "image" if prefs.use_case in IMAGE_USE_CASES else "text"
    parts = ["llmcost price"]

    parts.append(f"--input-ratio {prefs.input_ratio}")
    parts.append(f"--cache-hit-ratio {prefs.cache_hit_ratio}")

    if prefs.min_arena_score:
        parts.append(f"--min-arena-score {prefs.min_arena_score}")

    if prefs.max_price:
        parts.append(f"--max-price {prefs.max_price:.0f}")

    parts.append(f"--output {category}")

    if prefs.vision_input:
        parts.append("--vision-in")

    # Resolve model_source to --provider list when possible
    if prefs.model_source == "cn":
        parts.append(f"--provider {','.join(sorted(CN_PROVIDERS))}")
    elif prefs.providers is not None:
        parts.append(f"--provider {','.join(prefs.providers)}")

    # Flags with no pricing CLI equivalent — append as comments
    notes = []
    if prefs.min_context_length:
        notes.append(f"# min_context_length={prefs.min_context_length:,} (no --flag)")
    if prefs.model_source == "us":
        notes.append("# model_source=us (no --flag)")

    cmd = "  ".join(parts)
    if notes:
        cmd += "  " + "  ".join(notes)
    return cmd


_TIER_SUBTITLES: dict[str, str] = {
    "Best Value": "Lowest Cost",
    "Best Quality": "Highest Arena Score",
    "Balanced": "Best Cost/Quality Ratio",
}

_TIER_COLORS: dict[str, str] = {
    "Best Value": "green",
    "Best Quality": "blue",
    "Balanced": "yellow",
}


def _build_panel(rec: Recommendation) -> Panel:
    subtitle = _TIER_SUBTITLES.get(rec.tier, "")
    color = _TIER_COLORS.get(rec.tier, "white")

    lines = Text()
    lines.append(f"{rec.record.id}\n", style="bold")
    lines.append(f"Provider:  {rec.record.provider}\n")
    lines.append(f"Price:     ${rec.weighted_price:.2f}/M tokens\n")
    if rec.value_ratio is not None:
        lines.append(f"$/kArena:  {rec.value_ratio:.3f}\n")
    if rec.record.arena_score is not None:
        lines.append(f"Arena:     {rec.record.arena_score}\n")
    lines.append(f"\n{rec.rationale}", style="dim")

    return Panel(
        lines,
        title=f"[bold {color}]{rec.tier}[/bold {color}]",
        subtitle=f"[dim]{subtitle}[/dim]",
        border_style=color,
        expand=True,
    )


def render_debug_candidates(
    candidates: list[ScoredCandidate],
    prefs: UserPreferences,
    console: Console,
) -> None:
    """Print all filtered candidates ranked by Balanced combined score.

    Args:
        candidates: Output of ModelRecommender.debug_candidates(), sorted by combined_score.
        prefs: User preferences (used to show preferred param counts).
        console: Rich Console instance for output.
    """
    n_preferred = len(prefs.preferred_parameters)
    t = Table(
        title="[dim]Debug: all candidates sorted by $/kArena (lower = better)[/dim]",
        box=box.SIMPLE_HEAD,
        show_lines=False,
    )
    t.add_column("#", justify="right", style="dim")
    t.add_column("Model", style="cyan", min_width=22)
    t.add_column("Provider", style="white")
    t.add_column("Arena", justify="right", style="magenta")
    t.add_column("$/kArena↑", justify="right", style="blue")
    t.add_column("Weighted$/M", justify="right", style="yellow")
    if n_preferred:
        t.add_column(f"Preferred ({n_preferred})", justify="center", style="green")
    t.add_column("Combined", justify="right", style="dim")

    candidates = sorted(candidates, key=lambda c: c.value_ratio if c.value_ratio is not None else float("inf"))

    unknown_ctx = prefs.min_context_length is not None
    has_unknown_ctx_model = False

    for rank, c in enumerate(candidates, 1):
        r = c.record
        display_id = r.direct_id if r.direct_id else r.id.split("/")[-1]
        if unknown_ctx and r.context_length is None:
            display_id += " [dim]?ctx[/dim]"
            has_unknown_ctx_model = True
        arena = str(r.arena_score) if r.arena_score is not None else "—"
        vr = f"{c.value_ratio:.3f}" if c.value_ratio is not None else "—"
        wp = f"${c.weighted_price:.3f}"
        combined = f"{c.combined_score:.3f}"
        row = [str(rank), display_id, r.provider_name, arena, vr, wp]
        if n_preferred:
            n_hit = round(c.preferred_score * n_preferred)
            row.append(f"{n_hit}/{n_preferred}")
        row.append(combined)
        t.add_row(*row)

    console.print(t)
    if has_unknown_ctx_model:
        console.print(
            "[dim]?ctx = context length unreported; passed min_context_length filter "
            "because actual capacity is unknown[/dim]"
        )


def display_filter_summary(
    prefs: UserPreferences,
    total_count: int,
    surviving_count: int,
    console: Console,
) -> None:
    """Print a compact table of the active filter conditions.

    Args:
        prefs: The user preferences collected by the wizard.
        total_count: Total records before filtering.
        surviving_count: Records that passed all filters.
        console: Rich Console instance for output.
    """
    t = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    t.add_column(style="dim")
    t.add_column()

    # Use case / category
    category = "image" if prefs.use_case in IMAGE_USE_CASES else "text"
    t.add_row("use_case", f"{prefs.use_case}  →  category={category}")

    # Token ratio — use :g to strip trailing zeros (e.g. 7.5:2.5, not 7:2)
    in_n = round(prefs.input_ratio * 10, 1)
    out_n = round((1 - prefs.input_ratio) * 10, 1)
    ratio_label = f"{in_n:g}:{out_n:g}"
    src = f"  [{prefs.input_ratio_source}]"
    t.add_row("input_ratio", f"{ratio_label}{src}  (cache_hit_ratio={prefs.cache_hit_ratio:.0%})")

    # Vision
    t.add_row("vision_input", "Yes" if prefs.vision_input else "No")

    # Context
    ctx = f"≥ {prefs.min_context_length:,}" if prefs.min_context_length else "no requirement"
    t.add_row("min_context_length", ctx)

    # Source
    t.add_row("model_source", prefs.model_source)

    # Arena
    t.add_row("min_arena_score", str(prefs.min_arena_score) if prefs.min_arena_score else "0 (no limit)")

    # Max price
    if prefs.max_price_model:
        price = f"≤ {prefs.max_price_model} price"
    elif prefs.max_price:
        price = f"${prefs.max_price:.0f}/M"
    else:
        price = "no limit"
    t.add_row("max_price", price)

    # Providers
    if prefs.providers is None:
        prov = "all providers"
    else:
        prov = f"{len(prefs.providers)} selected: {', '.join(prefs.providers)}"
    t.add_row("providers", prov)

    # Survivor count
    t.add_row("", "")
    t.add_row(
        "→ surviving",
        f"[bold]{surviving_count}[/bold] / {total_count} models",
    )

    # Equivalent CLI command
    t.add_row("", "")
    t.add_row("equivalent cmd", f"[bold cyan]{_format_price_command(prefs)}[/bold cyan]")

    console.print(Panel(t, title="[dim]Filter conditions[/dim]", border_style="dim"))


def display_recommendations(
    recommendations: list[Recommendation],
    surviving_count: int,
    console: Console,
) -> None:
    """Render recommendation panels to the console.

    Args:
        recommendations: List of Recommendation objects to display.
        surviving_count: Number of models that passed all filters.
        console: Rich Console instance for output.
    """
    if surviving_count == 0:
        console.print(
            "[bold red]No models matched your criteria.[/bold red] "
            "Try relaxing the Arena score threshold or context length requirement."
        )
        return

    if not recommendations:
        console.print("[yellow]No recommendations could be generated.[/yellow]")
        return

    panels = [_build_panel(r) for r in recommendations]

    if len(panels) == 1:
        console.print(panels[0])
        console.print(
            f"[yellow]Only {surviving_count} model(s) matched your criteria — "
            "showing best value only. Try relaxing filters for more options.[/yellow]"
        )
    else:
        console.print(Columns(panels, equal=True, expand=True))

    console.print(
        f"\n[dim]Filtered from {surviving_count} model(s) matching your criteria.[/dim]"
    )
