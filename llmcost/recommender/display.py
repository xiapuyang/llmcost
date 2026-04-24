"""Render model recommendations to the terminal using Rich."""

from __future__ import annotations

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from llmcost.recommender.engine import Recommendation
from llmcost.recommender.wizard import UserPreferences

_CN_PROVIDERS = frozenset({"zhipu", "minimax", "moonshotai", "dashscope", "bytedance-seed"})


def _format_price_command(prefs: UserPreferences) -> str:
    """Build the equivalent `llmcost price` command for these preferences."""
    category = "image" if prefs.use_case in {"text-to-image", "image editing"} else "text"
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
        parts.append(f"--provider {','.join(sorted(_CN_PROVIDERS))}")
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
    category = "image" if prefs.use_case in {"text-to-image", "image editing"} else "text"
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
    price = f"${prefs.max_price:.0f}/M" if prefs.max_price else "no limit"
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
