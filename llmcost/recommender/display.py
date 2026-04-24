"""Render model recommendations to the terminal using Rich."""

from __future__ import annotations

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from llmcost.recommender.engine import Recommendation

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
