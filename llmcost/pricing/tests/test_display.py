# tests/test_display.py
import pytest
from llmcost.pricing.models import ModelRecord
from llmcost.pricing.display.table import compute_weighted, fmt_price, format_context, render_table
from llmcost.pricing.display.markdown import render_markdown


def _record(**kwargs) -> ModelRecord:
    defaults = dict(
        id="anthropic/claude-sonnet-4.6",
        name="Claude Sonnet 4.6",
        provider="anthropic",
        provider_name="Anthropic",
        pricing_url="https://www.anthropic.com/pricing#api",
        modality_raw="text+image->text",
        input_modalities=["text", "image"],
        output_modalities=["text"],
        category="text",
        context_length=1_000_000,
        input_per_mtok=3.0,
        output_per_mtok=15.0,
        source="openrouter",
        fetched_at="2026-04-23T10:00:00Z",
    )
    defaults.update(kwargs)
    return ModelRecord(**defaults)


def test_compute_weighted_default_ratio():
    r = _record()
    assert compute_weighted(r, input_ratio=0.2) == pytest.approx(0.2 * 3.0 + 0.8 * 15.0)


def test_compute_weighted_image_unit():
    r = _record(image_per_unit=0.04, image_unit="image", input_per_mtok=None, output_per_mtok=None)
    assert compute_weighted(r, input_ratio=0.2) == 0.04


def test_compute_weighted_token_price_beats_image_unit():
    """Token pricing takes priority over image_per_unit when both are present."""
    r = _record(image_per_unit=0.0003, image_unit="image", input_per_mtok=0.3, output_per_mtok=2.5)
    assert compute_weighted(r, input_ratio=0.2) == pytest.approx(0.2 * 0.3 + 0.8 * 2.5)


def test_compute_weighted_none_when_no_price():
    r = _record(input_per_mtok=None, output_per_mtok=None)
    assert compute_weighted(r, input_ratio=0.2) is None


def test_format_context_k():
    assert format_context(65_536) == "64K"


def test_format_context_m():
    assert format_context(1_048_576) == "1M"


def test_format_context_none():
    assert format_context(None) == "—"


def test_render_table_runs_without_error():
    records = [_record()]
    render_table(records, input_ratio=0.2, category="text")


def test_render_markdown_contains_model_name():
    records = [_record()]
    md = render_markdown(records, input_ratio=0.2)
    assert "Claude Sonnet 4.6" in md
    assert "Anthropic" in md


def test_compute_weighted_zero_price():
    """A price of 0.0 should return 0.0, not None."""
    r = _record(input_per_mtok=0.0, output_per_mtok=0.0)
    assert compute_weighted(r, input_ratio=0.2) == pytest.approx(0.0)


def test_render_markdown_sorted_by_price():
    """Cheapest model should appear first in markdown output."""
    cheap = _record(id="a/cheap", name="Cheap", input_per_mtok=0.1, output_per_mtok=0.1)
    expensive = _record(id="b/expensive", name="Expensive", input_per_mtok=10.0, output_per_mtok=30.0)
    md = render_markdown([expensive, cheap], input_ratio=0.2)
    assert md.index("Cheap") < md.index("Expensive")
