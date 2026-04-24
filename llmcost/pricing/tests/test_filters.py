# tests/test_filters.py
from pathlib import Path
from llmcost.pricing.models import ModelRecord
from llmcost.pricing.filters.blacklist import BlacklistFilter
from llmcost.pricing.filters.arena import ArenaFilter
from llmcost.pricing.filters.pipeline import RecordFilter, _is_redundant_pinned


def _record(model_id: str, arena_score: int | None = None, **kwargs) -> ModelRecord:
    defaults = dict(
        id=model_id,
        name=model_id,
        provider=model_id.split("/")[0],
        provider_name=model_id.split("/")[0],
        pricing_url="",
        modality_raw="text->text",
        input_modalities=["text"],
        output_modalities=["text"],
        category="text",
        input_per_mtok=1.0,
        output_per_mtok=4.0,
        arena_score=arena_score,
        source="openrouter",
        fetched_at="2026-04-23T10:00:00Z",
    )
    defaults.update(kwargs)
    return ModelRecord(**defaults)


def test_blacklist_marks_model(tmp_path):
    bl_file = tmp_path / "blacklist.yaml"
    bl_file.write_text("- id: openai/old-model\n  reason: deprecated\n")
    f = BlacklistFilter(bl_file)
    records = [_record("openai/old-model"), _record("openai/gpt-4o")]
    result = f.apply(records, show_all=False)
    assert len(result) == 1
    assert result[0].id == "openai/gpt-4o"


def test_blacklist_show_all_keeps_marked(tmp_path):
    bl_file = tmp_path / "blacklist.yaml"
    bl_file.write_text("- id: openai/old-model\n  reason: deprecated\n")
    f = BlacklistFilter(bl_file)
    records = [_record("openai/old-model"), _record("openai/gpt-4o")]
    result = f.apply(records, show_all=True)
    assert len(result) == 2
    blacklisted = next(r for r in result if r.id == "openai/old-model")
    assert blacklisted.blacklisted is True
    clean = next(r for r in result if r.id == "openai/gpt-4o")
    assert clean.blacklisted is False


def test_arena_filter_excludes_below_threshold():
    f = ArenaFilter(threshold=1100)
    records = [
        _record("a/model", arena_score=1200),
        _record("b/model", arena_score=900),
        _record("c/model", arena_score=None),  # no score → keep
    ]
    result = f.apply(records)
    ids = [r.id for r in result]
    assert "a/model" in ids
    assert "c/model" in ids
    assert "b/model" not in ids


# ── has_required_parameters (T2) ──────────────────────────────────────────

def test_has_required_parameters_or_logic_any_match_passes():
    """A model supporting any one required param passes (OR logic)."""
    r = _record("p/partial", arena_score=1300, supported_parameters=["tools"])
    result = RecordFilter([r]).has_required_parameters(("tools", "tool_choice")).build()
    assert len(result) == 1


def test_has_required_parameters_excluded_when_none_match():
    """A model supporting none of the required params is excluded."""
    r = _record("p/missing", arena_score=1300, supported_parameters=["temperature", "top_p"])
    result = RecordFilter([r]).has_required_parameters(("tools", "tool_choice")).build()
    assert len(result) == 0


def test_has_required_parameters_none_supported_parameters_kept():
    """supported_parameters=None (unreported) passes the filter — capability unknown."""
    r = _record("p/unknown", arena_score=1300, supported_parameters=None)
    result = RecordFilter([r]).has_required_parameters(("tools", "tool_choice")).build()
    assert len(result) == 1


def test_has_required_parameters_empty_params_noop():
    """Empty params tuple is a no-op — all records pass."""
    records = [_record("p/a", arena_score=1300), _record("p/b", arena_score=1350)]
    result = RecordFilter(records).has_required_parameters(()).build()
    assert len(result) == 2


# ── _is_redundant_pinned (T3) ─────────────────────────────────────────────

def test_redundant_pinned_date_in_slug_excluded():
    """Model with an 8-digit date in the slug is always redundant."""
    assert _is_redundant_pinned("google/gemini-2.5-flash-20250514", {"google/gemini-2.5-flash-20250514"}) is True


def test_redundant_pinned_preview_with_date_suffix_excluded():
    """Preview model with date after '-preview' is excluded (regression: gemini-2.5-flash-lite-preview-09-2025)."""
    all_ids = {"google/gemini-2.5-flash-lite-preview-09-2025"}
    assert _is_redundant_pinned("google/gemini-2.5-flash-lite-preview-09-2025", all_ids) is True


def test_redundant_pinned_plain_preview_without_date_kept():
    """A plain '-preview' slug with no date is kept when no stable base exists."""
    all_ids = {"google/gemini-2.5-flash-preview"}
    assert _is_redundant_pinned("google/gemini-2.5-flash-preview", all_ids) is False


def test_redundant_pinned_plain_preview_dropped_when_base_exists():
    """A plain '-preview' slug is redundant when the stable base ID is present."""
    all_ids = {"google/gemini-2.5-flash", "google/gemini-2.5-flash-preview"}
    assert _is_redundant_pinned("google/gemini-2.5-flash-preview", all_ids) is True
