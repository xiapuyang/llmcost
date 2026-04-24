"""Tests for RecommendWizard and UserPreferences."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llmcost.recommender.wizard import RecommendWizard, UserPreferences


# ── Helpers ────────────────────────────────────────────────────────────────

def _default_answers(overrides: dict | None = None) -> dict:
    """Return a mapping of questionary prompt substrings → answer values."""
    answers = {
        "use case":       "chat",
        "images":         "No",
        "input sample":   "",          # empty → skip
        "context length": "No requirement",
        "source region":  "Any (default)",
        "caching":        "Yes",
        "Arena score":    "1300 (default)",
        "ceiling":        "claude-opus-4.7 (default)",
    }
    if overrides:
        answers.update(overrides)
    return answers


def _run_wizard_with_mocks(
    select_map: dict,
    checkbox_return: list,
    text_sequence: list | None = None,
):
    """Run the wizard with mocked questionary calls.

    Args:
        select_map: Maps select question substrings to answer strings.
        checkbox_return: The list returned by checkbox.ask().
        text_sequence: Sequential answers for text prompts (or empty strings).

    Returns:
        UserPreferences from wizard.run().
    """
    text_iter = iter(text_sequence or [])

    def make_question_mock(answer):
        m = MagicMock()
        m.ask.return_value = answer
        return m

    def mock_select(prompt, **kwargs):
        for key, val in select_map.items():
            if key in prompt:
                return make_question_mock(val)
        return make_question_mock(kwargs.get("default", ""))

    def mock_checkbox(prompt, **kwargs):
        m = MagicMock()
        m.ask.return_value = checkbox_return
        return m

    def mock_text(prompt, **kwargs):
        try:
            answer = next(text_iter)
        except StopIteration:
            answer = ""
        return make_question_mock(answer)

    def mock_confirm(prompt, **kwargs):
        return make_question_mock(False)

    with (
        patch("llmcost.recommender.wizard.questionary.select", side_effect=mock_select),
        patch("llmcost.recommender.wizard.questionary.checkbox", side_effect=mock_checkbox),
        patch("llmcost.recommender.wizard.questionary.text", side_effect=mock_text),
        patch("llmcost.recommender.wizard.questionary.confirm", side_effect=mock_confirm),
        patch("sys.stdin.isatty", return_value=True),
    ):
        return RecommendWizard().run()


# ── Tests ──────────────────────────────────────────────────────────────────

def test_default_values_all_defaults():
    """Accepting all defaults returns expected UserPreferences (chat presets)."""
    from llmcost.pricing.config import PROVIDERS
    all_providers = list(PROVIDERS.keys())

    prefs = _run_wizard_with_mocks(
        select_map={
            "use case":       "chat",
            "images":         "No",
            "context length": "No requirement",
            "source region":  "Any (default)",
            "Arena score":    "1300 (default)",
        },
        checkbox_return=all_providers,
        text_sequence=[""],
    )

    assert prefs.use_case == "chat"
    assert prefs.vision_input is False
    assert prefs.input_ratio == 0.75       # chat registry preset
    assert prefs.input_ratio_source == "preset"
    assert prefs.min_context_length is None
    assert prefs.model_source == "any"
    assert prefs.cache_hit_ratio == 0.20   # chat registry preset
    assert prefs.min_arena_score == 1300
    assert prefs.providers == all_providers
    assert prefs.max_price_model == "claude-opus-4.7"


def test_sample_blends_with_preset_ratio():
    """When Q1.6 sample is provided, input_ratio is blended 1:1 with use-case preset."""
    from llmcost.pricing.config import PROVIDERS
    all_providers = list(PROVIDERS.keys())

    # summarization preset = 0.92; sample: input=300 chars, output=100 chars → 0.75
    # blended = (0.92 + 0.75) / 2 = 0.835
    input_sample = "a" * 300
    output_sample = "b" * 100

    prefs = _run_wizard_with_mocks(
        select_map={
            "use case": "summarization", "images": "No",
            "context length": "No requirement", "source region": "Any (default)", "Arena score": "1300 (default)",
        },
        checkbox_return=all_providers,
        text_sequence=[input_sample, output_sample],
    )

    assert prefs.input_ratio_source == "blended"
    assert abs(prefs.input_ratio - 0.835) < 0.001


def test_use_case_presets_applied_from_registry():
    """Use-case registry presets set input_ratio and cache_hit_ratio correctly."""
    from llmcost.pricing.config import PROVIDERS
    all_providers = list(PROVIDERS.keys())

    prefs = _run_wizard_with_mocks(
        select_map={
            "use case": "summarization", "images": "No",
            "source region": "Any (default)", "Arena score": "1300 (default)",
        },
        checkbox_return=all_providers,
        text_sequence=[""],
    )

    assert prefs.use_case == "summarization"
    assert prefs.input_ratio == 0.92
    assert prefs.cache_hit_ratio == 0.10
    assert prefs.input_ratio_source == "preset"


def test_image_use_case_skips_q16():
    """text-to-image use case skips Q1.6 sample step entirely."""
    from llmcost.pricing.config import PROVIDERS
    all_providers = list(PROVIDERS.keys())

    text_called = []

    def mock_select(prompt, **kwargs):
        m = MagicMock()
        if "use case" in prompt:
            m.ask.return_value = "text-to-image"
        else:
            m.ask.return_value = kwargs.get("default", "")
        return m

    def mock_text(prompt, **kwargs):
        text_called.append(True)
        m = MagicMock()
        m.ask.return_value = ""
        return m

    with (
        patch("llmcost.recommender.wizard.questionary.select", side_effect=mock_select),
        patch("llmcost.recommender.wizard.questionary.checkbox", return_value=MagicMock(ask=MagicMock(return_value=all_providers))),
        patch("llmcost.recommender.wizard.questionary.text", side_effect=mock_text),
        patch("llmcost.recommender.wizard.questionary.confirm", return_value=MagicMock(ask=MagicMock(return_value=False))),
        patch("sys.stdin.isatty", return_value=True),
    ):
        prefs = RecommendWizard().run()

    assert prefs.use_case == "text-to-image"
    assert text_called == [], "Q1.6 text prompt should not be called for image use cases"


def test_vision_input_true_skips_q16():
    """vision_input=Yes skips Q1.6 sample step."""
    from llmcost.pricing.config import PROVIDERS
    all_providers = list(PROVIDERS.keys())

    text_called = []

    def mock_select(prompt, **kwargs):
        m = MagicMock()
        if "use case" in prompt:
            m.ask.return_value = "chat"
        elif "images" in prompt:
            m.ask.return_value = "Yes"
        else:
            m.ask.return_value = kwargs.get("default", "")
        return m

    def mock_text(prompt, **kwargs):
        text_called.append(prompt)
        m = MagicMock()
        m.ask.return_value = ""
        return m

    with (
        patch("llmcost.recommender.wizard.questionary.select", side_effect=mock_select),
        patch("llmcost.recommender.wizard.questionary.checkbox", return_value=MagicMock(ask=MagicMock(return_value=all_providers))),
        patch("llmcost.recommender.wizard.questionary.text", side_effect=mock_text),
        patch("llmcost.recommender.wizard.questionary.confirm", return_value=MagicMock(ask=MagicMock(return_value=False))),
        patch("sys.stdin.isatty", return_value=True),
    ):
        prefs = RecommendWizard().run()

    assert prefs.vision_input is True
    assert text_called == [], "Q1.6 text prompt should not be called when vision_input=True"


def test_ctrl_c_raises_system_exit():
    """Ctrl+C (None from .ask()) raises SystemExit(0)."""
    def mock_select(prompt, **kwargs):
        m = MagicMock()
        m.ask.return_value = None
        return m

    with (
        patch("llmcost.recommender.wizard.questionary.select", side_effect=mock_select),
        patch("sys.stdin.isatty", return_value=True),
    ):
        with pytest.raises(SystemExit) as exc:
            RecommendWizard().run()
    assert exc.value.code == 0


def test_non_tty_raises_system_exit():
    """Non-interactive terminal raises SystemExit(1)."""
    with patch("sys.stdin.isatty", return_value=False):
        with pytest.raises(SystemExit) as exc:
            RecommendWizard().run()
    assert exc.value.code == 1


def test_empty_checkbox_returns_none_providers():
    """Deselecting all providers results in providers=None."""
    prefs = _run_wizard_with_mocks(
        select_map={
            "use case": "chat", "images": "No",
            "context length": "No requirement", "source region": "Any (default)", "Arena score": "1300 (default)",
        },
        checkbox_return=[],
        text_sequence=[""],
    )
    assert prefs.providers is None


def test_multiple_samples_blended_with_preset():
    """Multiple Q1.6 sample pairs are combined then blended 1:1 with preset."""
    from llmcost.pricing.config import PROVIDERS
    all_providers = list(PROVIDERS.keys())

    # summarization preset = 0.92
    # Pair 1: input=200, output=200 → 0.5; Pair 2: input=300, output=100 → 0.75
    # Combined sample: (200+300)/(800) = 0.625
    # Blended: (0.92 + 0.625) / 2 = 0.7725
    samples = ["a" * 200, "b" * 200, "c" * 300, "d" * 100]
    confirm_answers = iter([True, False])

    def mock_select(prompt, **kwargs):
        m = MagicMock()
        if "use case" in prompt:
            m.ask.return_value = "summarization"
        elif "images" in prompt:
            m.ask.return_value = "No"
        else:
            m.ask.return_value = kwargs.get("default", "")
        return m

    text_iter = iter(samples)

    def mock_text(prompt, **kwargs):
        m = MagicMock()
        try:
            m.ask.return_value = next(text_iter)
        except StopIteration:
            m.ask.return_value = ""
        return m

    def mock_confirm(prompt, **kwargs):
        m = MagicMock()
        try:
            m.ask.return_value = next(confirm_answers)
        except StopIteration:
            m.ask.return_value = False
        return m

    with (
        patch("llmcost.recommender.wizard.questionary.select", side_effect=mock_select),
        patch("llmcost.recommender.wizard.questionary.checkbox", return_value=MagicMock(ask=MagicMock(return_value=all_providers))),
        patch("llmcost.recommender.wizard.questionary.text", side_effect=mock_text),
        patch("llmcost.recommender.wizard.questionary.confirm", side_effect=mock_confirm),
        patch("sys.stdin.isatty", return_value=True),
    ):
        prefs = RecommendWizard().run()

    assert prefs.input_ratio_source == "blended"
    assert abs(prefs.input_ratio - 0.7725) < 0.001


def test_max_price_sota_model_selected():
    """Selecting a SOTA model stores its direct_id in max_price_model."""
    from llmcost.pricing.config import PROVIDERS
    all_providers = list(PROVIDERS.keys())

    prefs = _run_wizard_with_mocks(
        select_map={
            "use case": "chat", "images": "No",
            "context length": "No requirement", "source region": "Any (default)", "Arena score": "1300 (default)",
            "ceiling": "gpt-5.4",
        },
        checkbox_return=all_providers,
        text_sequence=[""],
    )
    assert prefs.max_price_model == "gpt-5.4"


def test_max_price_no_limit_is_none():
    """Selecting 'No limit' stores max_price_model=None."""
    from llmcost.pricing.config import PROVIDERS
    all_providers = list(PROVIDERS.keys())

    prefs = _run_wizard_with_mocks(
        select_map={
            "use case": "chat", "images": "No",
            "context length": "No requirement", "source region": "Any (default)", "Arena score": "1300 (default)",
            "ceiling": "No limit",
        },
        checkbox_return=all_providers,
        text_sequence=[""],
    )
    assert prefs.max_price_model is None
