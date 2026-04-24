"""Tests for RecommendWizard and UserPreferences."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llmcost.recommender.wizard import RecommendWizard, UserPreferences, _RATIO_OPTIONS


# ── Helpers ────────────────────────────────────────────────────────────────

def _default_answers(overrides: dict | None = None) -> dict:
    """Return a mapping of questionary prompt substrings → answer values.

    All non-image defaults: Q1=chat, Q1.5=No, Q1.6=skip, Q2=7:3,
    Q3=No requirement, Q4=Any, Q5=50%, Q6=1300, Q7=all providers.
    """
    answers = {
        "Q1.": "chat",
        "Q1.5.": "No (default)",
        "Q1.6.": "",        # empty → skip → triggers Q2
        "Q2.": "7:3  balanced",
        "Q3.": "No requirement (default)",
        "Q4.": "Any (default)",
        "Q5.": "50% (default)",
        "Q6.": "1300 (default)",
        "Q7.": None,        # checkbox — handled separately
    }
    if overrides:
        answers.update(overrides)
    return answers


def _run_wizard_with_mocks(select_map: dict, checkbox_return: list, text_sequence: list | None = None):
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
        # use default if set in kwargs
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
        return make_question_mock(False)  # always "no more samples"

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
    """Accepting all defaults returns expected UserPreferences."""
    from llmcost.pricing.config import PROVIDERS
    all_providers = list(PROVIDERS.keys())

    prefs = _run_wizard_with_mocks(
        select_map={
            "Q1.": "chat",
            "Q1.5.": "No (default)",
            "Q2.": "7:3  balanced",
            "Q3.": "No requirement (default)",
            "Q4.": "Any (default)",
            "Q5.": "50% (default)",
            "Q6.": "1300 (default)",
        },
        checkbox_return=all_providers,
        text_sequence=[""],  # Q1.6 skipped
    )

    assert prefs.use_case == "chat"
    assert prefs.vision_input is False
    assert prefs.input_ratio == 0.7
    assert prefs.input_ratio_source == "preset"
    assert prefs.min_context_length is None
    assert prefs.model_source == "any"
    assert prefs.cache_hit_ratio == 0.5
    assert prefs.min_arena_score == 1300
    assert prefs.providers == all_providers
    assert prefs.max_price == 10.0


def test_sample_provided_skips_q2():
    """When Q1.6 sample is provided, input_ratio_source is 'sample' and Q2 is not asked."""
    from llmcost.pricing.config import PROVIDERS
    all_providers = list(PROVIDERS.keys())

    q2_called = []

    def mock_select(prompt, **kwargs):
        m = MagicMock()
        if "Q2." in prompt:
            q2_called.append(True)
            m.ask.return_value = "7:3  balanced"
        elif "Q1." in prompt and "Q1.5." not in prompt:
            m.ask.return_value = "chat"
        elif "Q1.5." in prompt:
            m.ask.return_value = "No (default)"
        else:
            m.ask.return_value = kwargs.get("default", "")
        return m

    # input=300 chars, output=100 chars → ratio = 300/400 = 0.75
    input_sample = "a" * 300
    output_sample = "b" * 100

    text_iter = iter([input_sample, output_sample])

    def mock_text(prompt, **kwargs):
        m = MagicMock()
        try:
            m.ask.return_value = next(text_iter)
        except StopIteration:
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

    assert prefs.input_ratio_source == "sample"
    assert abs(prefs.input_ratio - 0.75) < 0.001
    assert q2_called == [], "Q2 should not be called when samples are provided"


def test_summarization_use_case_q2_default():
    """use_case='summarization' sets Q2 default to 9:1."""
    from llmcost.pricing.config import PROVIDERS
    all_providers = list(PROVIDERS.keys())

    q2_defaults = []

    def mock_select(prompt, **kwargs):
        m = MagicMock()
        if "Q2." in prompt:
            q2_defaults.append(kwargs.get("default", ""))
            m.ask.return_value = kwargs.get("default", "")
        elif "Q1." in prompt and "Q1.5." not in prompt:
            m.ask.return_value = "summarization"
        elif "Q1.5." in prompt:
            m.ask.return_value = "No (default)"
        else:
            m.ask.return_value = kwargs.get("default", "")
        return m

    with (
        patch("llmcost.recommender.wizard.questionary.select", side_effect=mock_select),
        patch("llmcost.recommender.wizard.questionary.checkbox", return_value=MagicMock(ask=MagicMock(return_value=all_providers))),
        patch("llmcost.recommender.wizard.questionary.text", return_value=MagicMock(ask=MagicMock(return_value=""))),
        patch("llmcost.recommender.wizard.questionary.confirm", return_value=MagicMock(ask=MagicMock(return_value=False))),
        patch("sys.stdin.isatty", return_value=True),
    ):
        prefs = RecommendWizard().run()

    assert prefs.use_case == "summarization"
    assert prefs.input_ratio == 0.9
    # Q2 default should be the 9:1 label
    assert q2_defaults and "9:1" in q2_defaults[0]


def test_image_use_case_skips_q16_and_q2():
    """text-to-image use case skips Q1.6 sample step entirely."""
    from llmcost.pricing.config import PROVIDERS
    all_providers = list(PROVIDERS.keys())

    q16_called = []
    q2_called = []

    def mock_select(prompt, **kwargs):
        m = MagicMock()
        if "Q1.6." in prompt:
            q16_called.append(True)
            m.ask.return_value = ""
        elif "Q2." in prompt:
            q2_called.append(True)
            m.ask.return_value = "7:3  balanced"
        elif "Q1." in prompt and "Q1.5." not in prompt:
            m.ask.return_value = "text-to-image"
        elif "Q1.5." in prompt:
            m.ask.return_value = "No (default)"
        else:
            m.ask.return_value = kwargs.get("default", "")
        return m

    def mock_text(prompt, **kwargs):
        q16_called.append(True)
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
    assert q16_called == [], "Q1.6 text prompt should not be called for image use cases"
    assert q2_called == [], "Q2 should not be called for image use cases"


def test_vision_input_true_skips_q16():
    """vision_input=True skips Q1.6 sample step."""
    from llmcost.pricing.config import PROVIDERS
    all_providers = list(PROVIDERS.keys())

    text_called = []

    def mock_select(prompt, **kwargs):
        m = MagicMock()
        if "Q1." in prompt and "Q1.5." not in prompt:
            m.ask.return_value = "chat"
        elif "Q1.5." in prompt:
            m.ask.return_value = "Yes"  # vision input = True
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
        m.ask.return_value = None  # simulate Ctrl+C
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
    def mock_select(prompt, **kwargs):
        m = MagicMock()
        if "Q1." in prompt and "Q1.5." not in prompt:
            m.ask.return_value = "chat"
        elif "Q1.5." in prompt:
            m.ask.return_value = "No (default)"
        else:
            m.ask.return_value = kwargs.get("default", "")
        return m

    with (
        patch("llmcost.recommender.wizard.questionary.select", side_effect=mock_select),
        patch("llmcost.recommender.wizard.questionary.checkbox", return_value=MagicMock(ask=MagicMock(return_value=[]))),
        patch("llmcost.recommender.wizard.questionary.text", return_value=MagicMock(ask=MagicMock(return_value=""))),
        patch("llmcost.recommender.wizard.questionary.confirm", return_value=MagicMock(ask=MagicMock(return_value=False))),
        patch("sys.stdin.isatty", return_value=True),
    ):
        prefs = RecommendWizard().run()

    assert prefs.providers is None


def test_multiple_samples_averaged():
    """Multiple Q1.6 sample pairs are averaged for input_ratio."""
    from llmcost.pricing.config import PROVIDERS
    all_providers = list(PROVIDERS.keys())

    # Pair 1: input=200, output=200 → ratio 0.5
    # Pair 2: input=300, output=100 → ratio 0.75
    # Combined: (200+300)/(200+300+200+100) = 500/800 = 0.625
    samples = ["a" * 200, "b" * 200, "c" * 300, "d" * 100]
    confirm_answers = iter([True, False])  # add more: yes, then no

    def mock_select(prompt, **kwargs):
        m = MagicMock()
        if "Q1." in prompt and "Q1.5." not in prompt:
            m.ask.return_value = "chat"
        elif "Q1.5." in prompt:
            m.ask.return_value = "No (default)"
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

    assert prefs.input_ratio_source == "sample"
    assert abs(prefs.input_ratio - 0.625) < 0.001


def test_max_price_selected():
    """Selecting a non-default max price is stored as a float."""
    from llmcost.pricing.config import PROVIDERS
    all_providers = list(PROVIDERS.keys())

    prefs = _run_wizard_with_mocks(
        select_map={
            "Q1.": "chat",
            "Q1.5.": "No (default)",
            "Q2.": "7:3  balanced",
            "Q3.": "No requirement (default)",
            "Q4.": "Any (default)",
            "Q5.": "50% (default)",
            "Q6.": "1300 (default)",
            "Q8.": "$75/M",
        },
        checkbox_return=all_providers,
        text_sequence=[""],
    )
    assert prefs.max_price == 75.0


def test_max_price_no_limit_is_none():
    """Selecting 'No limit' stores max_price=None."""
    from llmcost.pricing.config import PROVIDERS
    all_providers = list(PROVIDERS.keys())

    prefs = _run_wizard_with_mocks(
        select_map={
            "Q1.": "chat",
            "Q1.5.": "No (default)",
            "Q2.": "7:3  balanced",
            "Q3.": "No requirement (default)",
            "Q4.": "Any (default)",
            "Q5.": "50% (default)",
            "Q6.": "1300 (default)",
            "Q8.": "No limit",
        },
        checkbox_return=all_providers,
        text_sequence=[""],
    )
    assert prefs.max_price is None
