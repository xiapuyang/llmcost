"""Interactive recommendation wizard that collects user preferences."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import questionary
from questionary import Choice

from llmcost.pricing.config import PROVIDERS

# ── Use-case constants ─────────────────────────────────────────────────────
USE_CASES = [
    "Hermes (tool calling / function agent)",
    "OpenClaw (agent framework)",
    "chat",
    "summarization",
    "translation",
    "coding",
    "document writing",
    "text-to-image",
    "image editing",
]

IMAGE_USE_CASES = {"text-to-image", "image editing"}

# Default input_ratio per use case when Q1.6 samples are skipped.
_USE_CASE_RATIO_DEFAULTS: dict[str, float] = {
    "Hermes (tool calling / function agent)": 0.7,
    "OpenClaw (agent framework)": 0.7,
    "chat": 0.7,
    "summarization": 0.9,
    "translation": 0.5,
    "coding": 0.7,
    "document writing": 0.3,
}

# Map ratio value → display label (for Q2 select)
_RATIO_OPTIONS: list[tuple[str, float]] = [
    ("3:7  short input, long output (creative writing)", 0.3),
    ("5:5  equal length (translation)", 0.5),
    ("7:3  balanced", 0.7),
    ("9:1  long input, short output (summarization / RAG)", 0.9),
]

# Context length options
_CONTEXT_OPTIONS: list[tuple[str, int | None]] = [
    ("No requirement (default)", None),
    ("≥ 64K", 64_000),
    ("≥ 128K", 128_000),
    ("≥ 256K", 256_000),
    ("≥ 384K", 384_000),
    ("≥ 1M", 1_000_000),
]

# Cache hit ratio options
_CACHE_OPTIONS: list[tuple[str, float]] = [
    ("0%", 0.0),
    ("25%", 0.25),
    ("50% (default)", 0.5),
    ("75%", 0.75),
]

# Arena score threshold options
_ARENA_OPTIONS: list[tuple[str, int]] = [
    ("No limit (0)", 0),
    ("1200", 1200),
    ("1300 (default)", 1300),
    ("1350", 1350),
    ("1400", 1400),
]


@dataclass
class UserPreferences:
    """Collected user preferences for model recommendation."""

    use_case: str = "chat"
    vision_input: bool = False
    input_ratio: float = 0.7
    input_ratio_source: str = "preset"  # "preset" | "sample"
    min_context_length: int | None = None
    model_source: str = "any"  # "any" | "cn" | "us"
    cache_hit_ratio: float = 0.5
    min_arena_score: int = 1300
    providers: list[str] | None = None  # None means all providers


class RecommendWizard:
    """Interactive terminal wizard that collects UserPreferences."""

    def run(self) -> UserPreferences:
        """Run the wizard and return collected preferences.

        Returns:
            UserPreferences with user-selected values.

        Raises:
            SystemExit: On Ctrl+C (code 0) or non-interactive terminal (code 1).
        """
        if not sys.stdin.isatty():
            print("llmcost recommend requires an interactive terminal", file=sys.stderr)
            raise SystemExit(1)

        prefs = UserPreferences()

        # Q1: Use case
        use_case = self._ask(
            questionary.select(
                "Q1. Select your use case:",
                choices=USE_CASES,
                default="chat",
            )
        )
        prefs.use_case = use_case

        # Q1.5: Vision input
        vision_answer = self._ask(
            questionary.select(
                "Q1.5. Does your input include images?",
                choices=["No (default)", "Yes"],
                default="No (default)",
            )
        )
        prefs.vision_input = vision_answer == "Yes"

        # Q1.6: Optional sample input/output (skipped for image use cases and when vision_input=True)
        is_image_use_case = use_case in IMAGE_USE_CASES
        if not is_image_use_case and not prefs.vision_input:
            input_ratio = self._collect_samples()
            if input_ratio is not None:
                prefs.input_ratio = input_ratio
                prefs.input_ratio_source = "sample"
            else:
                prefs.input_ratio = self._ask_token_ratio(use_case)
                prefs.input_ratio_source = "preset"
        else:
            prefs.input_ratio = _USE_CASE_RATIO_DEFAULTS.get(use_case, 0.7)
            prefs.input_ratio_source = "preset"

        # Q3: Context length
        context_label = self._ask(
            questionary.select(
                "Q3. Minimum context length requirement:",
                choices=[label for label, _ in _CONTEXT_OPTIONS],
                default="No requirement (default)",
            )
        )
        prefs.min_context_length = dict(_CONTEXT_OPTIONS)[context_label]

        # Q4: Model source
        source_label = self._ask(
            questionary.select(
                "Q4. Model source region:",
                choices=["Any (default)", "US / International", "China only"],
                default="Any (default)",
            )
        )
        prefs.model_source = {"Any (default)": "any", "US / International": "us", "China only": "cn"}[source_label]

        # Q5: Cache hit ratio
        cache_label = self._ask(
            questionary.select(
                "Q5. Estimated cache hit ratio:",
                choices=[label for label, _ in _CACHE_OPTIONS],
                default="50% (default)",
            )
        )
        prefs.cache_hit_ratio = dict(_CACHE_OPTIONS)[cache_label]

        # Q6: Min Arena score
        arena_label = self._ask(
            questionary.select(
                "Q6. Minimum Arena score threshold:",
                choices=[label for label, _ in _ARENA_OPTIONS],
                default="1300 (default)",
            )
        )
        prefs.min_arena_score = dict(_ARENA_OPTIONS)[arena_label]

        # Q7: Provider subset
        all_providers = list(PROVIDERS.keys())
        selected = self._ask(
            questionary.checkbox(
                "Q7. Select providers to include (Space to toggle, Enter to confirm):",
                choices=[Choice(p, checked=True) for p in all_providers],
            )
        )
        prefs.providers = selected if selected else None

        return prefs

    def _collect_samples(self) -> float | None:
        """Prompt for optional sample input/output pairs and compute ratio.

        Returns:
            Computed input_ratio float, or None if user skipped.
        """
        in_chars_total = 0
        out_chars_total = 0

        first_input = self._ask(
            questionary.text(
                "Q1.6. Optional: paste a typical input sample to estimate token ratio (Enter to skip):"
            )
        )
        if not first_input:
            return None

        out_text = self._ask(questionary.text("     Corresponding output:"))
        in_chars_total += len(first_input)
        out_chars_total += len(out_text or "")

        while True:
            more = self._ask(questionary.confirm("     Add another sample?", default=False))
            if not more:
                break
            in_text = self._ask(questionary.text("     Input sample:"))
            out_text = self._ask(questionary.text("     Corresponding output:"))
            in_chars_total += len(in_text or "")
            out_chars_total += len(out_text or "")

        total = in_chars_total + out_chars_total
        if total == 0:
            return None
        return in_chars_total / total

    def _ask_token_ratio(self, use_case: str) -> float:
        """Ask Q2 (token ratio select) with a use-case-specific default cursor.

        Args:
            use_case: The selected use case from Q1.

        Returns:
            Selected input_ratio float.
        """
        default_ratio = _USE_CASE_RATIO_DEFAULTS.get(use_case, 0.7)
        default_label = next(
            (label for label, val in _RATIO_OPTIONS if val == default_ratio),
            _RATIO_OPTIONS[2][0],  # fall back to "7:3 balanced"
        )
        label = self._ask(
            questionary.select(
                "Q2. Input:output token ratio:",
                choices=[label for label, _ in _RATIO_OPTIONS],
                default=default_label,
            )
        )
        return dict(_RATIO_OPTIONS)[label]

    @staticmethod
    def _ask(question: questionary.Question) -> object:
        """Call .ask() and raise SystemExit(0) on Ctrl+C (None result).

        Args:
            question: A questionary Question object.

        Returns:
            The answered value.

        Raises:
            SystemExit: If the user cancels with Ctrl+C.
        """
        result = question.ask()
        if result is None:
            raise SystemExit(0)
        return result
