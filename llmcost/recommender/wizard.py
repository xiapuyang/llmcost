"""Interactive recommendation wizard that collects user preferences."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import questionary
from questionary import Choice, Separator

from llmcost.pricing.config import PROVIDERS
from llmcost.pricing.filters.pipeline import CN_PROVIDERS

# ── Use-case registry (single source of truth) ────────────────────────────
@dataclass(frozen=True)
class UseCaseDef:
    """Definition of a use case and its token-ratio defaults."""

    id: str
    desc: str
    input_token_ratio: float
    cache_hit_ratio: float
    vision_default: bool = False
    default_context_length: int | None = None
    ask_sample_input: bool = False
    # Models supporting NONE of these parameters are eliminated (OR logic: keep if any match).
    # Models with supported_parameters=None (unreported) are kept as capability is unknown.
    required_parameters: tuple[str, ...] = ()
    # Models are ranked higher when they support more of these parameters.
    preferred_parameters: tuple[str, ...] = ()


_USE_CASE_REGISTRY: list[tuple[str, list[UseCaseDef]]] = [
    ("Agent / tool use", [
        UseCaseDef(
            "Hermes", "tool calling / function agent", 0.92, 0.60,
            vision_default=True, default_context_length=256_000,
            required_parameters=("tools", "tool_choice"),
            preferred_parameters=("parallel_tool_calls", "stop", "reasoning", "seed"),
        ),
        UseCaseDef(
            "OpenClaw", "multi-turn agent framework", 0.85, 0.45,
            vision_default=True, default_context_length=256_000,
            required_parameters=("tools", "tool_choice"),
            preferred_parameters=("parallel_tool_calls", "stop", "reasoning", "seed"),
        ),
    ]),
    ("QA / extraction", [
        UseCaseDef(
            "rag_qa", "retrieval-augmented QA", 0.88, 0.25,
            default_context_length=256_000, ask_sample_input=True,
            preferred_parameters=("stop", "response_format", "logprobs"),
        ),
        UseCaseDef(
            "long_context_qa", "fixed doc loaded once, queried repeatedly", 0.90, 0.85,
            default_context_length=256_000, ask_sample_input=True,
            preferred_parameters=("seed", "response_format"),
        ),
        UseCaseDef(
            "structured_extraction", "document → structured JSON fields", 0.95, 0.15,
            default_context_length=256_000, ask_sample_input=True,
            required_parameters=("response_format",),
            preferred_parameters=("structured_outputs", "logit_bias", "seed", "tool_choice"),
        ),
        UseCaseDef(
            "classification", "text → label", 0.97, 0.55,
            ask_sample_input=True,
            required_parameters=("logprobs",),
            preferred_parameters=("top_logprobs", "logit_bias", "response_format"),
        ),
    ]),
    ("Code", [
        UseCaseDef(
            "coding", "code completion / generation", 0.30, 0.50,
            default_context_length=256_000,
            preferred_parameters=("stop", "reasoning_effort"),
        ),
        UseCaseDef(
            "code_review", "diff + context → review comments", 0.80, 0.30,
            default_context_length=128_000,
            preferred_parameters=("response_format", "reasoning"),
        ),
        UseCaseDef(
            "test_generation", "function → test suite", 0.55, 0.40,
            default_context_length=128_000,
            preferred_parameters=("stop", "tools", "response_format"),
        ),
    ]),
    ("Conversation", [
        UseCaseDef(
            "chat", "conversational assistant", 0.75, 0.20,
            vision_default=True,
            preferred_parameters=("tools", "presence_penalty", "stop"),
        ),
        UseCaseDef(
            "customer_support", "support bot with long fixed system prompt", 0.82, 0.80,
            vision_default=True, ask_sample_input=True,
            preferred_parameters=("tools", "response_format", "seed"),
        ),
        UseCaseDef(
            "roleplay_creative", "character / story generation", 0.35, 0.65,
            default_context_length=128_000, ask_sample_input=True,
            preferred_parameters=("top_k", "min_p", "presence_penalty", "frequency_penalty", "repetition_penalty"),
        ),
        UseCaseDef(
            "email_drafting", "short instruction → full email", 0.30, 0.20,
            ask_sample_input=True,
            preferred_parameters=("stop", "response_format"),
        ),
    ]),
    ("Transform", [
        UseCaseDef(
            "summarization", "long document → concise summary", 0.92, 0.10,
            default_context_length=256_000, ask_sample_input=True,
            preferred_parameters=("response_format", "verbosity"),
        ),
        UseCaseDef(
            "translation", "source text → target language", 0.55, 0.15,
            default_context_length=128_000,
            preferred_parameters=("logit_bias", "stop"),
        ),
        UseCaseDef(
            "document writing", "short instruction → long document", 0.25, 0.10,
            preferred_parameters=("presence_penalty", "frequency_penalty", "reasoning"),
        ),
    ]),
    ("Reasoning", [
        UseCaseDef(
            "chain_of_thought_reasoning", "step-by-step reasoning (thinking tokens)", 0.12, 0.30,
            required_parameters=("reasoning",),
            preferred_parameters=("include_reasoning", "reasoning_effort", "seed"),
        ),
        UseCaseDef(
            "math_science_solving", "problem → detailed solution", 0.10, 0.25,
            ask_sample_input=True,
            required_parameters=("reasoning",),
            preferred_parameters=("include_reasoning", "reasoning_effort", "tools"),
        ),
    ]),
    ("Image", [
        UseCaseDef("text-to-image", "text prompt → image",                0.50, 0.10),
        UseCaseDef("image editing", "image + instruction → edited image", 0.50, 0.10, vision_default=True),
    ]),
]

# Derived constants
USE_CASES = [uc.id for _, cases in _USE_CASE_REGISTRY for uc in cases]
IMAGE_USE_CASES = {"text-to-image", "image editing"}
_USE_CASE_RATIO_DEFAULTS: dict[str, UseCaseDef] = {
    uc.id: uc for _, cases in _USE_CASE_REGISTRY for uc in cases
}


def _use_case_choices() -> list:
    choices: list = []
    for group, cases in _USE_CASE_REGISTRY:
        choices.append(Separator(f"── {group} ──"))
        for uc in cases:
            choices.append(Choice(title=f"{uc.id:<30}{uc.desc}", value=uc.id))
    return choices

# Context length options
_CONTEXT_OPTIONS: list[tuple[str, int | None]] = [
    ("No requirement", None),
    ("≥ 64K",  64_000),
    ("≥ 128K", 128_000),
    ("≥ 256K", 256_000),
    ("≥ 384K", 384_000),
    ("≥ 1M",   1_000_000),
]

# Arena score threshold options
_ARENA_OPTIONS: list[tuple[str, int]] = [
    ("No limit (0)", 0),
    ("1200", 1200),
    ("1300 (default)", 1300),
    ("1350", 1350),
    ("1400", 1400),
]

# SOTA benchmark models for max-price ceiling; (label, direct_id); empty id = no limit
_SOTA_MODELS: list[tuple[str, str]] = [
    ("No limit",                       ""),
    ("claude-opus-4.7 (default)",      "claude-opus-4.7"),
    ("claude-opus-4.6",                "claude-opus-4.6"),
    ("gpt-5.4",                        "gpt-5.4"),
    ("gemini-3.1-pro-preview",         "gemini-3.1-pro-preview"),
]


@dataclass
class UserPreferences:
    """Collected user preferences for model recommendation."""

    use_case: str = "Hermes"
    vision_input: bool = False
    input_ratio: float = 0.7
    input_ratio_source: str = "preset"  # "preset" | "blended" | "sample"
    min_context_length: int | None = None
    model_source: str = "any"  # "any" | "cn" | "us"
    cache_hit_ratio: float = 0.5
    min_arena_score: int = 1300
    providers: list[str] | None = None  # None means all providers
    max_price: float | None = None  # None means no limit ($/M tokens); fallback when max_price_model unresolved
    max_price_model: str | None = "claude-opus-4.7"  # direct_id of SOTA ceiling model; None = no limit
    require_cache_pricing: bool = True
    required_parameters: tuple[str, ...] = ()   # from UseCaseDef; models supporting none are eliminated (OR)
    preferred_parameters: tuple[str, ...] = ()  # from UseCaseDef; more = higher rank


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
                "Q. Select your use case:",
                choices=_use_case_choices(),
                default="Hermes",
            )
        )
        prefs.use_case = use_case
        _uc = _USE_CASE_RATIO_DEFAULTS.get(use_case)
        if _uc is not None:
            prefs.input_ratio = _uc.input_token_ratio
            prefs.cache_hit_ratio = _uc.cache_hit_ratio
            prefs.required_parameters = _uc.required_parameters
            prefs.preferred_parameters = _uc.preferred_parameters

        # Q1.5: Vision input — default from use-case definition
        vision_default = _uc.vision_default if _uc else False
        vision_answer = self._ask(
            questionary.select(
                "Q. Does your input include images?",
                choices=["No", "Yes"],
                default="Yes" if vision_default else "No",
            )
        )
        prefs.vision_input = vision_answer == "Yes"

        # Q1.6: Optional sample input/output (blended 1:1 with use-case preset)
        is_image_use_case = use_case in IMAGE_USE_CASES
        if _uc is not None and not is_image_use_case and _uc.ask_sample_input:
            sampled = self._collect_samples()
            if sampled is not None:
                prefs.input_ratio = (prefs.input_ratio + sampled) / 2
                prefs.input_ratio_source = "blended"
            else:
                prefs.input_ratio_source = "preset"
        else:
            prefs.input_ratio_source = "preset"

        # Q3: Context length — default from use-case definition
        ctx_default = _uc.default_context_length if _uc else None
        ctx_default_label = next(
            (label for label, val in _CONTEXT_OPTIONS if val == ctx_default),
            _CONTEXT_OPTIONS[0][0],
        )
        context_label = self._ask(
            questionary.select(
                "Q. Minimum context length requirement:",
                choices=[label for label, _ in _CONTEXT_OPTIONS],
                default=ctx_default_label,
            )
        )
        prefs.min_context_length = dict(_CONTEXT_OPTIONS)[context_label]

        # Q4: Model source
        source_label = self._ask(
            questionary.select(
                "Q. Model source region:",
                choices=["Any (default)", "US / International", "China only"],
                default="Any (default)",
            )
        )
        prefs.model_source = {"Any (default)": "any", "US / International": "us", "China only": "cn"}[source_label]

        # Q5: Cache pricing requirement
        cache_answer = self._ask(
            questionary.select(
                "Q. Require prompt caching support?",
                choices=["No", "Yes"],
                default="Yes",
            )
        )
        prefs.require_cache_pricing = cache_answer == "Yes"

        # Q6: Min Arena score
        arena_label = self._ask(
            questionary.select(
                "Q. Minimum Arena score threshold:",
                choices=[label for label, _ in _ARENA_OPTIONS],
                default="1300 (default)",
            )
        )
        prefs.min_arena_score = dict(_ARENA_OPTIONS)[arena_label]

        # Q7: Provider subset — pre-check based on Q4 model source
        all_providers = list(PROVIDERS.keys())
        if prefs.model_source == "cn":
            default_checked = {p for p in all_providers if p in CN_PROVIDERS}
        elif prefs.model_source == "us":
            default_checked = {p for p in all_providers if p not in CN_PROVIDERS}
        else:
            default_checked = set(all_providers)
        selected = self._ask(
            questionary.checkbox(
                "Q. Select providers to include (Space to toggle, Enter to confirm):",
                choices=[Choice(p, checked=(p in default_checked)) for p in all_providers],
            )
        )
        prefs.providers = selected if selected else None

        # Q8: Max price
        sota_label = self._ask(
            questionary.select(
                "Q. Price ceiling — exclude models costlier than:",
                choices=[label for label, _ in _SOTA_MODELS],
                default="claude-opus-4.7 (default)",
            )
        )
        prefs.max_price_model = dict(_SOTA_MODELS)[sota_label] or None

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
                "Q. Paste a typical input sample for better accuracy (Enter to skip and use preset):"
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
