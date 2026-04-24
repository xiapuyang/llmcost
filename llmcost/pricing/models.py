"""ModelRecord dataclass and modality parsing utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


def parse_modality(modality_raw: str) -> tuple[list[str], list[str]]:
    """Split 'text+image->text' into (['text','image'], ['text']).

    Args:
        modality_raw: A modality string in format 'input1+input2->output1+output2'.

    Returns:
        A tuple of (input_modalities, output_modalities) lists.
    """
    if "->" not in modality_raw:
        return [modality_raw], [modality_raw]
    left, right = modality_raw.split("->", 1)
    return left.split("+"), right.split("+")


def derive_category(output_modalities: list[str]) -> str:
    """Derive category string from output modality list.

    Priority: image > video > audio > text.

    Args:
        output_modalities: List of output modalities.

    Returns:
        The category string.
    """
    if "image" in output_modalities:
        return "image"
    if "video" in output_modalities:
        return "video"
    if "audio" in output_modalities:
        return "audio"
    return "text"


@dataclass
class ModelRecord:
    """A single model's pricing record. weighted_price is never stored."""

    id: str
    name: str
    provider: str
    provider_name: str
    pricing_url: str
    modality_raw: str
    input_modalities: list[str]
    output_modalities: list[str]
    category: str

    context_length: int | None = None
    max_output_tokens: int | None = None
    input_per_mtok: float | None = None
    output_per_mtok: float | None = None
    cache_read_per_mtok: float | None = None
    cache_write_per_mtok: float | None = None
    image_per_unit: float | None = None
    image_unit: str | None = None

    source: str = "openrouter"
    fetched_at: str = ""
    arena_score: int | None = None
    blacklisted: bool = False
    opensource: bool = False
    direct_id: str | None = None  # provider's native API model name (if differs from slug)
    notes: str | None = None

    # Arena per-category raw scores used to compute arena_score
    arena_scores_detail: dict[str, int] | None = None

    # Fields that were patched by overrides.yaml (used to skip drift comparison selectively)
    overridden_fields: list[str] | None = None

    # OpenRouter-specific metadata
    supported_parameters: list[str] | None = None  # e.g. ["tools", "reasoning", "max_tokens"]
    per_request_limits: dict | None = None          # {"prompt_tokens": "...", "completion_tokens": "..."}
    description: str | None = None
    created: int | None = None                      # Unix timestamp

    def to_dict(self) -> dict[str, Any]:
        """Convert ModelRecord to dictionary.

        Returns:
            Dictionary representation of the record.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelRecord:
        """Create ModelRecord from dictionary.

        Args:
            d: Dictionary with model record fields.

        Returns:
            A new ModelRecord instance.
        """
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
