from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True, order=True)
class SolveResult:
    # sort_index comes first so dataclass ordering uses it automatically
    sort_index: tuple[float, float, int] = field(init=False, repr=False)

    cipher_name: str
    plaintext: str
    key: Optional[str] = None

    # Higher is better
    score: float = 0.0
    confidence: float = 0.0

    # For transparency / debugging (why this was chosen)
    notes: str = ""

    # Extra metadata for future (e.g., detected encoding, key length, etc.)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # dataclass(order=True) sorts ascending; we want score/conf descending,
        # so we negate them. Add plaintext length as a stable, weak tie-break.
        object.__setattr__(self, "sort_index", (-self.score, -self.confidence, -len(self.plaintext)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "cipher_name": self.cipher_name,
            "plaintext": self.plaintext,
            "key": self.key,
            "score": self.score,
            "confidence": self.confidence,
            "notes": self.notes,
            "meta": dict(self.meta),
        }


@dataclass(frozen=True)
class TextFeatures:
    length: int
    unique_chars: int
    charset: str  # e.g. "A-Z", "A-Z0-9", "base64-ish", "binary-ish", "mixed"
    alpha_ratio: float
    digit_ratio: float
    space_ratio: float
    printable_ratio: float
    entropy: float
    ioc: float  # index of coincidence for A-Z only (0 if not applicable)

    def to_dict(self) -> dict[str, Any]:
        return {
            "length": self.length,
            "unique_chars": self.unique_chars,
            "charset": self.charset,
            "alpha_ratio": self.alpha_ratio,
            "digit_ratio": self.digit_ratio,
            "space_ratio": self.space_ratio,
            "printable_ratio": self.printable_ratio,
            "entropy": self.entropy,
            "ioc": self.ioc,
        }
