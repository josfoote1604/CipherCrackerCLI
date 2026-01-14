from __future__ import annotations

import math
from dataclasses import dataclass
from importlib import resources

from ciphercracker.core.utils import normalize_az


@dataclass
class QuadgramScorer:
    logp: dict[str, float]
    floor: float

    @classmethod
    def from_package_data(cls, filename: str = "english_quadgrams.txt") -> "QuadgramScorer":
        pkg = "ciphercracker.data"
        text = resources.files(pkg).joinpath(filename).read_text(encoding="utf-8")

        # Collect (gram -> numeric value) from any "GRAM <number>" style line.
        vals: dict[str, float] = {}
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue

            # allow separators like comma or equals
            line = line.replace("=", " ").replace(",", " ")
            parts = line.split()
            if len(parts) < 2:
                continue

            gram = parts[0].strip().upper()
            if len(gram) != 4 or not gram.isalpha():
                continue

            try:
                v = float(parts[1])
            except ValueError:
                continue

            vals[gram] = v

        if not vals:
            raise ValueError("No valid quadgram lines found. Expected lines like 'ABCD 1234'.")

        values = list(vals.values())

        # Infer what kind of numbers these are.
        # - if any value > 1.5 -> treat as counts
        # - else if all values between 0..1 -> treat as probabilities
        # - else if many negative -> treat as log10 probabilities
        any_big = any(v > 1.5 for v in values)
        all_prob = all(0.0 <= v <= 1.0 for v in values)
        many_negative = sum(1 for v in values if v < 0.0) > (0.5 * len(values))

        logp: dict[str, float] = {}

        if many_negative and not any_big and not all_prob:
            # assume already log10 probs
            logp = {g: float(v) for g, v in vals.items()}
            floor = min(logp.values()) - 1.0
            return cls(logp=logp, floor=floor)

        if all_prob and not any_big:
            # probabilities (normalize just in case)
            total = sum(values)
            if total <= 0:
                raise ValueError("Quadgram probabilities sum to <= 0.")
            logp = {g: math.log10(v / total) for g, v in vals.items() if v > 0}
            floor = math.log10((min(v for v in values if v > 0) / total) * 0.01)
            return cls(logp=logp, floor=floor)

        # counts
        total = sum(values)
        if total <= 0:
            raise ValueError("Quadgram counts sum to <= 0.")
        logp = {g: math.log10(v / total) for g, v in vals.items() if v > 0}
        floor = math.log10(0.01 / total)
        return cls(logp=logp, floor=floor)

    def score(self, text: str) -> float:
        s = normalize_az(text)
        if len(s) < 4:
            return float("-inf")
        total = 0.0
        for i in range(len(s) - 3):
            g = s[i:i + 4]
            total += self.logp.get(g, self.floor)
        return total
