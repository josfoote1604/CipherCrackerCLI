from __future__ import annotations

from statistics import median

from ciphercracker.core.features import analyze_text, ioc_scan
from ciphercracker.core.utils import normalize_az


def _has_periodicity_signal(
    ct: str,
    *,
    max_len: int,
    min_len: int,
    abs_thresh: float,
    peak_delta: float,
    min_alpha_ratio: float = 0.70,
    min_col_len: int = 6,
) -> bool:
    az = normalize_az(ct)
    if len(az) < min_len:
        return False

    info = analyze_text(ct)
    if info.get("alpha_ratio", 0.0) < min_alpha_ratio:
        return False

    scan = ioc_scan(ct, max_len=max_len)
    if not scan:
        return False

    usable = [(k, v) for (k, v) in scan if k >= 2 and (len(az) // k) >= min_col_len]
    if not usable:
        return False

    best_k, best = max(usable, key=lambda kv: kv[1])
    med = median(v for _, v in usable)

    # Accept if:
    #  - absolute IoC is reasonably high, OR
    #  - a clear peak exists vs the median across candidate k
    return (best >= abs_thresh) or ((best - med) >= peak_delta)


def should_try_vigenere(ct: str) -> bool:
    # Shorter min_len + peak test helps on ~60–120 char inputs.
    return _has_periodicity_signal(
        ct,
        max_len=16,
        min_len=40,
        abs_thresh=0.055,
        peak_delta=0.010,
        min_col_len=6,
    )


def should_try_periodic_sub(ct: str) -> bool:
    # Periodic substitution needs more letters per column to be meaningful.
    return _has_periodicity_signal(
        ct,
        max_len=16,
        min_len=80,
        abs_thresh=0.054,
        peak_delta=0.010,
        min_col_len=10,
    )
