from __future__ import annotations

from statistics import median

from ciphercracker.core.features import ioc_scan
from ciphercracker.core.utils import normalize_az


def _has_periodicity_signal(
    ct: str,
    *,
    max_len: int,
    min_len: int,
    abs_thresh: float,
    peak_delta: float,
) -> bool:
    az = normalize_az(ct)
    if len(az) < min_len:
        return False

    scored = ioc_scan(ct, max_len=max_len)  # list[(k, avg_ioc)], sorted desc
    vals = [v for k, v in scored if k >= 2]
    if not vals:
        return False

    best = max(vals)
    med = median(vals)

    # Accept if it’s “pretty high” OR it’s a clear peak
    return (best >= abs_thresh) or ((best - med) >= peak_delta)


def should_try_vigenere(ct: str) -> bool:
    # More permissive: allows moderate-length texts to trigger Vigenère attempts
    return _has_periodicity_signal(
        ct,
        max_len=16,
        min_len=40,
        abs_thresh=0.058,
        peak_delta=0.010,
    )


def should_try_periodic_sub(ct: str) -> bool:
    # Periodic full-alphabet substitution needs more text, but don't be overly strict
    return _has_periodicity_signal(
        ct,
        max_len=16,
        min_len=80,
        abs_thresh=0.055,
        peak_delta=0.008,
    )
