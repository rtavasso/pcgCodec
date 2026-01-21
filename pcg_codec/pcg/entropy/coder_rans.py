from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np


RANS_L = 1 << 23


@dataclass(frozen=True)
class RansConfig:
    precision_bits: int = 16

    @property
    def total_freq(self) -> int:
        return 1 << int(self.precision_bits)


def pmf_to_cdf(pmf: np.ndarray, cfg: RansConfig = RansConfig()) -> np.ndarray:
    """
    Convert a PMF (shape: (K,)) to an integer CDF (shape: (K+1,)) summing to cfg.total_freq.

    Requirements for rANS:
    - All frequencies are positive (>=1)
    - Sum(freq) == total_freq
    """
    pmf = np.asarray(pmf, dtype=np.float64)
    if pmf.ndim != 1:
        raise ValueError("pmf must be 1D")
    if np.any(pmf < 0):
        raise ValueError("pmf must be non-negative")
    s = float(pmf.sum())
    if not np.isfinite(s) or s <= 0:
        raise ValueError("pmf must have positive finite sum")
    pmf = pmf / s

    total = int(cfg.total_freq)
    if pmf.size > total:
        raise ValueError("Alphabet too large for chosen precision_bits (need K <= total_freq)")

    freq = np.floor(pmf * total).astype(np.int64)
    freq[freq <= 0] = 1

    diff = total - int(freq.sum())
    if diff > 0:
        # Add remaining counts to the largest fractional parts.
        frac = (pmf * total) - np.floor(pmf * total)
        order = np.argsort(-frac)
        i = 0
        while diff > 0:
            freq[order[i % len(order)]] += 1
            diff -= 1
            i += 1
    elif diff < 0:
        # Remove extra counts from the largest frequencies.
        order = np.argsort(-freq)
        i = 0
        while diff < 0:
            idx = order[i % len(order)]
            if freq[idx] > 1:
                freq[idx] -= 1
                diff += 1
            i += 1

    if int(freq.sum()) != total:
        raise RuntimeError("Failed to normalize frequencies to total_freq")

    cdf = np.zeros((freq.size + 1,), dtype=np.int64)
    np.cumsum(freq, out=cdf[1:])
    cdf[-1] = total
    return cdf


def _find_symbol(cdf: np.ndarray, value: int) -> int:
    # cdf is monotone and ends at total_freq.
    idx = int(np.searchsorted(cdf, value, side="right") - 1)
    if idx < 0 or idx >= cdf.size - 1:
        raise RuntimeError("value out of range for cdf")
    return idx


def rans_encode(symbols: list[int], cdfs: list[np.ndarray], cfg: RansConfig = RansConfig()) -> bytes:
    if len(symbols) != len(cdfs):
        raise ValueError("symbols and cdfs must have same length")
    if len(symbols) == 0:
        return struct.pack("<I", RANS_L)

    total = int(cfg.total_freq)
    precision_bits = int(cfg.precision_bits)
    out = bytearray()
    state = int(RANS_L)

    for sym, cdf in zip(reversed(symbols), reversed(cdfs)):
        sym = int(sym)
        cdf = np.asarray(cdf, dtype=np.int64)
        if cdf.ndim != 1 or cdf.size < 2 or int(cdf[-1]) != total:
            raise ValueError("Invalid CDF")
        if sym < 0 or sym >= cdf.size - 1:
            raise ValueError("Symbol out of range for CDF")
        start = int(cdf[sym])
        freq = int(cdf[sym + 1] - cdf[sym])
        if freq <= 0:
            raise ValueError("CDF has non-positive frequency")

        x_max = ((RANS_L >> precision_bits) << 8) * freq
        while state >= x_max:
            out.append(state & 0xFF)
            state >>= 8

        state = ((state // freq) << precision_bits) + (state % freq) + start

    out.extend(struct.pack("<I", state))
    return bytes(out)


def rans_decode(data: bytes, cdfs: list[np.ndarray], cfg: RansConfig = RansConfig()) -> list[int]:
    if len(data) < 4:
        raise ValueError("data too short")
    total = int(cfg.total_freq)
    precision_bits = int(cfg.precision_bits)
    mask = total - 1
    if (total & mask) != 0:
        raise ValueError("total_freq must be a power of two")

    idx = len(data)
    (state,) = struct.unpack("<I", data[idx - 4 : idx])
    idx -= 4

    out: list[int] = []
    for cdf in cdfs:
        cdf = np.asarray(cdf, dtype=np.int64)
        if cdf.ndim != 1 or cdf.size < 2 or int(cdf[-1]) != total:
            raise ValueError("Invalid CDF")
        value = state & mask
        sym = _find_symbol(cdf, int(value))
        start = int(cdf[sym])
        freq = int(cdf[sym + 1] - cdf[sym])
        state = freq * (state >> precision_bits) + (int(value) - start)

        while state < RANS_L:
            if idx <= 0:
                raise ValueError("ran out of renormalization bytes during decode")
            idx -= 1
            state = (state << 8) | int(data[idx])
        out.append(int(sym))

    return out

