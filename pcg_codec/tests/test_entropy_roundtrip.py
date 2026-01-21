from __future__ import annotations

import numpy as np

from pcg_codec.pcg.entropy.coder_rans import RansConfig, pmf_to_cdf, rans_decode, rans_encode


def test_rans_roundtrip_random_symbols() -> None:
    rng = np.random.default_rng(0)
    cfg = RansConfig(precision_bits=12)  # smaller total for faster test
    k = 16
    n = 200

    symbols = [int(rng.integers(0, k)) for _ in range(n)]
    cdfs = []
    for _ in range(n):
        pmf = rng.random(k).astype(np.float64)
        pmf = pmf / pmf.sum()
        cdfs.append(pmf_to_cdf(pmf, cfg=cfg))

    data = rans_encode(symbols, cdfs, cfg=cfg)
    decoded = rans_decode(data, cdfs, cfg=cfg)
    assert decoded == symbols

