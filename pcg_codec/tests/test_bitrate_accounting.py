from __future__ import annotations

from pcg_codec.pcg.training.metrics import bitrate_from_bytes


def test_bitrate_accounting_formula_units() -> None:
    # 24 kHz, hop=320 => 13.333...ms per frame.
    sample_rate = 24_000
    hop = 320
    num_frames = 100
    total_bytes = 12_000  # 96,000 bits

    seconds = num_frames * hop / sample_rate
    expected_bps = 8.0 * total_bytes / seconds
    got_bps = bitrate_from_bytes(total_bytes, num_frames, hop=hop, sample_rate=sample_rate)
    assert abs(got_bps - expected_bps) < 1e-9

