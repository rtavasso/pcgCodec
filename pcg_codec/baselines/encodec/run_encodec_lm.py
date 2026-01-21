#!/usr/bin/env python3
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run EnCodec + token compression LM baseline.\n\n"
            "This reference implementation requires the official EnCodec LM code path. "
            "If your `encodec` installation exposes an LM tokenizer/compressor, wire it here."
        )
    )
    parser.add_argument("--config", default="pcg_codec/configs/encodec_baseline.yaml")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--input-wav", required=True)
    _ = parser.parse_args()
    raise SystemExit("encodec_lm baseline is not available in this environment (missing official LM code path).")


if __name__ == "__main__":
    main()
