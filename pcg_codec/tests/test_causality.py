from __future__ import annotations

import numpy as np
import torch

from pcg_codec.pcg.model.decoder import StreamingDecoder
from pcg_codec.pcg.model.encoder import StreamingEncoder
from pcg_codec.pcg.model.quantizers import FSQQuantizer
from pcg_codec.pcg.model.transforms import IdentityTransform


def _stream_roundtrip(
    x: np.ndarray,
    hop: int,
    latent_dim: int,
    num_blocks: int,
    block_dim: int,
    *,
    encoder: StreamingEncoder | None = None,
    decoder: StreamingDecoder | None = None,
    transform: IdentityTransform | None = None,
    quantizer: FSQQuantizer | None = None,
) -> np.ndarray:
    if encoder is None:
        encoder = StreamingEncoder(hop=hop, lookahead=0, latent_dim=latent_dim, hidden_dim=64)
    if decoder is None:
        decoder = StreamingDecoder(hop=hop, latent_dim=latent_dim, hidden_dim=64)
    if transform is None:
        transform = IdentityTransform()
    if quantizer is None:
        quantizer = FSQQuantizer(levels=8)
        quantizer.configure_blocks(num_blocks=num_blocks, block_dim=block_dim)

    encoder.eval()
    decoder.eval()
    encoder.reset()
    decoder.reset()
    out = []
    with torch.no_grad():
        for t in range(0, len(x), hop):
            frame = x[t : t + hop]
            if len(frame) < hop:
                frame = np.pad(frame, (0, hop - len(frame)))
            z = encoder.encode_frame(torch.from_numpy(frame).float())
            z = transform(z)
            q = quantizer.quantize(z)
            x_hat = decoder.decode_frame(q.z_hat.squeeze(0)).squeeze(0).cpu().numpy()
            out.append(x_hat)
    return np.concatenate(out, axis=0)[: len(x)]


def test_causality_no_lookahead() -> None:
    hop = 16
    latent_dim = 32
    num_blocks = 8
    block_dim = 4
    rng = np.random.default_rng(0)

    torch.manual_seed(0)
    encoder = StreamingEncoder(hop=hop, lookahead=0, latent_dim=latent_dim, hidden_dim=64)
    decoder = StreamingDecoder(hop=hop, latent_dim=latent_dim, hidden_dim=64)
    transform = IdentityTransform()
    quantizer = FSQQuantizer(levels=8)
    quantizer.configure_blocks(num_blocks=num_blocks, block_dim=block_dim)

    x = rng.standard_normal(hop * 10).astype(np.float32)
    y1 = _stream_roundtrip(
        x,
        hop=hop,
        latent_dim=latent_dim,
        num_blocks=num_blocks,
        block_dim=block_dim,
        encoder=encoder,
        decoder=decoder,
        transform=transform,
        quantizer=quantizer,
    )

    # Corrupt future samples after time t and ensure earlier output unchanged.
    t = hop * 5
    x2 = x.copy()
    x2[t:] = rng.standard_normal(len(x2) - t).astype(np.float32)
    y2 = _stream_roundtrip(
        x2,
        hop=hop,
        latent_dim=latent_dim,
        num_blocks=num_blocks,
        block_dim=block_dim,
        encoder=encoder,
        decoder=decoder,
        transform=transform,
        quantizer=quantizer,
    )

    assert np.allclose(y1[:t], y2[:t], atol=1e-5, rtol=0.0)

