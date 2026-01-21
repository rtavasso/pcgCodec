from __future__ import annotations

import torch

from pcg_codec.pcg.entropy.dag import build_dag_layers
from pcg_codec.pcg.entropy.prior_model import LayeredCausalPrior
from pcg_codec.pcg.entropy.streams import LayeredEntropyDecoder, LayeredEntropyEncoder


def test_depth_requires_d_sequential_layer_decodes() -> None:
    blocks = 9
    depth = 3
    parents_k = 2
    seed = 0
    codebook_size = 32

    dag = build_dag_layers(blocks=blocks, depth=depth, parents_k=parents_k, seed=seed)
    prior = LayeredCausalPrior(num_blocks=blocks, codebook_size=codebook_size, hidden_dim=32, embed_dim=16)

    enc = LayeredEntropyEncoder(dag=dag, prior=prior)
    dec = LayeredEntropyDecoder(dag=dag, prior=prior)

    q_frame = torch.randint(0, codebook_size, (blocks,), dtype=torch.long)
    layer_bytes = enc.encode_frame(q_frame)

    dec.start_frame()
    # Cannot produce output before decoding all layers.
    try:
        _ = dec.decoder_step()
        assert False, "expected decoder_step to fail before layers decoded"
    except RuntimeError:
        pass

    # Out-of-order decode should fail.
    dec.push_layer_bytes(2, layer_bytes[1])
    try:
        _ = dec.entropy_decode_layer(2)
        assert False, "expected out-of-order entropy_decode_layer to fail"
    except ValueError:
        pass

    # Decode in order.
    dec = LayeredEntropyDecoder(dag=dag, prior=prior)
    dec.start_frame()
    for i, lb in enumerate(layer_bytes, start=1):
        dec.push_layer_bytes(i, lb)
        dec.entropy_decode_layer(i)
    q_dec = dec.decoder_step()
    assert torch.equal(q_dec.cpu(), q_frame.cpu())

