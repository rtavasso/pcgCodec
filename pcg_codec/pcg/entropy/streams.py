from __future__ import annotations

import hashlib
import io
import json
import struct
from dataclasses import dataclass
from typing import Any, BinaryIO, Iterable, Iterator, Sequence

import numpy as np
import torch

from .coder_rans import RansConfig, pmf_to_cdf, rans_decode, rans_encode
from .dag import Dag
from .prior_model import LayeredCausalPrior

# =========================
# Container format (README 3.3)
# =========================

MAGIC = b"PCGC"
VERSION = 1


@dataclass(frozen=True)
class ContainerHeader:
    sample_rate_hz: int
    hop_samples: int
    encoder_lookahead_samples: int
    codec_name: str
    params_hash: str


@dataclass(frozen=True)
class FramePacket:
    frame_index: int
    layer_bytes: list[bytes]

    @property
    def depth(self) -> int:
        return len(self.layer_bytes)

    @property
    def payload_nbytes(self) -> int:
        return int(sum(len(b) for b in self.layer_bytes))


def stable_params_hash(params: Any) -> str:
    blob = json.dumps(params, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def write_container(fp: BinaryIO, header: ContainerHeader, frames: Iterable[FramePacket]) -> None:
    header_dict = {
        "sample_rate_hz": int(header.sample_rate_hz),
        "hop_samples": int(header.hop_samples),
        "encoder_lookahead_samples": int(header.encoder_lookahead_samples),
        "codec_name": str(header.codec_name),
        "params_hash": str(header.params_hash),
    }
    header_json = json.dumps(header_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )

    fp.write(MAGIC)
    fp.write(struct.pack("<B", VERSION))
    fp.write(struct.pack("<I", len(header_json)))
    fp.write(header_json)

    for frame in frames:
        if frame.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if frame.depth <= 0:
            raise ValueError("layer_bytes must be non-empty")
        fp.write(struct.pack("<I", int(frame.frame_index)))
        fp.write(struct.pack("<H", int(frame.depth)))
        for layer in frame.layer_bytes:
            fp.write(struct.pack("<I", len(layer)))
            fp.write(layer)


def read_container_header(fp: BinaryIO) -> ContainerHeader:
    magic = fp.read(4)
    if magic != MAGIC:
        raise ValueError("Not a PCG container (bad magic)")
    (version,) = struct.unpack("<B", fp.read(1))
    if version != VERSION:
        raise ValueError(f"Unsupported container version: {version}")
    (header_len,) = struct.unpack("<I", fp.read(4))
    header_json = fp.read(header_len)
    d = json.loads(header_json.decode("utf-8"))
    return ContainerHeader(
        sample_rate_hz=int(d["sample_rate_hz"]),
        hop_samples=int(d["hop_samples"]),
        encoder_lookahead_samples=int(d["encoder_lookahead_samples"]),
        codec_name=str(d["codec_name"]),
        params_hash=str(d["params_hash"]),
    )


def iter_container_frames(fp: BinaryIO) -> Iterator[FramePacket]:
    _ = read_container_header(fp)
    while True:
        raw = fp.read(4)
        if not raw:
            return
        (frame_index,) = struct.unpack("<I", raw)
        (depth,) = struct.unpack("<H", fp.read(2))
        layer_bytes: list[bytes] = []
        for _layer in range(depth):
            (n,) = struct.unpack("<I", fp.read(4))
            layer_bytes.append(fp.read(n))
        yield FramePacket(frame_index=int(frame_index), layer_bytes=layer_bytes)


class StreamingContainerReader:
    """Incremental reader for the container format to support TTFA simulation."""

    def __init__(self) -> None:
        self._buf = bytearray()
        self._header: ContainerHeader | None = None

    @property
    def header(self) -> ContainerHeader:
        if self._header is None:
            raise RuntimeError("Header not parsed yet")
        return self._header

    def feed(self, data: bytes) -> list[FramePacket]:
        self._buf.extend(data)
        out: list[FramePacket] = []
        view = memoryview(self._buf)
        offset = 0

        if self._header is None:
            if len(view) < 9:
                return []
            if bytes(view[0:4]) != MAGIC:
                raise ValueError("Not a PCG container (bad magic)")
            version = int(view[4])
            if version != VERSION:
                raise ValueError(f"Unsupported container version: {version}")
            header_len = struct.unpack_from("<I", view, 5)[0]
            if len(view) < 9 + header_len:
                return []
            header_json = bytes(view[9 : 9 + header_len])
            d: dict[str, Any] = json.loads(header_json.decode("utf-8"))
            self._header = ContainerHeader(
                sample_rate_hz=int(d["sample_rate_hz"]),
                hop_samples=int(d["hop_samples"]),
                encoder_lookahead_samples=int(d["encoder_lookahead_samples"]),
                codec_name=str(d["codec_name"]),
                params_hash=str(d["params_hash"]),
            )
            offset = 9 + header_len

        while True:
            if len(view) < offset + 6:
                break
            frame_index = struct.unpack_from("<I", view, offset)[0]
            depth = struct.unpack_from("<H", view, offset + 4)[0]
            tmp_off = offset + 6
            layers: list[bytes] = []
            ok = True
            for _layer in range(depth):
                if len(view) < tmp_off + 4:
                    ok = False
                    break
                n = struct.unpack_from("<I", view, tmp_off)[0]
                tmp_off += 4
                if len(view) < tmp_off + n:
                    ok = False
                    break
                layers.append(bytes(view[tmp_off : tmp_off + n]))
                tmp_off += n
            if not ok:
                break
            out.append(FramePacket(frame_index=int(frame_index), layer_bytes=layers))
            offset = tmp_off

        if offset > 0:
            # `memoryview(self._buf)` is an active export; release it before resizing.
            del view
            del self._buf[:offset]
        return out


def loads_container(data: bytes) -> tuple[ContainerHeader, list[FramePacket]]:
    fp = io.BytesIO(data)
    header = read_container_header(fp)
    frames = list(iter_container_frames(io.BytesIO(data)))
    return header, frames


# =========================
# Layered entropy streams (README 5.3)
# =========================


def _softmax_to_cdf(logits: torch.Tensor, cfg: RansConfig) -> np.ndarray:
    probs = torch.softmax(logits.float(), dim=-1).detach().cpu().numpy().astype(np.float64, copy=False)
    probs = probs + 1e-12
    probs = probs / float(np.sum(probs))
    return pmf_to_cdf(probs, cfg=cfg)


class LayeredEntropyEncoder:
    """Encodes one frame into D layer byte streams, using one rANS stream per layer."""

    def __init__(self, dag: Dag, prior: LayeredCausalPrior, cfg: RansConfig = RansConfig()) -> None:
        self.dag = dag
        self.prior = prior
        self.cfg = cfg
        self._state = self.prior.init_state(batch_size=1)

    def reset(self) -> None:
        self._state = self.prior.init_state(batch_size=1, device=next(self.prior.parameters()).device)

    def encode_frame(self, q_frame: torch.Tensor) -> list[bytes]:
        if q_frame.dim() != 1 or q_frame.numel() != self.prior.num_blocks:
            raise ValueError("q_frame must have shape (num_blocks,)")
        q_frame = q_frame.to(torch.long)

        layer_bytes: list[bytes] = []
        for layer in self.dag.layers:
            layer_blocks = list(layer)
            cdfs: list[np.ndarray] = []
            symbols: list[int] = []
            for b in layer_blocks:
                pa = self.dag.parents.get(int(b), [])
                parent_tokens = q_frame[pa].unsqueeze(0) if pa else None
                logits_b = self.prior.predict_block(self._state, int(b), parent_tokens=parent_tokens)
                cdfs.append(_softmax_to_cdf(logits_b[0], self.cfg))
                symbols.append(int(q_frame[b].item()))
            layer_bytes.append(rans_encode(symbols, cdfs, cfg=self.cfg))

        self._state = self.prior.update_state(self._state, q_frame.unsqueeze(0))
        return layer_bytes


class LayeredEntropyDecoder:
    """
    Decoder enforcing the README depth semantics:

    - A frame requires exactly D sequential calls to entropy_decode_layer(ℓ)
    - Only after all D layers are decoded can decoder_step() return the full frame symbols.
    """

    def __init__(self, dag: Dag, prior: LayeredCausalPrior, cfg: RansConfig = RansConfig()) -> None:
        self.dag = dag
        self.prior = prior
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self._state = self.prior.init_state(batch_size=1, device=next(self.prior.parameters()).device)
        self._layer_index = 0
        self._current_layer_bytes: list[bytes | None] | None = None
        self._q_frame = torch.full((self.prior.num_blocks,), -1, dtype=torch.long)

    def start_frame(self) -> None:
        self._current_layer_bytes = [None] * self.dag.depth
        self._layer_index = 0
        self._q_frame = torch.full((self.prior.num_blocks,), -1, dtype=torch.long, device=self._state.device)

    def push_layer_bytes(self, layer_1based: int, data: bytes) -> None:
        if self._current_layer_bytes is None:
            raise RuntimeError("start_frame() must be called before push_layer_bytes()")
        if layer_1based < 1 or layer_1based > self.dag.depth:
            raise ValueError("layer index out of range")
        self._current_layer_bytes[layer_1based - 1] = bytes(data)

    def entropy_decode_layer(self, layer_1based: int) -> torch.Tensor:
        if self._current_layer_bytes is None:
            raise RuntimeError("start_frame() must be called before decoding")
        if layer_1based != self._layer_index + 1:
            raise ValueError("Layers must be decoded in order")
        layer_data = self._current_layer_bytes[self._layer_index]
        if layer_data is None:
            raise RuntimeError("Layer bytes not available yet")
        layer_blocks = list(self.dag.layers[self._layer_index])
        cdfs: list[np.ndarray] = []
        for b in layer_blocks:
            pa = self.dag.parents.get(int(b), [])
            if pa:
                pt = self._q_frame[pa]
                if torch.any(pt < 0):
                    raise RuntimeError("Parent tokens missing; DAG violates layer ordering")
                parent_tokens = pt.unsqueeze(0)
            else:
                parent_tokens = None
            logits_b = self.prior.predict_block(self._state, int(b), parent_tokens=parent_tokens)
            cdfs.append(_softmax_to_cdf(logits_b[0], self.cfg))
        symbols = rans_decode(layer_data, cdfs, cfg=self.cfg)
        for b, s in zip(layer_blocks, symbols):
            self._q_frame[b] = int(s)
        self._layer_index += 1
        return torch.tensor(symbols, device=self._q_frame.device, dtype=torch.long)

    def decoder_step(self) -> torch.Tensor:
        if self._current_layer_bytes is None:
            raise RuntimeError("start_frame() must be called before decoder_step()")
        if self._layer_index != self.dag.depth:
            raise RuntimeError("Cannot produce frame until all layers decoded")
        q_frame = self._q_frame.clone()
        self._state = self.prior.update_state(self._state, q_frame.unsqueeze(0))
        self._current_layer_bytes = None
        return q_frame
