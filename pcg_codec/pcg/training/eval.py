from __future__ import annotations

import argparse
import io
import json
import os
import struct
import time
from dataclasses import dataclass
from typing import Iterator, Optional

try:
    import resource  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    resource = None  # type: ignore[assignment]

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly  # type: ignore

from ..entropy.coder_rans import RansConfig
from ..entropy.dag import build_dag_layers
from ..entropy.streams import (
    ContainerHeader,
    FramePacket,
    LayeredEntropyDecoder,
    LayeredEntropyEncoder,
    StreamingContainerReader,
    stable_params_hash,
    write_container,
)
from ..model.quantizers import BlockCodebookQuantizer, FSQQuantizer
from .losses import MultiResolutionSTFTLoss, waveform_l1
from .metrics import bitrate_from_bytes, summarize_distribution
from .train import PCGComponents, _extract_audio, build_components


def evaluate(
    components: PCGComponents,
    dataloader,
    device: torch.device,
    stft_loss: Optional[MultiResolutionSTFTLoss] = None,
) -> dict:
    components.encoder.eval()
    components.decoder.eval()
    components.transform.eval()
    components.quantizer.eval()
    if components.prior is not None:
        components.prior.eval()

    metrics_t = {"loss": torch.zeros((), device=device), "recon": torch.zeros((), device=device)}
    steps = 0
    with torch.no_grad():
        for batch in dataloader:
            x = _extract_audio(batch).to(device)
            z = components.encoder(x)
            z = components.transform(z)
            batch_size, frames, latent_dim = z.shape
            z_flat = z.reshape(batch_size * frames, latent_dim)
            q_out = components.quantizer(z_flat)
            z_hat = q_out.z_hat.reshape(batch_size, frames, latent_dim)
            x_hat = components.decoder(z_hat)
            recon = waveform_l1(x, x_hat)
            if stft_loss is not None:
                recon = recon + stft_loss(x, x_hat)
            metrics_t["loss"] += recon.detach()
            metrics_t["recon"] += recon.detach()
            steps += 1
    if steps <= 0:
        return {"loss": 0.0, "recon": 0.0}
    return {k: float((v / steps).detach().float().cpu()) for k, v in metrics_t.items()}


def _load_yaml(path: str) -> dict:
    import yaml  # type: ignore

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _list_wavs(root: str) -> list[str]:
    wavs: list[str] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(".wav"):
                wavs.append(os.path.join(dirpath, name))
    return sorted(wavs)


def _hash_path(path: str, split_hash: str) -> int:
    h = stable_params_hash({"split_hash": split_hash, "path": path})
    return int(h[:8], 16) % 100


def _split_files(files: list[str], split_hash: str, split: str, train_pct: int, val_pct: int) -> list[str]:
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train|val|test")
    out = []
    for p in files:
        r = _hash_path(p, split_hash)
        if r < train_pct:
            bucket = "train"
        elif r < train_pct + val_pct:
            bucket = "val"
        else:
            bucket = "test"
        if bucket == split:
            out.append(p)
    return out


def _load_audio_mono(path: str, target_sr: int) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    if int(sr) != int(target_sr):
        g = int(np.gcd(int(sr), int(target_sr)))
        up = int(target_sr) // g
        down = int(sr) // g
        audio = resample_poly(audio, up=up, down=down).astype(np.float32, copy=False)
    return audio.astype(np.float32, copy=False)


def _load_hf_dataset(d: dict, split: str):
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Hugging Face dataset support requires `datasets`. "
            "Install with `pip install datasets` (audio is decoded via `soundfile`)."
        ) from e

    path = str(d["path"])
    name = d.get("name", None)
    split_map = d.get("split_map", {}) or {}
    hf_split = str(split_map.get(split, split))

    kwargs: dict = {}
    for k in ("revision", "data_dir", "cache_dir"):
        if k in d and d[k] is not None:
            kwargs[k] = d[k]
    if "trust_remote_code" in d:
        kwargs["trust_remote_code"] = bool(d["trust_remote_code"])
    if "streaming" in d:
        kwargs["streaming"] = bool(d["streaming"])

    ds = load_dataset(path, name, split=hf_split, **kwargs)
    if not hasattr(ds, "__len__"):  # pragma: no cover
        raise RuntimeError("hf_dataset with streaming=True is not supported for random segment sampling")
    return ds


def _audio_array_and_sr(value, target_sr: int) -> tuple[np.ndarray, int]:
    if isinstance(value, dict) and "array" in value:
        arr = np.asarray(value["array"])
        sr = int(value.get("sampling_rate", target_sr))
        return arr, sr
    if isinstance(value, dict) and "bytes" in value and value["bytes"] is not None:
        data = value["bytes"]
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("Expected audio bytes to be bytes-like")
        audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        return np.asarray(audio), int(sr)
    if isinstance(value, dict) and "path" in value and value["path"]:
        audio, sr = sf.read(str(value["path"]), dtype="float32", always_2d=False)
        return np.asarray(audio), int(sr)
    if hasattr(value, "array") and hasattr(value, "sampling_rate"):  # datasets Audio decoding type
        arr = np.asarray(value.array)
        sr = int(value.sampling_rate)
        return arr, sr
    if isinstance(value, np.ndarray):
        return value, int(target_sr)
    return np.asarray(value), int(target_sr)


def _load_audio_hf_example(example: dict, audio_column: str, target_sr: int) -> np.ndarray:
    if audio_column not in example:
        raise KeyError(f"Expected audio_column={audio_column!r} in dataset example")
    arr, sr = _audio_array_and_sr(example[audio_column], target_sr=target_sr)
    if arr.ndim == 2:
        arr = np.mean(arr, axis=1)
    arr = arr.astype(np.float32, copy=False)
    if int(sr) != int(target_sr):
        g = int(np.gcd(int(sr), int(target_sr)))
        up = int(target_sr) // g
        down = int(sr) // g
        arr = resample_poly(arr, up=up, down=down).astype(np.float32, copy=False)
    return arr.astype(np.float32, copy=False)


def _hf_get_raw_example(ds, index: int) -> dict:
    # Avoid datasets' feature decoding (Audio -> torchcodec) by reading the underlying Arrow row directly.
    table = getattr(ds, "data", None)
    if table is None and hasattr(ds, "_data"):
        backing = getattr(ds, "_data")
        for attr in ("table", "_table", "data"):
            candidate = getattr(backing, attr, None)
            if candidate is not None and hasattr(candidate, "slice"):
                table = candidate
                break
    if table is not None and hasattr(table, "slice"):
        try:
            sliced = table.slice(int(index), 1)
            if hasattr(sliced, "to_pylist"):
                rows = sliced.to_pylist()
                if rows:
                    return rows[0]
            if hasattr(sliced, "to_pydict"):
                d = sliced.to_pydict()
                return {k: (v[0] if isinstance(v, list) and v else v) for k, v in d.items()}
        except Exception:
            pass
    ex = ds[int(index)]
    return ex if isinstance(ex, dict) else dict(ex)


def _iter_eval_segments(
    files: list[str],
    num_batches: int,
    batch_size: int,
    segment_samples: int,
    hop_samples: int,
    sample_rate_hz: int,
    seed: int,
) -> Iterator[torch.Tensor]:
    rng = np.random.default_rng(seed)
    for _ in range(num_batches):
        batch = []
        for _b in range(batch_size):
            path = str(files[int(rng.integers(0, len(files)))])
            audio = _load_audio_mono(path, sample_rate_hz)
            if len(audio) < segment_samples:
                audio = np.pad(audio, (0, segment_samples - len(audio)))
            start = int(rng.integers(0, max(1, len(audio) - segment_samples + 1)))
            seg = audio[start : start + segment_samples]
            rem = len(seg) % hop_samples
            if rem != 0:
                seg = np.pad(seg, (0, hop_samples - rem))
            batch.append(torch.from_numpy(seg))
        yield torch.stack(batch, dim=0)


def _iter_eval_segments_hf(
    ds,
    num_batches: int,
    batch_size: int,
    segment_samples: int,
    hop_samples: int,
    sample_rate_hz: int,
    seed: int,
    audio_column: str = "audio",
    max_retries: int = 50,
) -> Iterator[torch.Tensor]:
    try:
        from datasets import Audio  # type: ignore

        ds = ds.cast_column(audio_column, Audio(decode=False))
    except Exception:
        pass

    rng = np.random.default_rng(seed)
    if len(ds) <= 0:
        raise RuntimeError("No examples found in Hugging Face dataset for the requested split")
    for _ in range(num_batches):
        batch = []
        for _b in range(batch_size):
            last_exc: Exception | None = None
            for _try in range(max_retries):
                i = int(rng.integers(0, len(ds)))
                try:
                    ex = _hf_get_raw_example(ds, i)
                    audio = _load_audio_hf_example(ex, audio_column=audio_column, target_sr=sample_rate_hz)
                    break
                except Exception as e:  # noqa: BLE001
                    last_exc = e
                    continue
            else:
                raise RuntimeError(
                    f"Failed to load audio from hf dataset after {max_retries} attempts; "
                    "check `audio_column` and your audio backend dependencies."
                ) from last_exc
            if len(audio) < segment_samples:
                audio = np.pad(audio, (0, segment_samples - len(audio)))
            start = int(rng.integers(0, max(1, len(audio) - segment_samples + 1)))
            seg = audio[start : start + segment_samples]
            rem = len(seg) % hop_samples
            if rem != 0:
                seg = np.pad(seg, (0, hop_samples - rem))
            batch.append(torch.from_numpy(seg))
        yield torch.stack(batch, dim=0)


def _si_sdr(x: torch.Tensor, x_hat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if x.dim() != 2 or x_hat.dim() != 2:
        raise ValueError("Expected (batch, time)")
    x = x - torch.mean(x, dim=-1, keepdim=True)
    x_hat = x_hat - torch.mean(x_hat, dim=-1, keepdim=True)
    alpha = torch.sum(x_hat * x, dim=-1, keepdim=True) / (torch.sum(x * x, dim=-1, keepdim=True) + eps)
    s_target = alpha * x
    e_noise = x_hat - s_target
    ratio = (torch.sum(s_target * s_target, dim=-1) + eps) / (torch.sum(e_noise * e_noise, dim=-1) + eps)
    return 10.0 * torch.log10(ratio)


def _pcg_container_header_end(data: bytes) -> int:
    # Container layout is defined in pcg_codec/pcg/entropy/streams.py.
    if len(data) < 9:
        raise ValueError("container too short to include header")
    if data[0:4] != b"PCGC":
        raise ValueError("Not a PCG container (bad magic)")
    header_len = struct.unpack_from("<I", data, 5)[0]
    return 9 + int(header_len)


@dataclass(frozen=True)
class StreamingEvalResult:
    bytes_total: int
    frames_total: int
    bitrate_bps_per_frame: list[float]
    recon_mrstft: float
    recon_wave_l1: float
    si_sdr_db: float
    ttfa_ms: float
    rtf_cpu_1: float
    rtf_cpu_4: float
    container_packets: list[FramePacket]


def evaluate_streaming(
    cfg: dict,
    ds_cfg: dict,
    components: PCGComponents,
    device: torch.device,
) -> StreamingEvalResult:
    progress = bool(cfg.get("evaluation", {}).get("progress", False)) or os.environ.get("PCG_EVAL_PROGRESS") == "1"
    streaming = cfg["streaming"]
    hop = int(streaming["hop_samples"])
    sr = int(streaming["sample_rate_hz"])
    lookahead = int(streaming.get("lookahead_samples", 0))

    if components.dag is None or components.prior is None:
        raise ValueError("components.dag and components.prior must be set for streaming evaluation")

    dag = components.dag
    prior = components.prior
    rans_cfg = RansConfig(precision_bits=16)
    entropy_encoder = LayeredEntropyEncoder(dag=dag, prior=prior, cfg=rans_cfg)
    entropy_decoder = LayeredEntropyDecoder(dag=dag, prior=prior, cfg=rans_cfg)

    dataset_name = str(cfg["dataset"])
    d = ds_cfg["datasets"][dataset_name]
    split = str(cfg.get("evaluation", {}).get("split", "test"))
    kind = str(d.get("kind", "local_wav_dir"))

    files: list[str] | None = None
    hf_ds = None
    hf_audio_column = str(d.get("audio_column", "audio"))

    if kind == "local_wav_dir":
        root = str(d["root"])
        files = _list_wavs(root)
        if not files:
            raise RuntimeError(f"No wav files found under {root}")

        split_hash = str(d["split_hash"])
        split_defaults = ds_cfg.get("defaults", {}).get("split", {})
        train_pct = int(split_defaults.get("train_pct", 80))
        val_pct = int(split_defaults.get("val_pct", 10))
        files = _split_files(files, split_hash, split=split, train_pct=train_pct, val_pct=val_pct)
        if not files:
            raise RuntimeError(f"No wav files found for split={split}")
    elif kind == "hf_dataset":
        hf_ds = _load_hf_dataset(d, split=split)
    else:
        raise RuntimeError(f"Unsupported datasets.kind={kind!r}; expected local_wav_dir|hf_dataset")

    seg_s = float(d.get("segment_seconds", ds_cfg.get("defaults", {}).get("segment_seconds", 4.0)))
    seg_samples = int(round(seg_s * sr))
    batch_size = int(cfg["training"]["batch_size"])
    num_batches = int(cfg.get("evaluation", {}).get("num_batches", 10))
    seed = int(cfg["training"]["seed"])
    if progress:
        frames_per_seg = int(np.ceil(seg_samples / max(hop, 1)))
        print(
            f"[eval] kind={kind} split={split} batches={num_batches} batch_size={batch_size} "
            f"sr={sr} hop={hop} seg_s={seg_s:g} frames/seg~{frames_per_seg} lookahead={lookahead}",
            flush=True,
        )

    stft_cfg = cfg["loss"]["mrstft"]
    stft_loss = None
    if bool(stft_cfg.get("enabled", True)):
        stft_loss = MultiResolutionSTFTLoss(
            fft_sizes=stft_cfg["fft_sizes"], hop_sizes=stft_cfg["hop_sizes"], win_lengths=stft_cfg["win_lengths"]
        ).to(device)

    components.encoder.eval()
    components.decoder.eval()
    components.transform.eval()
    components.quantizer.eval()
    prior.eval()

    bytes_total = 0
    frames_total = 0
    bps_per_frame: list[float] = []
    recon_mrstft_sum = 0.0
    recon_l1_sum = 0.0
    si_sdr_sum = 0.0
    sample_batches = 0

    # Compute TTFA on the first decoded frame of the first batch (per README definition).
    ttfa_ms: float | None = None
    container_packets: list[FramePacket] = []

    # Decode timing for RTF.
    decode_seconds_total = 0.0
    decode_time_1 = 0.0
    decode_time_4 = 0.0

    batch_iter: Iterator[torch.Tensor]
    if files is not None:
        batch_iter = _iter_eval_segments(
            files,
            num_batches=num_batches,
            batch_size=batch_size,
            segment_samples=seg_samples,
            hop_samples=hop,
            sample_rate_hz=sr,
            seed=seed,
        )
    else:
        if hf_ds is None:
            raise RuntimeError("hf_dataset was selected but dataset failed to load")
        batch_iter = _iter_eval_segments_hf(
            hf_ds,
            num_batches=num_batches,
            batch_size=batch_size,
            segment_samples=seg_samples,
            hop_samples=hop,
            sample_rate_hz=sr,
            seed=seed,
            audio_column=hf_audio_column,
        )

    for bi, batch in enumerate(batch_iter, start=1):
        if progress:
            print(f"[eval] batch {bi}/{num_batches}", flush=True)
        x = batch.to(device).float()

        # Streaming encode/decode each item independently (to respect causal state).
        x_hat_items = []
        for b in range(x.size(0)):
            x_b = x[b].detach().cpu().numpy()

            entropy_encoder.reset()
            entropy_decoder.reset()
            components.encoder.reset()
            components.decoder.reset()

            # Encode all frames and keep per-frame packets.
            packets: list[FramePacket] = []
            for t in range(0, len(x_b), hop):
                x_frame = x_b[t : t + hop]
                if len(x_frame) < hop:
                    x_frame = np.pad(x_frame, (0, hop - len(x_frame)))
                with torch.no_grad():
                    x_t = torch.from_numpy(x_frame).to(device=device, dtype=torch.float32)
                    z = components.encoder.encode_frame(x_t)  # (1, latent_dim)
                    z = components.transform(z)
                    q_out = components.quantizer.quantize(z)
                    q_frame = q_out.symbols.squeeze(0)
                    layer_bytes = entropy_encoder.encode_frame(q_frame)
                packets.append(FramePacket(frame_index=len(packets), layer_bytes=layer_bytes))
                bytes_frame = sum(len(lb) for lb in layer_bytes)
                bytes_total += int(bytes_frame)
                frames_total += 1
                bps_per_frame.append(8.0 * bytes_frame / (hop / sr))

            if ttfa_ms is None:
                chunk_bytes = int(cfg.get("evaluation", {}).get("ttfa_chunk_bytes", 256))
                if chunk_bytes <= 0:
                    chunk_bytes = 256

                header = ContainerHeader(
                    sample_rate_hz=sr,
                    hop_samples=hop,
                    encoder_lookahead_samples=lookahead,
                    codec_name=str(cfg.get("codec", "pcg")),
                    params_hash=stable_params_hash(cfg),
                )
                buf = io.BytesIO()
                write_container(buf, header=header, frames=packets)
                container_bytes = buf.getvalue()

                reader = StreamingContainerReader()
                header_end = _pcg_container_header_end(container_bytes)
                _ = reader.feed(container_bytes[:header_end])  # parse header (not timed)

                # Measure decode time from first post-header byte receipt to first PCM frame output.
                ttfa_decoder = LayeredEntropyDecoder(dag=dag, prior=prior, cfg=rans_cfg)
                ttfa_decoder.reset()
                components.decoder.reset()
                t0 = time.perf_counter()
                offset = header_end
                while offset < len(container_bytes):
                    chunk = container_bytes[offset : offset + chunk_bytes]
                    offset += len(chunk)
                    ready = reader.feed(chunk)
                    if not ready:
                        continue
                    pkt0 = ready[0]
                    ttfa_decoder.start_frame()
                    for li, lb in enumerate(pkt0.layer_bytes, start=1):
                        ttfa_decoder.push_layer_bytes(li, lb)
                        ttfa_decoder.entropy_decode_layer(li)
                    q_dec = ttfa_decoder.decoder_step()
                    if isinstance(components.quantizer, FSQQuantizer):
                        z_q = components.quantizer.dequantize(q_dec.unsqueeze(0))
                    elif isinstance(components.quantizer, BlockCodebookQuantizer):
                        z_q = components.quantizer.dequantize(q_dec.unsqueeze(0))
                    else:
                        raise RuntimeError("Unsupported quantizer type for streaming decode")
                    z_inv = components.transform.inverse(z_q)
                    _ = components.decoder.decode_frame(z_inv.squeeze(0))
                    t1 = time.perf_counter()
                    ttfa_ms = 1000.0 * (t1 - t0)
                    break

                # Restore decoder state for the main decode loop.
                entropy_decoder.reset()
                components.decoder.reset()

            # Decode frames back to audio.
            decoded_frames: list[torch.Tensor] = []
            for fi, pkt in enumerate(packets):
                entropy_decoder.start_frame()
                for li, lb in enumerate(pkt.layer_bytes, start=1):
                    entropy_decoder.push_layer_bytes(li, lb)
                    entropy_decoder.entropy_decode_layer(li)
                q_dec = entropy_decoder.decoder_step()
                if isinstance(components.quantizer, FSQQuantizer):
                    z_q = components.quantizer.dequantize(q_dec.unsqueeze(0))
                elif isinstance(components.quantizer, BlockCodebookQuantizer):
                    z_q = components.quantizer.dequantize(q_dec.unsqueeze(0))
                else:
                    raise RuntimeError("Unsupported quantizer type for streaming decode")
                z_inv = components.transform.inverse(z_q)
                x_hat = components.decoder.decode_frame(z_inv.squeeze(0))  # (1, hop)
                decoded_frames.append(x_hat.squeeze(0).detach().cpu())

            x_hat_items.append(torch.cat(decoded_frames, dim=0).numpy()[: len(x_b)])

            if not container_packets and b == 0:
                container_packets = packets

        x_hat = torch.from_numpy(np.stack(x_hat_items, axis=0)).to(device)

        # Quality metrics on this batch
        with torch.no_grad():
            l1 = waveform_l1(x, x_hat)
            recon_l1_sum += float(l1.detach().cpu())
            if stft_loss is not None:
                m = stft_loss(x, x_hat)
                recon_mrstft_sum += float(m.detach().cpu())
            else:
                recon_mrstft_sum += 0.0
            si = _si_sdr(x, x_hat)
            si_sdr_sum += float(torch.mean(si).detach().cpu())
        sample_batches += 1

        # Decode-only RTF measurement. Use the already-produced packets for item 0.
        audio_seconds = float(x.size(-1)) / float(sr)
        decode_seconds_total += audio_seconds
        packets0 = container_packets or packets

        for threads, acc in [(1, "decode_time_1"), (4, "decode_time_4")]:
            if progress and threads == 1:
                print(f"[eval] measuring RTF (threads=1,4) on {len(packets0)} packets...", flush=True)
            torch.set_num_threads(threads)
            entropy_decoder.reset()
            components.decoder.reset()
            t0 = time.perf_counter()
            for pkt in packets0:
                entropy_decoder.start_frame()
                for li, lb in enumerate(pkt.layer_bytes, start=1):
                    entropy_decoder.push_layer_bytes(li, lb)
                    entropy_decoder.entropy_decode_layer(li)
                q_dec = entropy_decoder.decoder_step()
                if isinstance(components.quantizer, FSQQuantizer):
                    z_q = components.quantizer.dequantize(q_dec.unsqueeze(0))
                else:
                    z_q = components.quantizer.dequantize(q_dec.unsqueeze(0))
                z_inv = components.transform.inverse(z_q)
                _ = components.decoder.decode_frame(z_inv.squeeze(0))
            t1 = time.perf_counter()
            if acc == "decode_time_1":
                decode_time_1 += t1 - t0
            else:
                decode_time_4 += t1 - t0

    if ttfa_ms is None:
        ttfa_ms = 0.0
    rtf_cpu_1 = (decode_time_1 / decode_seconds_total) if decode_seconds_total > 0 else 0.0
    rtf_cpu_4 = (decode_time_4 / decode_seconds_total) if decode_seconds_total > 0 else 0.0

    return StreamingEvalResult(
        bytes_total=int(bytes_total),
        frames_total=int(frames_total),
        bitrate_bps_per_frame=bps_per_frame,
        recon_mrstft=float(recon_mrstft_sum / max(sample_batches, 1)),
        recon_wave_l1=float(recon_l1_sum / max(sample_batches, 1)),
        si_sdr_db=float(si_sdr_sum / max(sample_batches, 1)),
        ttfa_ms=float(ttfa_ms),
        rtf_cpu_1=float(rtf_cpu_1),
        rtf_cpu_4=float(rtf_cpu_4),
        container_packets=container_packets,
    )


def evaluate_streaming_to_json(cfg: dict, ds_cfg: dict, run_dir: str, checkpoint_path: str) -> None:
    os.makedirs(run_dir, exist_ok=True)
    progress = bool(cfg.get("evaluation", {}).get("progress", False)) or os.environ.get("PCG_EVAL_PROGRESS") == "1"
    device = torch.device(str(cfg["training"]["device"]))
    components = build_components(cfg)
    ckpt = torch.load(checkpoint_path, map_location=device)
    components.encoder.load_state_dict(ckpt["encoder"])
    components.decoder.load_state_dict(ckpt["decoder"])
    components.transform.load_state_dict(ckpt["transform"])
    components.quantizer.load_state_dict(ckpt["quantizer"])
    if ckpt.get("prior") is not None and components.prior is not None:
        components.prior.load_state_dict(ckpt["prior"])
    if components.dag is None:
        # Rebuild DAG deterministically from config.
        dag_cfg = cfg["model"]["dag"]
        components.dag = build_dag_layers(
            blocks=int(cfg["model"]["num_blocks"]),
            depth=int(dag_cfg["depth"]),
            parents_k=int(dag_cfg["parents_k"]),
            seed=int(dag_cfg.get("seed", cfg["training"]["seed"])),
        )

    components.encoder.to(device)
    components.decoder.to(device)
    components.transform.to(device)
    components.quantizer.to(device)
    if components.prior is not None:
        components.prior.to(device)

    if progress:
        print(f"[eval] loading checkpoint: {checkpoint_path}", flush=True)
    res = evaluate_streaming(cfg, ds_cfg, components, device=device)

    def _count_params(module: torch.nn.Module) -> int:
        return int(sum(p.numel() for p in module.parameters()))

    params_total = (
        _count_params(components.encoder)
        + _count_params(components.decoder)
        + _count_params(components.transform)
        + _count_params(components.quantizer)
        + (_count_params(components.prior) if components.prior is not None else 0)
    )

    bitrate_bps = bitrate_from_bytes(
        res.bytes_total,
        res.frames_total,
        hop=int(cfg["streaming"]["hop_samples"]),
        sample_rate=int(cfg["streaming"]["sample_rate_hz"]),
    )
    dist = summarize_distribution(res.bitrate_bps_per_frame)

    def _peak_rss_kb() -> int:
        if resource is not None:
            return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        try:
            import psutil  # type: ignore

            return int(psutil.Process(os.getpid()).memory_info().rss // 1024)
        except Exception:
            return 0

    eval_json = {
        "run_id": os.path.basename(run_dir),
        "codec": str(cfg.get("codec", "pcg")),
        "dataset": {"name": str(cfg["dataset"])},
        "streaming": {
            "sample_rate_hz": int(cfg["streaming"]["sample_rate_hz"]),
            "hop_samples": int(cfg["streaming"]["hop_samples"]),
            "lookahead_samples": int(cfg["streaming"].get("lookahead_samples", 0)),
            "depth": int(cfg["model"]["dag"]["depth"]),
        },
        "metrics": {
            "bitrate": {
                "bps": float(bitrate_bps),
                "p5_bps": float(dist["p5"]),
                "p50_bps": float(dist["p50"]),
                "p95_bps": float(dist["p95"]),
                "bits_per_frame": float(8.0 * res.bytes_total / max(res.frames_total, 1)),
            },
            "quality": {
                "mrstft": float(res.recon_mrstft),
                "wave_l1": float(res.recon_wave_l1),
                "si_sdr_db": float(res.si_sdr_db),
            },
            "latency": {
                "ttfa_ms": float(res.ttfa_ms),
                "rtf_cpu_1": float(res.rtf_cpu_1),
                "rtf_cpu_4": float(res.rtf_cpu_4),
                "peak_rss_kb": int(_peak_rss_kb()),
            },
            "compute": {
                "params_total": int(params_total),
            },
        },
        "bytes_total": int(res.bytes_total),
        "frames_total": int(res.frames_total),
    }

    with open(os.path.join(run_dir, "eval.json"), "w", encoding="utf-8") as f:
        json.dump(eval_json, f, indent=2, sort_keys=False)

    # Write a container for reproducibility (first segment, first item).
    header = ContainerHeader(
        sample_rate_hz=int(cfg["streaming"]["sample_rate_hz"]),
        hop_samples=int(cfg["streaming"]["hop_samples"]),
        encoder_lookahead_samples=int(cfg["streaming"].get("lookahead_samples", 0)),
        codec_name=str(cfg.get("codec", "pcg")),
        params_hash=stable_params_hash(cfg),
    )
    with open(os.path.join(run_dir, "stream.pcg"), "wb") as f:
        write_container(f, header=header, frames=res.container_packets)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained run and write eval.json")
    parser.add_argument("--config", required=True)
    parser.add_argument("--datasets", default="pcg_codec/configs/datasets.yaml")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    ds_cfg = _load_yaml(args.datasets)
    evaluate_streaming_to_json(cfg, ds_cfg, run_dir=args.run_dir, checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()
