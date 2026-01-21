"""Training loop utilities.

Includes a CLI runner that trains a config and writes `checkpoint.pt` and `eval.json`
to the run directory, as required by the experimentation spec.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly  # type: ignore

from ..entropy.dag import Dag, build_dag_layers
from ..entropy.prior_model import LayeredCausalPrior
from ..model.decoder import StreamingDecoder
from ..model.encoder import StreamingEncoder
from ..model.quantizers import BlockCodebookQuantizer, FSQQuantizer
from ..model.transforms import IdentityTransform, MixingTransform
from .losses import MultiResolutionSTFTLoss, sensitivity_equalization, waveform_l1


@dataclass
class PCGComponents:
    encoder: torch.nn.Module
    decoder: torch.nn.Module
    transform: torch.nn.Module
    quantizer: torch.nn.Module
    prior: Optional[LayeredCausalPrior] = None
    dag: Optional[Dag] = None


def _extract_audio(batch) -> torch.Tensor:
    if torch.is_tensor(batch):
        return batch
    if isinstance(batch, dict):
        for key in ("audio", "waveform", "x"):
            if key in batch:
                return batch[key]
    if isinstance(batch, (list, tuple)):
        return batch[0]
    raise TypeError("Unsupported batch type")


def _infer_num_blocks(quantizer: torch.nn.Module, symbols: torch.Tensor) -> int:
    if hasattr(quantizer, "num_blocks"):
        return int(getattr(quantizer, "num_blocks"))
    return symbols.size(-1)


def teacher_forced_rate_loss(prior: LayeredCausalPrior, dag: Dag, q_frames: torch.Tensor) -> torch.Tensor:
    """Teacher-forced -log p(q) consistent with the DAG factorization."""
    if q_frames.dim() != 3:
        raise ValueError("q_frames must have shape (batch, time, num_blocks)")
    batch, time, num_blocks = q_frames.shape
    if num_blocks != prior.num_blocks:
        raise ValueError("q_frames num_blocks mismatch prior")
    state = prior.init_state(batch_size=batch, device=q_frames.device)
    total = torch.zeros((), device=q_frames.device)
    count = 0
    for t in range(time):
        q_t = q_frames[:, t, :].long()
        for layer in dag.layers:
            for b in layer:
                pa = dag.parents.get(int(b), [])
                parent_tokens = q_t[:, pa] if pa else None
                logits = prior.predict_block(state, int(b), parent_tokens=parent_tokens)
                total = total + F.cross_entropy(logits, q_t[:, int(b)], reduction="sum")
                count += batch
        state = prior.update_state(state, q_t)
    return total / max(count, 1)


def train_step(
    components: PCGComponents,
    batch,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    stft_loss: Optional[MultiResolutionSTFTLoss] = None,
    lambda_rate: float = 0.0,
    beta_sens: float = 0.0,
) -> dict:
    components.encoder.train()
    components.decoder.train()
    components.transform.train()
    components.quantizer.train()
    if components.prior is not None:
        components.prior.train()
    if components.prior is not None and components.dag is None:
        raise ValueError("components.dag must be set when components.prior is used")

    x = _extract_audio(batch).to(device)
    optimizer.zero_grad(set_to_none=True)
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

    rate_loss = torch.tensor(0.0, device=x.device)
    if components.prior is not None and lambda_rate > 0.0:
        symbols = q_out.symbols.reshape(batch_size, frames, -1)
        rate_loss = teacher_forced_rate_loss(components.prior, components.dag, symbols)

    sens_loss = torch.tensor(0.0, device=x.device)
    if beta_sens > 0.0:
        num_blocks = _infer_num_blocks(components.quantizer, q_out.symbols.reshape(batch_size, frames, -1))
        sens_loss = sensitivity_equalization(z_flat, recon, num_blocks)

    total = recon + lambda_rate * rate_loss + beta_sens * sens_loss
    total.backward()
    optimizer.step()

    return {
        "loss": float(total.detach().cpu()),
        "recon": float(recon.detach().cpu()),
        "rate": float(rate_loss.detach().cpu()),
        "sens": float(sens_loss.detach().cpu()),
    }


def train_one_epoch(
    components: PCGComponents,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    stft_loss: Optional[MultiResolutionSTFTLoss] = None,
    lambda_rate: float = 0.0,
    beta_sens: float = 0.0,
) -> dict:
    metrics = {"loss": 0.0, "recon": 0.0, "rate": 0.0, "sens": 0.0}
    steps = 0
    for batch in dataloader:
        out = train_step(
            components,
            batch,
            optimizer,
            device,
            stft_loss=stft_loss,
            lambda_rate=lambda_rate,
            beta_sens=beta_sens,
        )
        for key in metrics:
            metrics[key] += out[key]
        steps += 1
    if steps > 0:
        for key in metrics:
            metrics[key] /= steps
    return metrics


def _load_yaml(path: str) -> dict:
    import yaml  # type: ignore

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _hash_path(path: str, split_hash: str) -> int:
    h = hashlib.sha256((split_hash + ":" + path).encode("utf-8")).hexdigest()
    return int(h[:8], 16) % 100


def _list_wavs(root: str) -> list[str]:
    wavs: list[str] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(".wav"):
                wavs.append(os.path.join(dirpath, name))
    return sorted(wavs)


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


def _iter_batches(
    files: list[str],
    batch_size: int,
    segment_samples: int,
    hop_samples: int,
    sample_rate_hz: int,
    seed: int,
) -> Iterator[torch.Tensor]:
    rng = np.random.default_rng(seed)
    if not files:
        raise RuntimeError("No audio files found for the requested split")
    while True:
        batch = []
        for _ in range(batch_size):
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


def build_components(cfg: dict) -> PCGComponents:
    streaming = cfg["streaming"]
    model_cfg = cfg["model"]
    hop = int(streaming["hop_samples"])
    lookahead = int(streaming.get("lookahead_samples", 0))

    latent_dim = int(model_cfg["latent_dim"])
    num_blocks = int(model_cfg["num_blocks"])
    block_dim = int(model_cfg["block_dim"])
    if latent_dim != num_blocks * block_dim:
        raise ValueError("model.latent_dim must equal model.num_blocks * model.block_dim")

    encoder = StreamingEncoder(
        hop=hop,
        lookahead=lookahead,
        latent_dim=latent_dim,
        hidden_dim=int(model_cfg["encoder"]["hidden_dim"]),
    )
    decoder = StreamingDecoder(
        hop=hop,
        latent_dim=latent_dim,
        hidden_dim=int(model_cfg["decoder"]["hidden_dim"]),
    )

    tf_type = str(model_cfg["transform"]["type"])
    transform = MixingTransform(latent_dim) if tf_type == "mixing" else IdentityTransform()

    q_type = str(model_cfg["quantizer"]["type"])
    if q_type == "fsq":
        levels = int(model_cfg["quantizer"]["levels"])
        quantizer = FSQQuantizer(levels=levels)
        quantizer.configure_blocks(num_blocks=num_blocks, block_dim=block_dim)
        codebook_size = int(levels) ** int(block_dim)
    elif q_type == "block_codebook":
        codebook_size = int(model_cfg["quantizer"]["codebook_size"])
        quantizer = BlockCodebookQuantizer(num_blocks=num_blocks, block_dim=block_dim, codebook_size=codebook_size)
    else:
        raise ValueError("model.quantizer.type must be fsq|block_codebook")

    dag_cfg = model_cfg["dag"]
    dag = build_dag_layers(
        blocks=num_blocks,
        depth=int(dag_cfg["depth"]),
        parents_k=int(dag_cfg["parents_k"]),
        seed=int(dag_cfg.get("seed", cfg["training"]["seed"])),
    )
    prior_cfg = model_cfg["prior"]
    prior = LayeredCausalPrior(
        num_blocks=num_blocks,
        codebook_size=codebook_size,
        hidden_dim=int(prior_cfg["hidden_dim"]),
        embed_dim=int(prior_cfg["embed_dim"]),
    )

    return PCGComponents(
        encoder=encoder,
        decoder=decoder,
        transform=transform,
        quantizer=quantizer,
        prior=prior,
        dag=dag,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PCG-Codec config and write artifacts.")
    parser.add_argument("--config", required=True, help="Run config YAML (e.g. generated by run_ablation_grid.py).")
    parser.add_argument("--datasets", default="pcg_codec/configs/datasets.yaml", help="Dataset registry YAML.")
    parser.add_argument("--run-dir", required=True, help="Output run directory for checkpoint.pt and eval.json.")
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    ds_cfg = _load_yaml(args.datasets)
    os.makedirs(args.run_dir, exist_ok=True)
    from .eval import evaluate_streaming_to_json  # local import to avoid circular import

    seed = int(cfg["training"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(str(cfg["training"]["device"]))
    components = build_components(cfg)
    components.encoder.to(device)
    components.decoder.to(device)
    components.transform.to(device)
    components.quantizer.to(device)
    if components.prior is not None:
        components.prior.to(device)

    dataset_name = str(cfg["dataset"])
    d = ds_cfg["datasets"][dataset_name]
    if d["kind"] != "local_wav_dir":
        raise SystemExit("Only datasets.kind=local_wav_dir is supported in this reference implementation")
    root = str(d["root"])
    files = _list_wavs(root)
    split_hash = str(d["split_hash"])
    split_defaults = ds_cfg.get("defaults", {}).get("split", {})
    train_pct = int(split_defaults.get("train_pct", 80))
    val_pct = int(split_defaults.get("val_pct", 10))
    hop = int(cfg["streaming"]["hop_samples"])
    sr = int(cfg["streaming"]["sample_rate_hz"])
    seg_s = float(d.get("segment_seconds", ds_cfg.get("defaults", {}).get("segment_seconds", 4.0)))
    seg_samples = int(round(seg_s * sr))

    train_files = _split_files(files, split_hash, "train", train_pct=train_pct, val_pct=val_pct)
    train_iter = _iter_batches(
        train_files,
        batch_size=int(cfg["training"]["batch_size"]),
        segment_samples=seg_samples,
        hop_samples=hop,
        sample_rate_hz=sr,
        seed=seed,
    )

    stft_cfg = cfg["loss"]["mrstft"]
    stft_loss = None
    if bool(stft_cfg.get("enabled", True)):
        stft_loss = MultiResolutionSTFTLoss(
            fft_sizes=stft_cfg["fft_sizes"],
            hop_sizes=stft_cfg["hop_sizes"],
            win_lengths=stft_cfg["win_lengths"],
        )

    params = (
        list(components.encoder.parameters())
        + list(components.decoder.parameters())
        + list(components.transform.parameters())
        + list(components.quantizer.parameters())
        + (list(components.prior.parameters()) if components.prior is not None else [])
    )
    optimizer = torch.optim.Adam(params, lr=float(cfg["training"]["lr"]))

    lambda_rd = float(cfg["training"]["lambda_rd"])
    sens_cfg = cfg["loss"]["sens_eq"]
    beta_sens = float(sens_cfg["weight"]) if bool(sens_cfg.get("enabled", False)) else 0.0
    steps = int(cfg["training"]["steps"])
    log_every = int(cfg["training"]["log_every"])
    eval_every = int(cfg["training"]["eval_every"])

    log_path = os.path.join(args.run_dir, "train_log.jsonl")
    t_start = time.perf_counter()
    for step in range(1, steps + 1):
        batch = next(train_iter).to(device)
        out = train_step(
            components,
            batch,
            optimizer,
            device,
            stft_loss=stft_loss,
            lambda_rate=lambda_rd,
            beta_sens=beta_sens,
        )

        if step % log_every == 0:
            out["step"] = step
            out["wall_s"] = time.perf_counter() - t_start
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(out) + "\n")

        if step % eval_every == 0:
            ckpt_path = os.path.join(args.run_dir, "checkpoint.pt")
            torch.save(
                {
                    "cfg": cfg,
                    "encoder": components.encoder.state_dict(),
                    "decoder": components.decoder.state_dict(),
                    "transform": components.transform.state_dict(),
                    "quantizer": components.quantizer.state_dict(),
                    "prior": components.prior.state_dict() if components.prior is not None else None,
                },
                ckpt_path,
            )
            evaluate_streaming_to_json(cfg, ds_cfg, run_dir=args.run_dir, checkpoint_path=ckpt_path)

    ckpt_path = os.path.join(args.run_dir, "checkpoint.pt")
    torch.save(
        {
            "cfg": cfg,
            "encoder": components.encoder.state_dict(),
            "decoder": components.decoder.state_dict(),
            "transform": components.transform.state_dict(),
            "quantizer": components.quantizer.state_dict(),
            "prior": components.prior.state_dict() if components.prior is not None else None,
        },
        ckpt_path,
    )
    evaluate_streaming_to_json(cfg, ds_cfg, run_dir=args.run_dir, checkpoint_path=ckpt_path)


if __name__ == "__main__":
    main()
