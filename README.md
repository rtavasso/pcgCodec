**PCG-Codec: Parallel-Conditional Graph Codec**
(“parallel conditional” = bounded-depth intra-frame dependencies)

### Core thesis (reviewer-proof)

> **Given a fixed streaming latency budget, the best discrete audio representation is the one that minimizes expected perceptual distortion at a given expected codelength, under a constraint on intra-frame decoding depth.**
> RVQ implicitly chooses a depth equal to the number of stages. FSQ chooses depth 0 but sacrifices dependence modeling. We optimize the RD curve *subject to depth ≤ D*.

That’s genuinely new as a codec-first framing.

---

## 1) Latent manifold structuring (math-forward but practical)

### 1.1 Two-geometry decomposition: distortion vs entropy

We explicitly separate objectives:

* **Distortion geometry**: how quantization error maps to perceptual loss.
* **Entropy geometry**: how predictable the symbols are given causal context.

Formally, your RD Lagrangian is:
[
\mathbb{E}[d(x,\hat x)] + \lambda,\mathbb{E}[-\log p_\psi(q|q_{<t})]
]

Any transform (T) changes both. So we introduce **two transforms**:

1. (T_D): distortion-aware shaping (bit allocation / companding)
2. (T_H): entropy-aware shaping (makes symbols predictable)

But we **don’t** implement two separate networks; we implement one transform with two *regularizers* that target each property in expectation.

### 1.2 Metric shaping without a full tensor

Instead of (G(y)) PSD matching, use **expected block sensitivity equalization**:

Let (z = T_\theta(y)). Partition into blocks (z^{(b)}).

Define sensitivity statistic:
[
s_b = \mathbb{E}\left[\left|\frac{\partial \ell_{\text{percep}}}{\partial z^{(b)}}\right|*2\right]
]
Then add:
[
\mathcal{L}*{\text{sens-eq}} = \mathrm{Var}\left({\log s_b}_b\right)
]

This is a *companding/bit-allocation principle* expressed in gradients, but it avoids Hessians and is stable. It’s also streaming-friendly (no expensive per-sample PSD computations).

**Mathematical justification:** Under small quantization noise, expected loss increase is proportional to local sensitivity; equalizing sensitivity across quantized coordinates is a first-order approximation to optimal bit allocation.

### 1.3 Distribution shaping for packing

Add a whitening/mixing constraint:

* Lightweight orthogonal mixing (Householder/1×1 invertible conv) to reduce inter-block correlation.
* Penalty on off-diagonal block covariance (in expectation).

This makes blockwise quantization closer to the high-rate assumptions behind efficient packing (as in classical quantization theory). ([arXiv][1])

---

## 2) Optimal packing: start with “rotation + block quantizer,” treat lattice as an ablation

### 2.1 Quantizer family (progressive complexity)

Define blocks of dimension (m) and (B) blocks.

Quantizer options (in increasing “shape gain”):

1. **FSQ-like** per-dimension levels (fastest, most stable)
2. **Blockwise codebook** (small (K_b), learned)
3. **Lattice-regularized blockwise codebook** (optional)

Your packing story becomes:

* Baseline: axis-aligned cells (FSQ)
* Improvement: learned mixing makes cells non-axis-aligned (effectively “rotated FSQ”)
* Further improvement: learned codebooks + mild regularization approximating good cell shapes

You don’t claim “lattice beats FSQ” as a theorem. You claim:

> “Given a fixed depth and entropy model, better cell shapes can improve distortion at fixed rate; we evaluate how much shape gain remains once the encoder is learned.”

That’s honest and mathematically grounded.

---

## 3) Blockwise conditional factorization under a depth constraint (the main contribution)

### 3.1 Definition

For each frame (t), you output block symbols (q_{t,1..B}).

You define a DAG (\mathcal{G}) over blocks with depth ≤ (D) (small, like 2–3), and you entropy-code as:
[
p(q_t|q_{<t}) = \prod_{b=1}^B p(q_{t,b}\mid q_{t,\mathrm{pa}(b)}, q_{<t})
]

### 3.2 The real optimization target

Minimize expected code length subject to decode depth:
[
\min_{\mathcal{G},,p_\psi};; \sum_{b=1}^B H(q_{t,b}\mid q_{t,\mathrm{pa}(b)}, q_{<t})
\quad \text{s.t.}\quad \mathrm{depth}(\mathcal{G})\le D
]

This is the principled version of your “depth knob.”

### 3.3 Implementation that actually preserves low latency

* Group blocks by DAG layers (\mathcal{L}_1,\dots,\mathcal{L}_D).
* Use **D entropy streams** (one per layer), so decoding is **D sequential steps**, not B.
* Within a layer, symbols are decoded independently conditioned on earlier layers + past frames.

This directly addresses the critique about sequential entropy decoding: yes, entropy decoding is sequential per stream, but depth is bounded by (D), not by RVQ stages.

---

## 4) Autoregressive prediction over tokens (compression + generation)

This is two closely related priors:

### 4.1 Entropy model prior (for compression)

A causal transformer (or SSM) outputs distributions for each layer:
[
p_\psi(q_{t,\mathcal{L}*\ell}\mid q*{<t}, q_{t,\mathcal{L}_{<\ell}})
]
where the network outputs all blocks in the layer in parallel.

### 4.2 Generation prior (for synthesis/LLMs)

Same factorization gives you **time-to-first-audio = D steps**, not “#RVQ codebooks” steps. This directly targets the “staged generation” pain that real-time dialogue stacks face. ([arXiv][2])

This also positions cleanly versus EnCodec’s token compression transformer and the broader move toward token-centric audio modeling. ([arXiv][1])

---

# How this advances beyond SOTA (concretely)

### Compared to RVQ codecs (EnCodec-class)

* Removes intra-frame staged refinement; replaces with **bounded-depth dependency**.
* RD win comes from (a) strong entropy model *and* (b) learned latent shaping for block quantization.
* Measurable systems win: time-to-first-audio ∝ (D), not ∝ (N_{\text{RVQ}}). (You will report this explicitly.)

EnCodec is the baseline you must beat on RD at matched actual bitrate and streaming constraints. ([arXiv][1])

### Compared to FSQ codecs (NeuCodec-class)

NeuCodec shows FSQ is a compelling alternative to RVQ and emphasizes robustness and simplicity. ([arXiv][3])
You’re not “replacing FSQ.” You’re **strictly generalizing it**:

* FSQ corresponds to (D=0), independent blocks.
* PCG-Codec allows (D>0) for rate gains via conditional entropy reduction, while keeping latency bounded and decoding parallel inside layers.

That’s a clean, publishable, “math-first” generalization.

---

# What you should *claim* (and what you shouldn’t)

### Don’t claim “RD-optimal” absolutely.

The critique is right that with perceptual/adversarial losses and learned nonstationary latents, you can’t claim global RD optimality.

### Do claim:

> “We optimize a rate–distortion objective under an explicit *intra-frame decoding depth constraint* and show improved RD–latency trade-offs versus RVQ and FSQ baselines.”

That’s strong and defensible.

---

# Minimal prototype spec (so this doesn’t die in complexity)

Start with the **publishable core** and make the rest optional:

1. **Quantizer**: FSQ or small block codebooks (no lattice constraints initially)
2. **Transform**: lightweight orthogonal mixing + sensitivity equalization regularizer (diagonal/blockwise)
3. **Entropy model**: depth-constrained DAG with D∈{0,1,2,3} and D-stream entropy coder
4. **Loss**: multi-res STFT + waveform loss (GAN only after baseline works), to avoid training instability initially
5. **Ablations**: show each component’s RD benefit at matched actual bitrate

EnCodec-style training stability issues are real; they explicitly add mechanisms like loss balancing and adversarial setups to stabilize high-fidelity training. ([arXiv][1])
So you keep your first milestone stable and then add GAN later.

---

# The new experiment suite that directly answers the critique

* **Quantizer family @ matched entropy**: FSQ vs block-codebook vs lattice-regularized (if you add it)
* **Depth sweep**: D=0,1,2,3; report RD + time-to-first-audio + RTF
* **Sensitivity equalization**: off/on; show it improves RD beyond what entropy model learns
* **Streaming**: strict causal and fixed lookahead; report algorithmic latency explicitly
* **Baselines**: EnCodec (RVQ) and at least one FSQ codec line (NeuCodec if you can reproduce) ([arXiv][1])

---

[1]: https://arxiv.org/abs/2210.13438?utm_source=chatgpt.com "High Fidelity Neural Audio Compression"
[2]: https://arxiv.org/html/2410.00037v2?utm_source=chatgpt.com "Moshi: a speech-text foundation model for real-time dialogue"
[3]: https://arxiv.org/abs/2509.09550?utm_source=chatgpt.com "Finite Scalar Quantization Enables Redundant and Transmission-Robust Neural Audio Compression at Low Bit-rates"


---

# PCG-Codec Experimentation Spec

## 0) Purpose and non-negotiables

**Goal:** Evaluate a new discrete audio representation (“PCG-Codec”) designed to reduce RVQ-like intra-frame hierarchy while improving RD/latency tradeoffs under strict streaming constraints.

**Non-negotiable rigor requirements:**

1. **Bitrate accounting:** report *actual* bitrate after entropy coding.
2. **Causality/streaming:** fixed hop + fixed lookahead; stated and enforced.
3. **Depth definition:** unambiguous mapping from “depth steps” to decode operations.
4. **Baselines:** EnCodec causal 24 kHz model + its optional token compression LM (up to ~40% additional compression). ([GitHub][1])
5. **Ablations:** (quantizer) × (D) × (sens-eq on/off) at matched compute.
6. **Latency metrics:** time-to-first-audio + CPU RTF.

Deliverables: RD curves and latency curves; reproducible scripts; logged artifacts; a results table.

---

## 1) Definitions and success criteria

### 1.1 Streaming model constraints (must be enforced)

* Sample rate: **24,000 Hz**, mono.
* Frame hop: **H samples** (default H=320 → 13.33 ms).
* Encoder lookahead: **LA samples** (default LA=0 unless you intentionally allow lookahead).
* Decoder lookahead: **0** (decoder is strictly causal).

**Success criteria:**

* Any model configured with LA>0 must report LA in results and cannot be compared to LA=0 models unless clearly separated.
* A test asserts that, for a streaming input, outputs up to time (t) do not depend on inputs after (t + LA).

### 1.2 Depth definition (must be unambiguous)

PCG-Codec defines **intra-frame decoding depth D** as:

> The number of sequential entropy-decoding “layers” required to reconstruct all symbols for one frame, where symbols within each layer may be decoded in parallel (conceptually), but layers are decoded in order.

Concrete operational definition:

* Blocks are partitioned into DAG layers (\mathcal{L}_1,\dots,\mathcal{L}_D).
* Decoding frame (t) requires D sequential calls to `entropy_decode_layer(ℓ)` (or D sequential rANS stream decodes), then one call to `decoder_step()`.

**Time-to-first-audio (TTFA)** is measured as wall-clock time from (i) receipt of the first frame’s compressed bytes to (ii) first non-empty decoded PCM samples emitted.

**Success criteria:**

* TTFA scales with D, not with number of blocks B.
* Unit tests verify that the decoder cannot produce audio until required layer bytes are decoded (i.e., depth is real, not just conceptual).

### 1.3 “Actual bitrate” definition (must be post-entropy coding)

For a stream of N frames, each with a bytestream (s_t):

[
\text{bitrate} = \frac{8 \sum_{t=1}^N |s_t|}{N \cdot H / f_s} \quad \text{bits/sec}
]

**Success criteria:**

* Report mean bitrate across a dataset, plus distribution (p5/p50/p95).
* Also report bits/frame and bits/symbol (optional but recommended).
* Never use “raw token bits” for headline numbers.

---

## 2) Repository layout (required)

```
pcg_codec/
  configs/
    datasets.yaml
    encodec_baseline.yaml
    pcg_base.yaml
    ablations/
  data/
    (empty; use scripts to download)
  pcg/
    model/
      encoder.py
      decoder.py
      quantizers.py
      transforms.py
    entropy/
      dag.py
      prior_model.py
      coder_rans.py
      streams.py
    training/
      losses.py
      train.py
      eval.py
      metrics.py
      schedulers.py
  baselines/
    encodec/
      run_encodec.py
      run_encodec_lm.py
      wrappers.py
  experiments/
    run_ablation_grid.py
    summarize_results.py
    plot_rd_curves.py
    plot_latency.py
  tests/
    test_causality.py
    test_bitrate_accounting.py
    test_depth_semantics.py
    test_entropy_roundtrip.py
  results/
    (generated)
```

**Success criteria:** `pytest` passes on a clean machine after installing deps; a single command runs the full ablation grid with deterministic seeds.

---

## 3) Baseline implementation: EnCodec (required)

### 3.1 Baseline targets

* Baseline codec: **EnCodec 24 kHz causal mono model**.
* Bitrates: 1.5, 3, 6, 12, 24 kbps (as supported by EnCodec 24k model). ([GitHub][1])
* Token compression LM: run the provided pretrained LM to compress tokens further **up to ~40%** (reported claim). ([GitHub][1])

### 3.2 Implementation requirements

* Use official EnCodec repo weights and code paths where possible. ([GitHub][1])
* Two modes:

  1. `encodec_raw`: output EnCodec code indices per frame, then pack indices into bytes (raw).
  2. `encodec_lm`: run EnCodec token compression LM prior and entropy-code the token stream.

**Success criteria:**

* Exact reconstruction matches reference EnCodec output for fixed seed and input audio (within tolerance).
* Bitrate reported is based on actual bytestream size for both modes.
* Streaming is enforced: run EnCodec in streaming mode as provided (or with equivalent chunking).

### 3.3 Output format standardization

Define a shared container format:

* Header: sample_rate, hop, lookahead, codec_name, params hash
* For each frame: frame index, layer count, layer byte lengths, layer bytes

EnCodec is adapted to fit this container (it may not have layers; set D=1).

---

## 4) PCG-Codec: model specification

PCG-Codec = **(Encoder → Transform → Quantizer → DAG entropy model → Entropy coder) + Decoder**.

### 4.1 Encoder/decoder streaming interface

Define interfaces:

```python
class StreamingEncoder:
    def reset(self): ...
    def encode_frame(self, x_frame: np.ndarray) -> ContinuousLatent: ...

class StreamingDecoder:
    def reset(self): ...
    def decode_frame(self, q_frame: DiscreteFrame) -> np.ndarray: ...
```

* `x_frame` is exactly H samples (plus internally buffered LA if lookahead allowed).
* The decoder produces exactly H samples per frame.

**Success criteria:**

* Encoder and decoder can run incrementally on arbitrarily long streams with constant memory (bounded state).
* No dependence on future frames beyond allowed lookahead.

### 4.2 Transform module (latent shaping)

Two transform variants must be implemented:

1. **Mixing transform**: lightweight invertible/orthogonal mixing (e.g., 1×1 conv + orthogonal regularizer).
2. **Identity**: for ablation.

Outputs (z_t \in \mathbb{R}^d) and supports inverse mapping if needed by decoder.

**Success criteria:**

* Transform is causal (frame-local).
* Regularizer logs off-diagonal covariance metrics for monitoring.

### 4.3 Quantizer module (ablation factor “quantizer”)

Implement at minimum:

* **FSQ** (finite scalar quantization) style: per-dimension discrete levels.
* **Block codebook**: partition into B blocks of dim m; each block has K entries.

(You may optionally add “lattice-regularized” as a third quantizer after you have stable runs, but it is not required for the initial grid.)

**Quantizer outputs:**

* Discrete symbols (q_{t,1..B}), each in ([0, K_b-1])
* Reconstructed latent (\hat z_t)

**Success criteria:**

* Entropy-coded bitrate is computed from actual encoded bytes, not raw symbol counts.
* Quantizer round-trip is deterministic given seed and input.
* Code utilization stats per block are logged (avoid collapse).

### 4.4 Sensitivity equalization (ablation factor “sens-eq on/off”)

Implement a stable first-order approximation:

* Compute per-block gradient norm of perceptual loss w.r.t. latent (z^{(b)}):
  [
  s_b = \mathbb{E}\left[\left|\frac{\partial \ell_{\text{percep}}}{\partial z^{(b)}}\right|_2\right]
  ]
* Add regularizer:
  [
  \mathcal{L}_{\text{sens-eq}} = \mathrm{Var}\big({\log(s_b+\epsilon)}_b\big)
  ]

Where (\epsilon) prevents log(0).

**Success criteria:**

* The regularizer decreases the variance of sensitivities without destabilizing training (tracked over epochs).
* When “off,” the codepath is identical except for the missing loss term.

---

## 5) DAG dependency and entropy coding (ablation factor “D”)

### 5.1 DAG layers definition

Given B blocks and desired depth D:

* Partition blocks into layers (\mathcal{L}_1,\dots,\mathcal{L}_D)
* Define parent sets (\mathrm{pa}(b)\subset \cup_{\ell'<\ell(b)}\mathcal{L}_{\ell'})

Two required DAG families:

1. **Independent (D=1)**: (\mathrm{pa}(b)=\emptyset) for all b.
2. **Layered bipartite (D>1)**: each block depends on a small fixed number of parents from earlier layers (e.g., k=2).

**Success criteria:**

* Graph construction is deterministic given (B,D,k,seed).
* Depth is exactly D (verified by topological depth computation).

### 5.2 Prior model for entropy coding

Implement a causal prior (p_\psi) that outputs per-block categorical distributions:

[
p(q_{t,b} \mid q_{<t}, q_{t,\mathrm{pa}(b)})
]

Requirements:

* **Causal over time:** cannot use future frames.
* **Layer-aware:** can condition on previously decoded layers for the current frame.

Suggested architectures:

* Small causal Transformer over time on a compact state + per-layer conditioning embeddings.
* Or state-space model + per-layer MLP heads.

**Success criteria:**

* Cross-entropy on held-out set decreases over training.
* When D increases, expected codelength decreases (or stays equal) at fixed reconstruction model, otherwise the graph is not doing useful work.

### 5.3 Entropy coder implementation

Implement rANS (recommended) with **one stream per layer**:

* For each frame t:

  * For ℓ = 1..D:

    * Encode symbols in layer (\mathcal{L}_\ell) using model probabilities conditioned on prior decoded layers + past frames.
    * Append bytes for that layer.

Decoding mirrors this exactly.

**Success criteria:**

* `test_entropy_roundtrip.py` confirms bit-exact encode/decode for random symbol streams and real model outputs.
* Measured bitrate uses the actual emitted bytes.

---

## 6) Training specification

### 6.1 Losses

Required reconstruction/perceptual losses (start stable, add GAN later):

* waveform L1 or L2
* multi-resolution STFT magnitude loss

Optional later:

* adversarial discriminator (only after baseline stable)

**Success criteria:**

* Training runs do not diverge for ≥ N steps (define N=200k or your budget).
* Loss curves are logged; reconstructions are listenable.

### 6.2 RD training objective

Use a Lagrangian:

[
\mathcal{L} = \mathbb{E}[d(x,\hat x)] + \lambda ,\mathbb{E}\big[-\log p_\psi(q)\big] + \beta \mathcal{L}_{\text{sens-eq}}
]

Where:

* (-\log p_\psi(q)) is computed from the entropy model used by the coder (teacher-forced during training).
* For fair bitrate accounting, also periodically run full entropy coding on validation and log actual bitrate.

**Success criteria:**

* RD curves can be traced by sweeping (\lambda).
* Actual bitrate correlates with modeled cross-entropy (sanity check).

### 6.3 Compute-matched ablation enforcement

To match compute across ablations:

* Fix encoder/decoder parameter budget across quantizers.
* Fix entropy model capacity across D conditions (or adjust to equal FLOPs).
* Measure and log:

  * training FLOPs proxy (tokens processed × model size)
  * inference FLOPs proxy or wall-clock throughput

**Success criteria:**

* Ablation comparisons include compute logs; plots are annotated “compute-matched” or explicitly not.

### 6.4 Throughput benchmarking (training-time bottlenecks)

This repo includes a lightweight benchmark runner to separate input/transfer/step time:

* `python -m pcg_codec.pcg.training.benchmark --config pcg_codec/configs/pcg_hf_dummy.yaml --datasets pcg_codec/configs/datasets.yaml --iters 500 --warmup 50 --sync-cuda`
* Use `--synthetic` to measure compute-only (bypasses dataset IO).

Training configs can also set:

* `training.dataloader.*` (e.g., `num_workers`, `pin_memory`, `persistent_workers`, `prefetch_factor`)
* `training.amp.*` (e.g., `enabled`, `dtype`)
* `training.timing.enabled: true` to emit `data_s`, `transfer_s`, `step_s` into `train_log.jsonl` (logging steps only).

---

## 7) Evaluation specification (must produce publishable artifacts)

### 7.1 Datasets

Specify at least:

* Speech dataset (e.g., LibriSpeech subset)
* Music dataset (e.g., a public music dataset)
* General audio (optional)

Each dataset config includes:

* sample rate conversion
* segment length
* train/val/test split hash

**Success criteria:** dataset download + preprocessing is scripted and reproducible.

### 7.2 Metrics

For each model and bitrate point report:

**Rate:**

* mean bitrate (bps), p5/p50/p95
* bits/frame

**Distortion/quality:**

* multi-res STFT loss on test
* at least one established quality metric per domain (choose appropriate for speech/music)
* optional: FAD for music; objective speech metrics for speech

**Latency:**

* time-to-first-audio (ms)
* RTF on CPU (e.g., 1 thread and 4 threads)
* peak memory

**Success criteria:** evaluation script runs end-to-end and outputs a single JSON + plots.

### 7.3 Streaming + TTFA benchmark harness

Implement a real-time simulation:

* Input stream: chunked into H-sample frames.
* Decoder fed encoded bytes as they would arrive.
* TTFA measured precisely:

  * start timer at receipt of first bytes for frame 0
  * stop timer when first PCM samples emitted

**Success criteria:** TTFA is reproducible across runs (low variance); CPU affinity pinned for benchmarks.

---

## 8) Ablation grid (the required experiment matrix)

### 8.1 Factors

* Quantizer: {FSQ, BlockCodebook}
* Depth D: {1,2,3} (D=1 is independent / no intra-frame conditioning)
* Sens-eq: {off, on}

Total: 2 × 3 × 2 = **12 models** per (\lambda) point.

Also run baseline:

* EnCodec raw at matched nominal kbps settings (1.5/3/6/12/24)
* EnCodec + LM token compression (where applicable) ([GitHub][1])

### 8.2 Matching bitrates fairly

Because your models won’t naturally land on EnCodec’s exact kbps points:

* For each target bitrate (e.g., 1.5, 3, 6, 12 kbps), tune (\lambda) until actual bitrate matches within ±2%.
* Log the achieved bitrate and the (\lambda) used.

**Success criteria:** for every RD point, bitrate matches target within tolerance; otherwise it is excluded from “matched” plots.

---

## 9) Plots and reporting (must be standardized)

Produce:

1. **RD curves**: quality metric vs actual bitrate (with confidence intervals if possible)
2. **Latency curves**: TTFA vs depth D at matched bitrate
3. **RTF table**: per model, per bitrate
4. **Ablation deltas**: Δquality at fixed bitrate and Δbitrate at fixed quality

**Success criteria:** all plots auto-generated from evaluation JSONs; no manual editing required.

---

## 10) Implementation milestones (so engineering knows what “done” means)

### Milestone A: Baseline plumbing (1–2 weeks)

* EnCodec raw runs in streaming mode and produces valid bytestream container. ([GitHub][1])
* Bitrate accounting and TTFA harness passes tests.

**Exit criteria:** baseline RD/latency results reproducible.

### Milestone B: PCG-Codec minimal (D=1, FSQ, no sens-eq)

* PCG encoder/decoder round-trip works in streaming.
* Entropy model optional; coder can pack symbols losslessly.

**Exit criteria:** outputs are listenable; actual bitrate logged.

### Milestone C: DAG depth + entropy coding (D=2/3)

* Layered entropy coder with D streams works and reduces bitrate vs D=1 at comparable distortion.

**Exit criteria:** depth sweep shows monotonic or near-monotonic bitrate reductions.

### Milestone D: sens-eq ablation

* sens-eq regularizer integrated and shown to help at least one operating point.

**Exit criteria:** sens-eq effect quantified and stable.

### Milestone E: Full ablation grid + EnCodec comparisons

* 12-model grid + EnCodec and EnCodec+LM plotted, all at matched actual bitrate.

**Exit criteria:** final report artifact generated.

---

## 11) Testing suite (must exist before “results” are trusted)

1. `test_bitrate_accounting.py`

* Feeds known byte lengths; verifies bitrate formula and units.

2. `test_causality.py`

* Corrupt future input samples and verify earlier outputs unchanged (within numeric tolerance).

3. `test_depth_semantics.py`

* Verifies a D-layer stream requires exactly D sequential decode steps to output the first frame.

4. `test_entropy_roundtrip.py`

* Random symbols → encode → decode → identical symbols.

**Success criteria:** CI runs tests on every commit; benchmark scripts refuse to run if tests fail.

---

## 12) Notes on engineering choices

* Use EnCodec’s supported bitrates and 24 kHz causal mono model as baseline anchor. ([GitHub][1])
* Expect EnCodec token compression LM to yield up to ~40% further compression; you must report *actual* bitrate from bytes emitted, not cite the % claim. ([GitHub][1])

---

[1]: https://github.com/facebookresearch/encodec?utm_source=chatgpt.com "facebookresearch/encodec: State-of-the-art deep learning ..."
