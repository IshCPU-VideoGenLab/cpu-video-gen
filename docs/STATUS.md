# Project Status & How to Continue

Snapshot of where IshCPU-VideoGenLab actually stands, written so you can pick the
work back up cold. Everything below was measured on the **real Wan 1.3B DiT** on
an Apple M4 (CPU-only, bfloat16) unless noted.

## One-line state

The **methodology is fully validated and the paper is placeholder-free**. What
remains (working trained model, real-prompt FID) is gated on two external steps:
a precomputed real-text embedding, and fine-tuning.

## Environment / how to run

- Python venv with the model stack lives at `simd-kernels/.venv312`
  (`source .venv312/bin/activate`). It has torch, diffusers, numpy, psutil, and
  the editable `simd_kernels` + `wan_profiler` packages.
- The model is loaded **DiT-only** via diffusers from
  `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, `subfolder="transformer"` — **never** the
  11–22 GB T5 text encoder. Cached locally: the **DiT (5.68 GB)** and **VAE
  (0.5 GB)** are already downloaded.
- Load in **bfloat16** (float16 CPU kernels are largely unimplemented).
- A HuggingFace token is needed for downloads. **SECURITY: rotate the tokens that
  were shared during development** at https://huggingface.co/settings/tokens.
- One DiT forward (256 tokens) ≈ 5.4 s; peak RAM ≈ 6.3 GB. Stay ≤ ~768 tokens on
  16 GB (1024 OOMs).

## What's done (measured, in repos + paper)

| Pillar | Result |
|--------|--------|
| **Phase 1 — profiling** | Table 1: attention 60.1%, FFN 38.5%; self-attn O(n²) (`wan-profiler`) |
| **Phase 2 — Mamba surgery** | Replace **self-attention** → Mamba: 0.984–0.995 cosine, 25.5 dB; O(n) confirmed, SIMD scan integrated. **Keep cross-attention** (`mamba-video`) |
| **Phase 4 — 1-bit quant** | FFN 1-bit: 0.947 / 21.4 dB; all-linear 0.847; **16× memory** (2.84 GB → 178 MB) (`bitnet-video`) |
| **Phase 5 — SIMD kernels** | AVX2+NEON, multithreaded binary GEMM ~2× vs Accelerate, bit-exact scan, CI green (`simd-kernels`) |
| **Perceptual** | optimizations preserve output in pixel space (`cpu-video-gen/scripts/perceptual_validation.py`) |

## Key findings (including the honest negatives — these are contributions)

1. **Self-attention linearizes gracefully** even untrained; the residual stream +
   gating absorbs the surgery.
2. **Cross-attention must NOT be naively replaced** — it's only O(n·m) (linear in
   image tokens) and carries text conditioning; a pooled-text SSM collapses it
   (0.37). Replace self-attn, keep cross-attn.
3. **1-bit quantization is a recoverable perturbation** (post-training); the real
   win is 16× memory.
4. **Codec temporal quality needs real generation** — with dummy text the output
   is temporally incoherent by construction, so its redundancy is uninformative.

## What's gated, and the exact next steps

**Gate 1 — real generation (unlocks FID, real video, codec quality).**
Precompute a few real text embeddings **once** on a larger machine / free Colab
(load the T5, encode prompts, save the ~4 MB tensors), then use them in place of
the dummy `torch.randn(1, 512, 4096)`. Never load T5 on the 16 GB CPU target.
Then: re-run `codec-video-gen/scripts/temporal_redundancy.py`, generate real
clips, compute FID/SSIM.

**Gate 2 — fine-tuning (Phase 6, unlocks a working model).**
The surgery/quantization drift is recoverable but needs training. This is the
distributed-ES work in `cpu-distributed`. It is the hard, unproven part.

## Reproduce the key experiments

```bash
source simd-kernels/.venv312/bin/activate
export HF_TOKEN=...   # a freshly rotated token
HF_TOKEN=$HF_TOKEN python wan-profiler/scripts/run_profile.py --model wan-1.3b --frames 4
HF_TOKEN=$HF_TOKEN python mamba-video/scripts/wan_surgery_multistep.py
HF_TOKEN=$HF_TOKEN python mamba-video/scripts/wan_surgery_speed.py
HF_TOKEN=$HF_TOKEN python bitnet-video/scripts/quant_drift_experiment.py
HF_TOKEN=$HF_TOKEN python cpu-video-gen/scripts/perceptual_validation.py
```

## Suggested order when you return

1. Rotate the HF tokens.
2. Precompute real text embeddings (Gate 1) — highest leverage, ~10 min elsewhere.
3. Real-prompt generation + codec temporal validation + FID.
4. Then Phase 6 fine-tuning (Gate 2).
