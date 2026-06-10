# Perceptual Validation — do the cosine numbers hold up in pixel space?

The Phase 2/4 quality numbers are **latent cosine similarity**. This decodes the
DiT's output latent through the Wan VAE and measures **pixel-space PSNR** of each
optimization vs. the original model (`scripts/perceptual_validation.py`).

## Results

| optimization | latent cosine (4-step) | pixel PSNR vs. original |
|--------------|-----------------------:|------------------------:|
| Self-attention → Mamba (untrained) | 0.995 | **25.5 dB** |
| FFN → 1-bit (post-training) | 0.947 | **21.4 dB** |

**Reading:** the optimizations **preserve the model's output structure in pixel
space** — the decoded frames are visually near-identical to the original,
corroborating the latent-cosine numbers with a real perceptual metric. But
0.995 cosine corresponds to ~25.5 dB PSNR: *faithful, with minor visible
differences*, not pixel-perfect. The drift is real but recoverable (fine-tuning).

## Honest scope (read this)

This measures **fidelity (modified vs. original)**, **not absolute video quality**:

- The text embedding is a **dummy tensor** (the ~11 GB T5 is not loaded), so the
  generated content is **abstract**, not a real prompted scene.
- The denoise is a short (4-step) Euler loop, not the proper Wan scheduler.

So "does optimization X change the output?" is answered (a little, gracefully).
"Is the output good video?" needs: a real (precomputed) text embedding, the
proper multi-step scheduler, and perceptual metrics (FID / SSIM / LPIPS) against
real references. That is future work and would benefit from precomputing a few
real text embeddings once on a larger machine.

## Artifacts

PNGs are written to `results/perceptual/` (`original.png`,
`self_attn_mamba.png`, `ffn_1bit.png`). They are abstract textures (dummy text),
useful only for relative comparison.
