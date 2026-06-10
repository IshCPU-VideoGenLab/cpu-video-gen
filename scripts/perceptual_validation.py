#!/usr/bin/env python
"""Perceptual validation: do the optimizations' latent-cosine numbers hold up
in pixel space?

Decodes the DiT's output latent through the Wan VAE and compares the original
model against (a) self-attention Mamba surgery and (b) FFN 1-bit quantization,
reporting pixel-space PSNR and saving PNGs.

IMPORTANT — this measures *fidelity* (modified vs. original), not absolute video
quality: it uses a dummy text embedding (no T5) and a short Euler denoise, so the
decoded content is abstract. Absolute quality needs a real (precomputed) text
embedding, a proper scheduler, more steps, and FID/SSIM.

Requires the DiT + VAE cached, and simd-kernels installed (for the Mamba scan).

Usage:
    HF_TOKEN=... python scripts/perceptual_validation.py
"""
import math
import os
import sys
import types

import numpy as np
import psutil
import torch
import torch.nn as nn
from PIL import Image

torch.set_num_threads(psutil.cpu_count(logical=False) or 4)
torch.manual_seed(0)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "mamba-video", "src"))
from mamba_video.wan_adapter import WanMambaSelfAttention  # noqa: E402
from simd_kernels.ssm_scan import simd_ssm_scan  # noqa: E402
from diffusers import WanTransformer3DModel, AutoencoderKLWan  # noqa: E402

REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "perceptual")
STEPS = 4


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    vae = AutoencoderKLWan.from_pretrained(REPO, subfolder="vae", torch_dtype=torch.float32).eval()
    zmean = torch.tensor(vae.config.latents_mean).view(1, 16, 1, 1, 1)
    zstd = torch.tensor(vae.config.latents_std).view(1, 16, 1, 1, 1)
    txt = torch.randn(1, 512, 4096, dtype=torch.bfloat16)
    x_init = torch.randn(1, 16, 1, 32, 32, dtype=torch.bfloat16)

    def load_dit():
        return WanTransformer3DModel.from_pretrained(
            REPO, subfolder="transformer", torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True).eval()

    @torch.no_grad()
    def denoise(model):
        x = x_init.clone()
        for i in range(STEPS):
            t = torch.tensor([int((1 - i / STEPS) * 999)])
            x = x - (1 / STEPS) * model(x, t, txt, return_dict=False)[0]
        return x.float()

    @torch.no_grad()
    def decode(lat, name):
        img = vae.decode(lat * zstd + zmean, return_dict=False)[0]
        a = ((img[0, :, 0].permute(1, 2, 0).clamp(-1, 1) + 1) / 2 * 255).byte().numpy()
        Image.fromarray(a).save(os.path.join(OUT, f"{name}.png"))
        return a.astype(np.float32) / 255

    def psnr(a, b):
        mse = float(np.mean((a - b) ** 2))
        return 99.0 if mse < 1e-9 else 20 * math.log10(1.0 / math.sqrt(mse))

    m = load_dit()
    ref = decode(denoise(m), "original")
    dim = m.config.num_attention_heads * m.config.attention_head_dim
    for b in m.blocks:
        b.attn1 = WanMambaSelfAttention(dim)
        b.attn1.mamba._selective_scan = types.MethodType(
            lambda s, x, A, B, C: simd_ssm_scan(x[0], A[0], B[0], C[0]).unsqueeze(0).to(x.dtype),
            b.attn1.mamba)
    print(f"self-attn Mamba:  PSNR={psnr(ref, decode(denoise(m), 'self_attn_mamba')):.1f} dB")
    del m

    m = load_dit()
    for name, mod in m.named_modules():
        if isinstance(mod, nn.Linear) and ".ffn." in name:
            w = mod.weight.data
            sc = w.abs().float().mean(dim=1, keepdim=True).clamp(min=1e-8).to(w.dtype)
            mod.weight.data = w.sign() * sc
    print(f"FFN 1-bit:        PSNR={psnr(ref, decode(denoise(m), 'ffn_1bit')):.1f} dB")
    print(f"images in {OUT}/")


if __name__ == "__main__":
    main()
