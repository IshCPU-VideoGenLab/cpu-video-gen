<p align="center">
  <img src="https://raw.githubusercontent.com/IshCPU-VideoGenLab/.github/main/logo.svg" alt="IshCPU-VideoGenLab" width="80">
</p>

# cpu-video-gen

**CPU-Native Video Generation: A Codec-Inspired SSM Architecture for Commodity Hardware**

The first video generation model that trains and runs entirely on commodity CPUs — no CUDA, no cloud, no GPU.

---

## Paper

```
CPU-Native Video Generation: A Codec-Inspired
SSM Architecture for Commodity Hardware

              Ishmael Affum Kwakye
         University of Ghana, Legon
```

**[Arxiv Preprint](#)** · **[GitHub Org](https://github.com/IshCPU-VideoGenLab)** · **[Demo](#)**

---

## What Is This?

A complete video generation pipeline designed from the ground up for CPU execution:

| Component | Innovation | Speedup |
|-----------|-----------|---------|
| **Mamba SSM** | Replaces O(n²) attention with O(n) scan | ~2× |
| **Codec Temporal** | I-frame keyframes + P-frame deltas | ~4-6× |
| **1-Bit Quantization** | XNOR + popcount replaces float matmul | ~8-16× |
| **Portable SIMD** | Native execution on x86 (AVX2) + ARM (NEON) | Hardware-native |
| **Distributed ES** | Training across laptops via WiFi | Scales with workers |

**Combined theoretical speedup: 64-192×** over naive GPU-architecture-on-CPU.

---

## The Full Pipeline

```
Text Prompt
    │
    ▼
┌─────────────────────────────────────────────┐
│  GOP Scheduler (I-frame / P-frame plan)     │
└─────────────────┬───────────────────────────┘
                  │
    ┌─────────────┴─────────────┐
    │                           │
    ▼                           ▼
┌──────────┐            ┌──────────────┐
│ I-Frame  │            │   P-Frame    │
│ Generator│            │  Delta       │
│ (Full    │            │  Predictor   │
│  Mamba   │            │  (BitNet     │
│  1-bit)  │            │   Conv)      │
└────┬─────┘            └──────┬───────┘
     │                         │
     └─────────┬───────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  SIMD Execution Engine                      │
│  (XNOR+popcount GEMM, vectorized SSM scan) │
└─────────────────┬───────────────────────────┘
               │
               ▼
         Video Frames
```

---

## Quick Start

```bash
git clone https://github.com/IshCPU-VideoGenLab/cpu-video-gen.git
cd cpu-video-gen

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt && pip install -e .

# Generate a video
python scripts/generate_video.py --prompt "a cat walking" --frames 16

# Run all paper benchmarks
python scripts/run_benchmarks.py --output results/

# Reproduce paper results
python scripts/reproduce_paper.py
```

---

## Project Architecture

This repo integrates all previous phases:

| Phase | Repo | Role |
|-------|------|------|
| 1 | [wan-profiler](https://github.com/IshCPU-VideoGenLab/wan-profiler) | Profiling data |
| 2 | [mamba-video](https://github.com/IshCPU-VideoGenLab/mamba-video) | SSM backbone |
| 3 | [codec-video-gen](https://github.com/IshCPU-VideoGenLab/codec-video-gen) | Temporal design |
| 4 | [bitnet-video](https://github.com/IshCPU-VideoGenLab/bitnet-video) | 1-bit quantization |
| 5 | [simd-kernels](https://github.com/IshCPU-VideoGenLab/simd-kernels) | Portable SIMD kernels (AVX2 + NEON) |
| 6 | [cpu-distributed](https://github.com/IshCPU-VideoGenLab/cpu-distributed) | Distributed training |
| **7** | **cpu-video-gen** (this repo) | **Full integration** |

---

## Reproducing Paper Results

Every number in the paper comes from a script:

```bash
# Table 1: Compute profiling breakdown
python scripts/reproduce_paper.py --table 1

# Figure 2: Quality vs blocks replaced (Mamba surgery)
python scripts/reproduce_paper.py --figure 2

# Figure 3: Quality vs GOP size (codec design)
python scripts/reproduce_paper.py --figure 3

# Table 2: Full pipeline benchmark
python scripts/reproduce_paper.py --table 2
```

---

## Hardware Tested

CPU-native, no GPU, across both architectures (x86 and ARM):

| Machine | CPU | Arch | Cores | RAM | Role |
|---------|-----|------|-------|-----|------|
| **MacBook Air M4** | Apple M4 | ARM64 / NEON | 10 | 16–24 GB | **Primary — development + benchmark** |
| CI runner | Intel / AMD | x86-64 / AVX2 | — | — | Verified every push ([simd-kernels CI](https://github.com/IshCPU-VideoGenLab/simd-kernels/actions)) |
| Core i5-1135G7 | Intel | x86-64 / AVX2 | 4C/8T | 16 GB | x86 comparison |
| Ryzen 5 5500U | AMD | x86-64 / AVX2 | 6C/12T | 16 GB | x86 comparison |
| Pentium Gold 7505 | Intel | x86-64 / AVX2 | 2C/4T | 16 GB | Original proof-of-concept (retired) |

---

## Citation

```bibtex
@article{kwakye2026cpuvideogen,
  title={CPU-Native Video Generation: A Codec-Inspired SSM Architecture for Commodity Hardware},
  author={Kwakye, Ishmael Affum},
  year={2026},
  journal={arXiv preprint},
  institution={University of Ghana, Legon},
  url={https://github.com/IshCPU-VideoGenLab/cpu-video-gen}
}
```

---

## Contributing

See the [Contributing Guide](https://github.com/IshCPU-VideoGenLab/.github/blob/main/CONTRIBUTING.md)
and [Version Control Guide](https://github.com/IshCPU-VideoGenLab/.github/blob/main/VERSION_CONTROL_GUIDE.md).

---

## License

MIT License.

---

*A Ghanaian student levels the playing field — from a Pentium Gold laptop.*
