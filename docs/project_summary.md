# IshCPU-VideoGenLab — Complete Project Summary

## Author
**Ishmael Affum Kwakye** (Calyx)
University of Ghana, Legon | GitHub: calyxish
Org: github.com/IshCPU-VideoGenLab

---

## The Core Idea

Build the first video generation model that trains and runs entirely on commodity CPUs — no CUDA, no GPU, no cloud. Four technologies, never combined before, make this possible: 1-bit quantization, Mamba SSM architecture, codec-inspired temporal compression, and AVX2 SIMD kernels.

---

## What Was Built In This Conversation

### 7 Complete Repositories (234 files total)

| # | Repo | Files | Purpose |
|---|------|-------|---------|
| 1 | `wan-profiler` | 28 | Profile where Wan 1.3B spends compute |
| 2 | `mamba-video` | 31 | Replace O(n²) attention with O(n) Mamba SSM |
| 3 | `codec-video-gen` | 34 | I-frame/P-frame temporal design (like H.264) |
| 4 | `bitnet-video` | 33 | 1-bit weight quantization (XNOR + popcount) |
| 5 | `simd-kernels` | 39 | Portable C kernels (AVX2 + NEON intrinsics) |
| 6 | `cpu-distributed` | 34 | Distributed training via evolution strategies |
| 7 | `cpu-video-gen` | 35 | Flagship paper repo integrating all phases |

Every repo includes:
- `CLAUDE.md` — full project context for Claude Code sessions
- `tasks/todo.md` — milestone roadmap with checkboxes
- `lessons.md` — grows as mistakes are discovered
- `.claude/` — settings, slash commands (`/project:review`, `/project:progress`), Python rules
- Full source code with type hints and docstrings
- Unit tests (pytest)
- CLI entry points
- Documentation
- README with citation block

### 1 Research Paper (11 pages, LaTeX + PDF)

**"CPU-Native Video Generation: A Codec-Inspired SSM Architecture for Commodity Hardware"**

NeurIPS/ICML format. Sections fully written: Abstract, Introduction, Related Work (13 citations), full Method section with equations, Experimental Setup, Discussion, Conclusion, Appendix. Results sections have `\todo{}` placeholders for real experimental data.

---

## How Each Phase Works

### Phase 1: wan-profiler
Instruments Wan 1.3B with PyTorch forward hooks. Measures wall-clock time, FLOPs, and memory per module. Identifies that self-attention dominates compute, motivating all subsequent phases.

### Phase 2: mamba-video
Replaces transformer attention blocks with Mamba Selective State Space Models. Pure PyTorch implementation (no CUDA). Sequential scan: h[t] = A·h[t-1] + B·x[t]. Four surgery strategies: all, progressive, by-cost, alternating. Measures quality degradation per block replaced.

### Phase 3: codec-video-gen (most novel)
Borrows from H.264 video codecs. Instead of generating every frame through the full model, generates keyframes (I-frames) at full quality and predicts deltas (P-frames) with a tiny network (50-100M params). GOP structure: I P P P P P P P I P P P... Theoretical speedup: 4.7× at GOP size 8.

### Phase 4: bitnet-video
Constrains weights to {-1, +1}. Matrix multiply becomes XNOR + popcount. BitLinear and BitConv2d are drop-in replacements for nn.Linear and nn.Conv2d. Straight-Through Estimator for gradient flow. Memory reduction: 16× (2.6 GB → 160 MB).

### Phase 5: simd-kernels
Portable C kernels with a unified API over three backends — AVX2 (x86), NEON (ARM), and a scalar fallback — selected at compile time. Binary GEMM processes 256 weight bits per XNOR on AVX2 (128 on NEON). SSM scan vectorized with FMA (`_mm256_fmadd_ps` / `vfmaq_f32`). Popcount via nibble lookup table. Python bindings via ctypes with PyTorch fallback.

### Phase 6: cpu-distributed
Evolution strategies optimizer. Workers perturb parameters, run forward pass, return (seed, fitness_scalar). Communication: ~100 bytes per worker per step. Antithetic sampling halves variance. Adam momentum. Works over WiFi between laptops.

### Phase 7: cpu-video-gen
Integrates all phases into one pipeline. Builds model (Mamba surgery → BitNet quantization), generates video (GOP scheduler → I-frame full model → P-frame delta predictor → SIMD kernels). Includes ablation study framework and `reproduce_paper.py`.

---

## Speedup Composition

| Optimization | Mechanism | Individual Speedup |
|-------------|-----------|-------------------|
| Mamba SSM | O(n²) → O(n) attention removal | ~2× |
| Codec temporal | I/P frame scheduling | ~4-6× |
| BitNet 1-bit | Binary operations replace float | ~8-16× |
| SIMD kernels | Native execution (AVX2 + NEON) | ~2× |
| **Combined** | **Multiplicative** | **~64-192×** |

---

## Hardware Context

### Original Target
- Intel Pentium Gold 7505 (2 cores/4 threads, 3.5 GHz)
- 16 GB DDR4 (single channel)
- No GPU
- The "if it runs here, it runs everywhere" thesis

### Current Development Machine
- MacBook Air M4
- ARM64 architecture — natively supported via the portable simd-kernels NEON backend
- Much faster for development iteration

### Honest Assessment (discussed in conversation)
- The Pentium Gold will struggle with Wan 1.3B (possible OOM at 16 GB)
- Forward passes could take minutes, making iteration painful
- Recommended approach: develop on M4, benchmark on Pentium Gold
- Paper framing may need adjustment from "trains entirely on Pentium Gold" to "inference runs on commodity CPUs"

---

## Paper Status

### Written (ready):
- Abstract, Introduction, Related Work
- Full Method section (5 subsections with equations)
- Experimental Setup (hardware, metrics, ablation configs)
- Discussion (speedup composition, tradeoffs, limitations, broader impact)
- Conclusion, Appendix, 13 references

### Needs real data (marked with \todo{}):
- Table 1: Profiling breakdown (run wan-profiler)
- Figure 2: Quality vs Mamba blocks replaced
- Figure 3: Temporal redundancy + codec quality curves
- Table 2: Full ablation study
- Table 3: AVX2 kernel throughput
- Training curves from distributed ES

---

## Timeline (from original plan)

| Period | Milestone |
|--------|-----------|
| April 2026 | Exams |
| May 2026 | Phase 1 begins |
| June-August 2026 | Phases 2-3 (core research) |
| September-November 2026 | Phases 4-5 (engineering) |
| December 2026-January 2027 | Phases 6-7 (paper) |
| Early 2027 | Arxiv submission |
| Fall 2027 | PhD application deadline |

---

## All Deliverables Produced

| File | Type |
|------|------|
| `IshCPU-VideoGenLab_Project_Summary.md` | Project overview document |
| `wan-profiler.zip` | Phase 1 complete repo |
| `mamba-video.zip` | Phase 2 complete repo |
| `codec-video-gen.zip` | Phase 3 complete repo |
| `bitnet-video.zip` | Phase 4 complete repo |
| `simd-kernels.zip` | Phase 5 complete repo |
| `cpu-distributed.zip` | Phase 6 complete repo |
| `cpu-video-gen.zip` | Phase 7 complete repo |
| `CPU_Native_Video_Generation_Paper.pdf` | 11-page research paper |
| `paper_latex_source.zip` | LaTeX source files |
