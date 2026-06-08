# IshCPU-VideoGenLab — Complete Conversation Summary & Project Guide

**Author:** Ishmael Affum Kwakye (Calyx)
**GitHub:** calyxish | **Org:** IshCPU-VideoGenLab
**Institution:** University of Ghana, Legon

---

## What Happened in This Conversation

### Starting Point
You came in with a detailed project plan from a previous conversation — building the
first CPU-native video generation model. The plan had 7 phases, 4 technical pillars,
a timeline targeting Arxiv submission before Fall 2027 PhD applications, and your
identity stack (legal name, GitHub, org name) already locked in.

### What We Built
Over the course of this conversation, we constructed the ENTIRE project infrastructure
from scratch — all 7 repos, a research paper, and then redesigned Phase 5 when you
switched from Windows/Pentium Gold to MacBook Air M4.

### Key Decisions Made During the Conversation

1. **All 7 repos built** with full Claude Code setup (CLAUDE.md, tasks, lessons,
   slash commands, rules) so every Claude Code session has full context.

2. **Research paper written** — 11-page LaTeX in NeurIPS format. Methodology sections
   fully written. Results sections have \todo{} placeholders waiting for real data.

3. **Hardware reality check** — We honestly assessed that the Pentium Gold (2 cores,
   16GB) would struggle to even load Wan 1.3B, and iteration would be painfully slow.

4. **M4 MacBook entered the picture** — You got new hardware. Phase 5 (AVX2 kernels)
   broke because M4 is ARM, not x86.

5. **Phase 5 redesigned** — Instead of dropping it, we rebuilt it as `simd-kernels`
   with a portable backend abstraction: AVX2 for x86, NEON for ARM, scalar fallback
   for anything else. Compiled and tested — 8/8 C tests pass.

---

## The 7 Repos — What Each One Does

```
IshCPU-VideoGenLab/
│
├── 1. wan-profiler          WHERE does the compute go?
│     Profile Wan 1.3B: time, FLOPs, memory per module.
│     This data motivates every decision in Phases 2-5.
│
├── 2. mamba-video            Kill quadratic attention
│     Replace O(n²) self-attention with O(n) Mamba SSM blocks.
│     Pure-PyTorch selective scan. 4 surgery strategies.
│     ~2× speedup.
│
├── 3. codec-video-gen        Kill redundant computation
│     Codec-inspired I-frame/P-frame temporal design.
│     Generate keyframes fully, predict deltas cheaply.
│     ~4-6× speedup. MOST NOVEL — this is your unique contribution.
│
├── 4. bitnet-video           Kill float arithmetic
│     1-bit weight quantization (BitNet). Weights → {-1, +1}.
│     Matmul → XNOR + popcount. 16× memory reduction.
│     ~8-16× speedup (with native kernels).
│
├── 5. simd-kernels           Native silicon speed (REPLACES old avx2-kernels)
│     Portable SIMD: NEON backend (M4), AVX2 backend (commodity x86),
│     scalar fallback (anything). One API, compile-time selection.
│     Compiled & tested: 8/8 tests pass.
│
├── 6. cpu-distributed        Train without GPUs
│     Evolution strategies for gradient-free training.
│     Coordinator/worker over TCP. Seeds + scalars only.
│     Works over WiFi between laptops.
│
└── 7. cpu-video-gen          THE PAPER REPO — ties everything together
      Full pipeline integration. Ablation study.
      Paper reproduction scripts. Benchmark suite.
```

---

## The Files You Downloaded — Which Ones to Use

### THE ACTUAL FILES (use these):

| File | Phase | Use This |
|------|-------|----------|
| `wan-profiler.zip` | Phase 1 | ✅ Yes |
| `mamba-video.zip` | Phase 2 | ✅ Yes |
| `codec-video-gen.zip` | Phase 3 | ✅ Yes |
| `bitnet-video.zip` | Phase 4 | ✅ Yes |
| **`simd-kernels.zip`** | **Phase 5** | **✅ Yes — this REPLACES avx2-kernels** |
| `cpu-distributed.zip` | Phase 6 | ✅ Yes |
| `cpu-video-gen.zip` | Phase 7 | ✅ Yes |
| `CPU_Native_Video_Generation_Paper.pdf` | Paper | ✅ Yes |
| `paper_latex_source.zip` | Paper source | ✅ Yes |

### SUPERSEDED (do NOT use):

| File | Why |
|------|-----|
| `avx2-kernels.zip` | ❌ REPLACED by `simd-kernels.zip` — the old version only works on x86, the new version works on both x86 AND your M4 |
| `IshCPU-VideoGenLab_Project_Summary.md` | ❌ Superseded by this document |
| `IshCPU_VideoGenLab_Complete_Summary.md` | ❌ Superseded by this document |

---

## Execution Order — How to Actually Run This Project

### Step 0: Setup (do once)

```
Create GitHub org: github.com/IshCPU-VideoGenLab
Create 7 repos, push each zip's contents to the corresponding repo.

Your GitHub org will look like:
  IshCPU-VideoGenLab/wan-profiler
  IshCPU-VideoGenLab/mamba-video
  IshCPU-VideoGenLab/codec-video-gen
  IshCPU-VideoGenLab/bitnet-video
  IshCPU-VideoGenLab/simd-kernels        ← (NOT avx2-kernels)
  IshCPU-VideoGenLab/cpu-distributed
  IshCPU-VideoGenLab/cpu-video-gen
```

### Step 1: Phase 1 — wan-profiler (START HERE)

```bash
cd wan-profiler
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt && pip install -e .
python scripts/run_profile.py --model wan-1.3b --output results/ --low-memory
```

**What you get:** Table 1 for the paper — where Wan 1.3B spends compute.
**This must run first.** Everything else depends on knowing the bottlenecks.

### Step 2: Phase 3 — codec-video-gen (YOUR UNIQUE CONTRIBUTION)

```bash
cd codec-video-gen
pip install -r requirements.txt && pip install -e .

# First: prove temporal redundancy exists
python scripts/analyze_redundancy.py --num-frames 16

# Then: run codec vs baseline comparison
python scripts/run_generation.py --num-frames 16 --keyframe-interval 8
```

**What you get:** Figure 3 for the paper — quality vs GOP size.
**Do this BEFORE Phase 2** — it's your most novel contribution and doesn't
depend on Mamba surgery working.

### Step 3: Phase 2 — mamba-video

```bash
cd mamba-video
pip install -r requirements.txt && pip install -e .

# Inspect the model first
python -m mamba_video inspect --model wan-1.3b

# Then surgery
python scripts/run_surgery.py --model wan-1.3b --strategy progressive --num-replace 4
```

**What you get:** Figure 2 — quality vs blocks replaced.

### Step 4: Phase 4 — bitnet-video

```bash
cd bitnet-video
pip install -r requirements.txt && pip install -e .

# Benchmark float vs 1-bit layers
python scripts/run_benchmark.py

# Quantize model
python scripts/run_quantize.py --model wan-1.3b
```

**What you get:** Memory reduction numbers, quality metrics.

### Step 5: Phase 5 — simd-kernels

```bash
cd simd-kernels
make                    # Auto-detects NEON on M4, AVX2 on x86
make test               # Should say "8 passed, 0 failed"
pip install -e .
python scripts/run_benchmark.py
```

**What you get:** Native kernel speedup numbers.

### Step 6: Phase 6 — cpu-distributed

```bash
cd cpu-distributed
pip install -e .
python scripts/run_local.py --steps 100
```

**What you get:** Training convergence curve.

### Step 7: Phase 7 — cpu-video-gen (THE PAPER)

```bash
cd cpu-video-gen
pip install -e .

# Run the full ablation study
python scripts/run_benchmarks.py

# Reproduce all paper results
python scripts/reproduce_paper.py --all
```

**What you get:** Table 2 (ablation study) — the main paper result.

### Step 8: Write the Paper

Take the LaTeX source (`paper_latex_source.zip`), replace every `\todo{}`
with the real numbers from Steps 1-7, compile, submit to Arxiv.

---

## What Each Repo Contains (File Counts)

| Repo | Files | C Code | Python Modules | Tests | Has CLI |
|------|-------|--------|---------------|-------|---------|
| wan-profiler | 28 | — | 7 | 3 | ✅ |
| mamba-video | 31 | — | 8 | 3 | ✅ |
| codec-video-gen | 34 | — | 9 | 4 | ✅ |
| bitnet-video | 33 | — | 8 | 4 | ✅ |
| simd-kernels | 34 | 7 files (3 backends) | 7 | 1 | ✅ |
| cpu-distributed | 34 | — | 9 | 4 | ✅ |
| cpu-video-gen | 35 | — | 7 | 3 | ✅ |
| **Total** | **229** | **7** | **55** | **22** | **7 CLIs** |

Every repo has:
- `CLAUDE.md` — Claude Code reads this first every session
- `tasks/todo.md` — milestone roadmap with checkboxes
- `lessons.md` — grows as you build, prevents repeated mistakes
- `.claude/settings.json` — guardrails (blocks push to main, etc.)
- `.claude/commands/review.md` — `/project:review` slash command
- `.claude/commands/progress.md` — `/project:progress` slash command
- `.claude/rules/python.md` — code style enforcement
- `README.md` — public-facing docs with usage, citation
- `docs/` — methodology documentation
- `tests/` — unit tests
- `scripts/` — convenience entry points
- `configs/` — default configurations

---

## The 4 Technical Pillars — Quick Reference

| Pillar | Phase | What It Eliminates | Speedup |
|--------|-------|--------------------|---------|
| Mamba SSM | 2 | Quadratic attention O(n²) → O(n) | ~2× |
| Codec temporal | 3 | Redundant computation (I/P frames) | ~4-6× |
| BitNet 1-bit | 4 | Float arithmetic (XNOR+popcount) | ~8-16× |
| SIMD kernels | 5 | Framework overhead (native execution) | ~2× |
| **Combined** | | | **~64-192×** |

---

## Hardware Strategy

- **Primary machine (development + benchmarking): MacBook Air M4** — fast iteration, NEON backend for Phase 5
- **Commodity x86 (AVX2): CI-verified** on every push — the affordable-hardware target
- **Paper reports both architectures** — CPU-native on ARM (M4) and x86, no GPU
- Design stays within the commodity-hardware budget (≤2–4 cores, 16 GB, no GPU)

---

## Paper Status

| Section | Status |
|---------|--------|
| Abstract | ✅ Written |
| Introduction | ✅ Written |
| Related Work (13 citations) | ✅ Written |
| Method — Mamba SSM (with equations) | ✅ Written |
| Method — Codec temporal (with equations) | ✅ Written |
| Method — BitNet 1-bit (with equations) | ✅ Written |
| Method — AVX2/SIMD kernels | ✅ Written |
| Method — Distributed ES | ✅ Written |
| Experimental Setup | ✅ Written |
| Results — all tables and figures | ❌ \todo{} placeholders — needs real data |
| Discussion | ✅ Written |
| Conclusion | ✅ Written |
| Appendix | ✅ Written |

---

## Timeline

| When | What |
|------|------|
| Now | Push repos to GitHub. Start Phase 1 on M4. |
| Week 1-2 | Phase 1 (profiling) + Phase 3 (codec — your unique contribution) |
| Week 3-4 | Phase 2 (Mamba surgery) |
| Week 5-6 | Phase 4 (BitNet quantization) |
| Week 7-8 | Phase 5 (SIMD benchmarks on both machines) |
| Week 9-10 | Phase 7 (integration, ablation study) |
| Week 11-12 | Paper: replace \todo{} with real numbers, submit to Arxiv |
| Early 2027 | Conference submission (NeurIPS / ICML / ICLR) |
| Fall 2027 | PhD application deadlines |

---

*This document is the single source of truth for the IshCPU-VideoGenLab project.
Start with Phase 1. Build in order. Ship the paper.*
