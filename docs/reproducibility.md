# Reproducibility Guide

## Requirements

- Python 3.9+
- 16 GB RAM minimum
- CPU with AVX2 support (any post-2013 Intel/AMD)
- ~10 GB disk space
- Linux, macOS, or Windows with WSL

## Step-by-Step Reproduction

### 1. Clone and Setup

```bash
git clone https://github.com/IshCPU-VideoGenLab/cpu-video-gen.git
cd cpu-video-gen
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2. Install Phase Dependencies (Optional)

For full integration with all phases:

```bash
# Clone sibling repos
cd ..
git clone https://github.com/IshCPU-VideoGenLab/mamba-video.git
git clone https://github.com/IshCPU-VideoGenLab/codec-video-gen.git
git clone https://github.com/IshCPU-VideoGenLab/bitnet-video.git
git clone https://github.com/IshCPU-VideoGenLab/simd-kernels.git

# Install each
for repo in mamba-video codec-video-gen bitnet-video simd-kernels; do
    cd $repo && pip install -e . && cd ..
done

# Build SIMD kernels (auto-detects AVX2 on x86, NEON on ARM)
cd simd-kernels && make && cd ..
cd cpu-video-gen
```

### 3. Reproduce All Paper Results

```bash
python scripts/reproduce_paper.py --all --output results/paper/
```

### 4. Verify Results

All generated files appear in `results/paper/`:
- `table2_ablation.json` — Table 2 data
- Figures regenerated in `paper/figures/`

### Environment Pinning

```bash
pip freeze > requirements.lock
```

The `requirements.lock` file in the repo pins exact versions used
for the paper results.

## Hardware Used

Primary benchmark machine:
- Intel Pentium Gold 7505
- 2 cores / 4 threads, 3.5 GHz
- 16 GB DDR4 3200 MHz (single channel)
- Ubuntu 22.04 / Windows 11 (WSL2)

Results may vary on different hardware. The relative speedups
(ablation ratios) should be consistent across machines.
