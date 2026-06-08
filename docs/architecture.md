# Architecture

## Full Pipeline

```
Text Prompt → [Tokenizer] → Conditioning
                                │
                   ┌────────────┴────────────┐
                   │    GOP Scheduler         │
                   │ (I/P frame assignment)   │
                   └────────────┬────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │ I-frame         │ P-frame          │
              ▼                 ▼                  │
    ┌──────────────┐   ┌───────────────┐          │
    │ Full Model   │   │ Delta         │          │
    │ (Mamba SSM   │   │ Predictor     │          │
    │  + BitNet    │   │ (BitNet Conv) │          │
    │  1-bit)      │   │              │          │
    └──────┬───────┘   └──────┬────────┘          │
           │                  │                    │
           └────────┬─────────┘                    │
                    ▼                              │
         ┌──────────────────┐                     │
         │ AVX2 Execution   │                     │
         │ Engine           │                     │
         │ (XNOR+popcount,  │                     │
         │  vectorized scan)│                     │
         └────────┬─────────┘                     │
                  ▼                                │
           Video Latents ──────────────────────────┘
                  │
                  ▼
            [VAE Decoder]
                  │
                  ▼
            Video Frames
```

## Component Details

### 1. Mamba SSM Backbone (Phase 2)
- Replaces all self-attention blocks with Selective State Space Models
- O(n) complexity vs O(n²) for attention
- Sequential scan: CPU-cache-friendly processing
- d_state=16, expand=2

### 2. Codec Temporal Design (Phase 3)
- GOP scheduler assigns I-frame or P-frame type
- I-frames: full model generation (expensive, high quality)
- P-frames: delta predictor from previous frame (cheap)
- Typical: 2 I-frames + 14 P-frames in 16-frame video
- Compute savings: ~4-6×

### 3. BitNet 1-bit Quantization (Phase 4)
- All Linear and Conv2d weights: {-1, +1}
- Activations: 8-bit symmetric quantization
- Normalization and embeddings: kept in float
- Memory: ~16× reduction
- Compute: XNOR + popcount replaces float matmul

### 4. AVX2 Native Kernels (Phase 5)
- Binary GEMM: 256 bits per XNOR instruction
- SSM scan: vectorized across d_state with FMA
- 1-bit Conv: im2col + binary GEMM
- Fallback to PyTorch if not compiled

### 5. Distributed Training (Phase 6)
- Evolution strategies: gradient-free optimization
- Communication: seeds + scalar fitness values
- Works over WiFi between commodity laptops
- Embarrassingly parallel

## Speedup Composition

Each optimization multiplies:

```
Baseline (PyTorch on CPU):               1.0×
+ Mamba (remove O(n²) attention):        ~2×    → 2×
+ Codec (I/P frame scheduling):          ~5×    → 10×
+ BitNet (1-bit binary operations):      ~10×   → 100×
+ AVX2 (native SIMD execution):          ~2×    → 200×
                                         ─────
                        Theoretical:     ~200× combined
```

Conservative estimate: 50-100× real-world speedup.
