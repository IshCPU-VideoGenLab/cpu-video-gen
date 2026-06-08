"""Unified configuration for the cpu-video-gen pipeline."""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class PipelineConfig:
    """Complete configuration for CPU-native video generation.

    Integrates settings from all phases into a single config.

    Args:
        model_name: Base model name.
        num_frames: Total video frames to generate.
        resolution: Output resolution (H, W).
        keyframe_interval: GOP size for codec temporal design.
        d_state: Mamba SSM state dimension.
        d_conv: Mamba convolution width.
        mamba_expand: Mamba expansion factor.
        weight_bits: Quantization bits for weights.
        activation_bits: Quantization bits for activations.
        skip_quant_patterns: Module patterns to keep in float.
        use_avx2: Use AVX2 native kernels if available.
        delta_hidden_channels: Delta predictor hidden dim.
        delta_num_blocks: Delta predictor residual blocks.
        dtype: Data type for non-quantized ops.
        output_dir: Output directory.
        seed: Random seed for reproducibility.
        verbose: Print progress.
    """

    # Model
    model_name: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    model_path: Optional[str] = None

    # Video
    num_frames: int = 16
    resolution: Tuple[int, int] = (256, 256)
    latent_shape: Tuple[int, int, int] = (4, 32, 32)

    # Phase 2: Mamba
    mamba_strategy: str = "all"
    d_state: int = 16
    d_conv: int = 4
    mamba_expand: int = 2

    # Phase 3: Codec
    keyframe_interval: int = 8
    error_correction: bool = True
    max_delta_norm: float = 10.0
    delta_hidden_channels: int = 128
    delta_num_blocks: int = 4

    # Phase 4: BitNet
    weight_bits: int = 1
    activation_bits: int = 8
    skip_quant_patterns: List[str] = field(default_factory=lambda: [
        "norm", "embed", "layernorm",
    ])

    # Phase 5: AVX2
    use_avx2: bool = True

    # General
    dtype: str = "float16"
    output_dir: str = "results"
    seed: int = 42
    verbose: bool = True

    def __post_init__(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""
    num_warmup: int = 3
    num_steps: int = 10
    output_dir: str = "results"
    ablations: List[str] = field(default_factory=lambda: [
        "baseline", "mamba_only", "codec_only", "bitnet_only",
        "avx2_only", "full_pipeline",
    ])

    def __post_init__(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
