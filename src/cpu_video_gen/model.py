"""Full model assembly: Mamba SSM backbone + BitNet quantization.

Constructs the CPU-native video generation model by applying
Phase 2 (Mamba surgery) and Phase 4 (1-bit quantization) to
the base model.
"""

import gc
import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from cpu_video_gen.config import PipelineConfig

logger = logging.getLogger(__name__)


def build_model(config: PipelineConfig) -> Tuple[nn.Module, Dict[str, Any]]:
    """Build the full CPU-native model from config.

    Applies in order:
    1. Load base model
    2. Mamba surgery (replace attention with SSM)
    3. BitNet quantization (1-bit weights)

    Args:
        config: Pipeline configuration.

    Returns:
        Tuple of (model, build_report).
    """
    report = {"phases_applied": []}

    # Step 1: Create or load a base model
    logger.info("Building CPU-native model...")

    try:
        from transformers import AutoModel

        dtype_map = {"float16": torch.float16, "float32": torch.float32}
        torch_dtype = dtype_map.get(config.dtype, torch.float16)

        path = config.model_path or config.model_name
        model = AutoModel.from_pretrained(
            path, torch_dtype=torch_dtype,
            trust_remote_code=True, low_cpu_mem_usage=True,
        )
        model.eval()
        report["base_model"] = config.model_name
        report["base_params"] = sum(p.numel() for p in model.parameters())

    except Exception as e:
        logger.warning("Could not load base model: %s. Using dummy model.", e)
        model = _create_dummy_model(config)
        report["base_model"] = "dummy"
        report["base_params"] = sum(p.numel() for p in model.parameters())

    # Step 2: Mamba surgery
    try:
        from mamba_video.config import MambaConfig, SurgeryConfig
        from mamba_video.surgery import perform_surgery

        surgery_config = SurgeryConfig(
            model_name=config.model_name,
            strategy=config.mamba_strategy,
            output_dir=config.output_dir,
            mamba=MambaConfig(
                d_state=config.d_state,
                d_conv=config.d_conv,
                expand=config.mamba_expand,
            ),
        )
        model, surgery_report = perform_surgery(surgery_config)
        report["phases_applied"].append("mamba_surgery")
        report["mamba_replaced"] = surgery_report.replaced_blocks
        logger.info("Phase 2 (Mamba): %d blocks replaced", surgery_report.replaced_blocks)

    except ImportError:
        logger.info("mamba-video not installed. Skipping Mamba surgery.")

    # Step 3: BitNet quantization
    try:
        from bitnet_video.config import QuantConfig
        from bitnet_video.converter import quantize_model

        quant_config = QuantConfig(
            weight_bits=config.weight_bits,
            activation_bits=config.activation_bits,
            skip_patterns=config.skip_quant_patterns,
        )
        model, conv_report = quantize_model(model, quant_config)
        report["phases_applied"].append("bitnet_quantization")
        report["quantized_layers"] = conv_report.total_quantized
        logger.info("Phase 4 (BitNet): %d layers quantized", conv_report.total_quantized)

    except ImportError:
        logger.info("bitnet-video not installed. Skipping quantization.")

    report["final_params"] = sum(p.numel() for p in model.parameters())
    gc.collect()

    return model, report


def build_delta_predictor(config: PipelineConfig) -> nn.Module:
    """Build the delta predictor for P-frame generation.

    Args:
        config: Pipeline configuration.

    Returns:
        Delta predictor model.
    """
    try:
        from codec_video_gen.delta_gen import create_delta_predictor

        predictor = create_delta_predictor(
            in_channels=config.latent_shape[0],
            hidden_channels=config.delta_hidden_channels,
            num_res_blocks=config.delta_num_blocks,
        )

        # Optionally quantize the delta predictor too
        if config.weight_bits == 1:
            try:
                from bitnet_video.config import QuantConfig
                from bitnet_video.converter import quantize_model
                quant_config = QuantConfig(weight_bits=1, activation_bits=config.activation_bits)
                predictor, _ = quantize_model(predictor, quant_config)
                logger.info("Delta predictor quantized to 1-bit")
            except ImportError:
                pass

        return predictor

    except ImportError:
        logger.warning("codec-video-gen not installed. Using dummy delta predictor.")
        return nn.Sequential(
            nn.Conv2d(config.latent_shape[0], 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, config.latent_shape[0], 3, padding=1),
        )


def _create_dummy_model(config: PipelineConfig) -> nn.Module:
    """Create a small dummy model for testing without real weights."""
    c = config.latent_shape[0]
    return nn.Sequential(
        nn.Conv2d(c, 64, 3, padding=1), nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        nn.Conv2d(64, c, 3, padding=1),
    )
