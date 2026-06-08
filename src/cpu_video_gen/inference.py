"""Inference entry point for CPU-native video generation."""

import logging
from typing import Dict, Optional

from cpu_video_gen.config import PipelineConfig
from cpu_video_gen.pipeline import CPUVideoGenPipeline

logger = logging.getLogger(__name__)


def generate_video(
    prompt: str = "a cat walking",
    num_frames: int = 16,
    keyframe_interval: int = 8,
    seed: int = 42,
    output_dir: str = "results",
) -> Dict:
    """High-level inference function.

    Args:
        prompt: Text prompt.
        num_frames: Number of frames.
        keyframe_interval: GOP size.
        seed: Random seed.
        output_dir: Output directory.

    Returns:
        Generation results dictionary.
    """
    config = PipelineConfig(
        num_frames=num_frames,
        keyframe_interval=keyframe_interval,
        output_dir=output_dir,
        seed=seed,
    )

    pipeline = CPUVideoGenPipeline(config)
    result = pipeline.generate(prompt=prompt, seed=seed)

    logger.info(
        "Generated %d frames (%d I, %d P) in %.1f ms",
        num_frames, result["num_i_frames"], result["num_p_frames"],
        result["total_time_ms"],
    )

    return result
