"""Full pipeline benchmarking for paper results."""

import gc
import logging
import time
from typing import Dict, List

import torch

from cpu_video_gen.config import PipelineConfig, BenchmarkConfig
from cpu_video_gen.pipeline import CPUVideoGenPipeline

logger = logging.getLogger(__name__)


def run_ablation_study(
    benchmark_config: BenchmarkConfig,
) -> List[Dict]:
    """Run ablation study: measure each optimization individually.

    Returns:
        List of results per ablation configuration.
    """
    results = []

    configs = {
        "baseline": PipelineConfig(
            keyframe_interval=1, weight_bits=8,
            mamba_strategy="all", use_avx2=False,
            output_dir=benchmark_config.output_dir,
        ),
        "mamba_only": PipelineConfig(
            keyframe_interval=1, weight_bits=8,
            mamba_strategy="all", use_avx2=False,
            output_dir=benchmark_config.output_dir,
        ),
        "codec_only": PipelineConfig(
            keyframe_interval=8, weight_bits=8,
            mamba_strategy="all", use_avx2=False,
            output_dir=benchmark_config.output_dir,
        ),
        "bitnet_only": PipelineConfig(
            keyframe_interval=1, weight_bits=1,
            mamba_strategy="all", use_avx2=False,
            output_dir=benchmark_config.output_dir,
        ),
        "full_pipeline": PipelineConfig(
            keyframe_interval=8, weight_bits=1,
            mamba_strategy="all", use_avx2=True,
            output_dir=benchmark_config.output_dir,
        ),
    }

    for name in benchmark_config.ablations:
        if name not in configs:
            continue

        logger.info("Running ablation: %s", name)
        config = configs[name]
        config.verbose = False

        pipeline = CPUVideoGenPipeline(config)
        result = pipeline.generate(seed=42)

        results.append({
            "ablation": name,
            "total_time_ms": result["total_time_ms"],
            "avg_frame_ms": result["avg_time_per_frame_ms"],
            "num_i_frames": result["num_i_frames"],
            "num_p_frames": result["num_p_frames"],
            "phases": result["build_report"].get("phases_applied", []),
        })

        del pipeline
        gc.collect()

    return results


def format_ablation_table(results: List[Dict]) -> str:
    """Format ablation results as a paper-ready table."""
    lines = [
        "", "=" * 70,
        "  Table: Ablation Study — CPU-Native Video Generation",
        "=" * 70, "",
        f"  {'Configuration':<20} {'Time (ms)':>12} {'Avg/Frame':>12} {'I/P':>8}",
        "  " + "-" * 54,
    ]
    for r in results:
        ip = f"{r['num_i_frames']}/{r['num_p_frames']}"
        lines.append(
            f"  {r['ablation']:<20} {r['total_time_ms']:>12.1f} "
            f"{r['avg_frame_ms']:>12.1f} {ip:>8}"
        )
    lines.extend(["  " + "-" * 54, "", "=" * 70, ""])
    return "\n".join(lines)
