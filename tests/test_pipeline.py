"""Tests for cpu_video_gen.pipeline."""
import pytest
import torch
from cpu_video_gen.config import PipelineConfig
from cpu_video_gen.pipeline import CPUVideoGenPipeline


class TestCPUVideoGenPipeline:
    def _config(self, **kwargs) -> PipelineConfig:
        import tempfile
        defaults = {"output_dir": tempfile.mkdtemp(), "num_frames": 4,
                    "keyframe_interval": 2, "latent_shape": (4, 8, 8), "verbose": False}
        defaults.update(kwargs)
        return PipelineConfig(**defaults)

    def test_generate_returns_frames(self) -> None:
        config = self._config()
        pipeline = CPUVideoGenPipeline(config)
        result = pipeline.generate()
        assert result["frames"].shape[0] == 4

    def test_schedule_correct(self) -> None:
        config = self._config(num_frames=8, keyframe_interval=4)
        pipeline = CPUVideoGenPipeline(config)
        result = pipeline.generate()
        assert result["schedule"] == ["I", "P", "P", "P", "I", "P", "P", "P"]

    def test_timing_recorded(self) -> None:
        config = self._config()
        pipeline = CPUVideoGenPipeline(config)
        result = pipeline.generate()
        assert len(result["frame_times_ms"]) == 4
        assert result["total_time_ms"] > 0

    def test_reproducible(self) -> None:
        config = self._config()
        p1 = CPUVideoGenPipeline(config)
        p2 = CPUVideoGenPipeline(config)
        r1 = p1.generate(seed=123)
        r2 = p2.generate(seed=123)
        assert torch.allclose(r1["frames"], r2["frames"])
