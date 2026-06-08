"""Tests for cpu_video_gen.model and inference."""
import pytest
import torch
from cpu_video_gen.config import PipelineConfig
from cpu_video_gen.model import build_delta_predictor, _create_dummy_model
from cpu_video_gen.inference import generate_video


class TestModel:
    def test_dummy_model_forward(self) -> None:
        config = PipelineConfig(latent_shape=(4, 8, 8))
        model = _create_dummy_model(config)
        x = torch.randn(1, 4, 8, 8)
        with torch.no_grad():
            y = model(x)
        assert y.shape == x.shape

    def test_delta_predictor_builds(self) -> None:
        config = PipelineConfig(latent_shape=(4, 8, 8), delta_hidden_channels=32, delta_num_blocks=2)
        pred = build_delta_predictor(config)
        assert pred is not None
        x = torch.randn(1, 4, 8, 8)
        with torch.no_grad():
            y = pred(x)
        assert y.shape == x.shape


class TestInference:
    def test_generate_video_runs(self) -> None:
        import tempfile
        result = generate_video(
            prompt="test", num_frames=4, keyframe_interval=2,
            output_dir=tempfile.mkdtemp(),
        )
        assert result["num_i_frames"] + result["num_p_frames"] == 4
        assert result["total_time_ms"] > 0
