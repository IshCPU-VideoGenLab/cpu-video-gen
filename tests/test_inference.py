"""Tests for cpu_video_gen inference CLI."""
import pytest
from cpu_video_gen.cli import main


class TestCLI:
    def test_info_command(self) -> None:
        result = main(["info"])
        assert result == 0

    def test_unknown_command(self) -> None:
        result = main(["nonexistent"])
        assert result == 1
