#!/usr/bin/env python
"""Profile the full pipeline. Usage: python scripts/profile_pipeline.py"""
import sys, os, time, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
from cpu_video_gen.config import PipelineConfig
from cpu_video_gen.pipeline import CPUVideoGenPipeline

config = PipelineConfig(num_frames=8, verbose=True)
pipeline = CPUVideoGenPipeline(config)
result = pipeline.generate()
print(f"\nTotal: {result['total_time_ms']:.1f} ms")
for i, (ft, t) in enumerate(zip(result['schedule'], result['frame_times_ms'])):
    print(f"  Frame {i} [{ft}]: {t:.1f} ms")
