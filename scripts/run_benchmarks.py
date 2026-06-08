#!/usr/bin/env python
"""Run full benchmark suite. Usage: python scripts/run_benchmarks.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cpu_video_gen.cli import main
if __name__ == "__main__": sys.exit(main(["benchmark"] + sys.argv[1:]))
