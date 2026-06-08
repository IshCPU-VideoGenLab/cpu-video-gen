#!/usr/bin/env python
"""Generate a video. Usage: python scripts/generate_video.py --prompt 'a cat walking'"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cpu_video_gen.cli import main
if __name__ == "__main__": sys.exit(main(["generate"] + sys.argv[1:]))
