"""CLI for cpu-video-gen."""
import argparse, logging, sys
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="cpu-video-gen",
        description="CPU-native video generation pipeline")
    sub = parser.add_subparsers(dest="command")

    g = sub.add_parser("generate", help="Generate a video")
    g.add_argument("--prompt", type=str, default="a cat walking")
    g.add_argument("--frames", type=int, default=16)
    g.add_argument("--keyframe-interval", type=int, default=8)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--output", type=str, default="results")
    g.add_argument("--debug", action="store_true")

    b = sub.add_parser("benchmark", help="Run benchmark suite")
    b.add_argument("--output", type=str, default="results")
    b.add_argument("--debug", action="store_true")

    sub.add_parser("info", help="Show pipeline info")

    args = parser.parse_args(argv)
    if args.command is None:
        print("Usage: cpu-video-gen {generate|benchmark|info}")
        return 1

    level = logging.DEBUG if getattr(args, "debug", False) else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")

    if args.command == "generate":
        from cpu_video_gen.inference import generate_video
        from cpu_video_gen.report import save_json
        result = generate_video(
            prompt=args.prompt, num_frames=args.frames,
            keyframe_interval=args.keyframe_interval,
            seed=args.seed, output_dir=args.output,
        )
        # Remove tensor from JSON
        result_json = {k: v for k, v in result.items() if k != "frames"}
        save_json(result_json, args.output, "generation_result.json")
        print(f"\nGenerated {result['num_i_frames']+result['num_p_frames']} frames in {result['total_time_ms']:.1f} ms")
        return 0

    elif args.command == "benchmark":
        from cpu_video_gen.config import BenchmarkConfig
        from cpu_video_gen.benchmark import run_ablation_study, format_ablation_table
        from cpu_video_gen.report import save_json
        bench_config = BenchmarkConfig(output_dir=args.output)
        results = run_ablation_study(bench_config)
        print(format_ablation_table(results))
        save_json(results, args.output, "ablation_results.json")
        return 0

    elif args.command == "info":
        print("\ncpu-video-gen — CPU-Native Video Generation Pipeline")
        print("Author: Ishmael Affum Kwakye")
        print("Institution: University of Ghana, Legon")
        print("Org: github.com/IshCPU-VideoGenLab")
        print("\nComponents: Mamba SSM + Codec Temporal + BitNet 1-bit + AVX2 Kernels + ES Training")
        return 0

    return 1

if __name__ == "__main__":
    sys.exit(main())
