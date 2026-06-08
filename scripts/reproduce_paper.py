#!/usr/bin/env python
"""Reproduce all paper results. Usage: python scripts/reproduce_paper.py"""
import sys, os, argparse, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Reproduce paper results")
    parser.add_argument("--table", type=int, default=None, help="Reproduce specific table")
    parser.add_argument("--figure", type=int, default=None, help="Reproduce specific figure")
    parser.add_argument("--all", action="store_true", help="Reproduce everything")
    parser.add_argument("--output", type=str, default="results")
    args = parser.parse_args()

    if args.all or args.table == 2 or args.table is None:
        logger.info("Reproducing Table 2: Ablation study...")
        from cpu_video_gen.config import BenchmarkConfig
        from cpu_video_gen.benchmark import run_ablation_study, format_ablation_table
        from cpu_video_gen.report import save_json
        results = run_ablation_study(BenchmarkConfig(output_dir=args.output))
        print(format_ablation_table(results))
        save_json(results, os.path.join(args.output, "paper"), "table2_ablation.json")
        logger.info("Table 2 saved.")

    logger.info("Paper reproduction complete. Results in: %s", args.output)

if __name__ == "__main__": main()
