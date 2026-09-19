"""Compute matched, case-level ablations from newly generated validation CSVs."""

import argparse
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_flowvn_revision_ablation import (
    analyze_revision_ablation,
    write_analysis_outputs,
)
from utils.flowvn_results import load_validation_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", nargs=3, required=True,
                        metavar=("SEED", "ARM", "CSV"))
    parser.add_argument("--expected-seeds", nargs="+", default=["12345", "23456"])
    parser.add_argument("--expected-cases", type=int, default=16)
    parser.add_argument("--expected-usrates", type=int, nargs="+", default=[10, 20, 30, 40, 50])
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260809)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    runs, inputs = {}, []
    for seed, arm, filename in args.run:
        key = (seed, arm)
        if key in runs:
            parser.error(f"Duplicate run: seed={seed}, arm={arm}")
        path = Path(filename).resolve()
        runs[key] = load_validation_csv(path)
        inputs.append({"seed": seed, "arm": arm, "validation_csv_path": str(path),
                       "validation_csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

    analysis = analyze_revision_ablation(
        runs, expected_seeds=args.expected_seeds,
        expected_cases=args.expected_cases, expected_usrates=args.expected_usrates,
        bootstrap_samples=args.bootstrap_samples, bootstrap_seed=args.bootstrap_seed,
    )
    analysis["inputs"] = inputs
    paths = write_analysis_outputs(analysis, output_dir=args.output_dir, prefix="ablation")
    print(paths["analysis"])


if __name__ == "__main__":
    main()
