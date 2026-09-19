import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from utils.flowvn_results import REVISION_METRICS


ROOT = Path(__file__).resolve().parents[1]


class AnalysisCliTests(unittest.TestCase):
    def test_csv_workflow_recovers_known_paired_difference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = [sys.executable, "scripts/analyze.py"]
            for seed in (12345, 23456):
                for arm, shift in (("supervised", 0), ("final_only", 0.1),
                                   ("final_trajectory", 0.2), ("full_kd", 0.3)):
                    path = root / f"{arm}_{seed}.csv"
                    with path.open("w", newline="") as handle:
                        writer = csv.DictWriter(
                            handle, fieldnames=["case_id", "slice_start", "usrate", *REVISION_METRICS],
                        )
                        writer.writeheader()
                        for case in (0, 1):
                            for rate in (10, 20):
                                values = {metric: 1.0 + case - shift for metric in REVISION_METRICS}
                                values["ssim"] = 0.5 + shift
                                writer.writerow(dict(case_id=f"case_{case}-aaaaaaaaaaaa",
                                                     slice_start=0, usrate=rate, **values))
                    command += ["--run", str(seed), arm, str(path)]
            command += ["--expected-cases", "2", "--expected-usrates", "10", "20",
                        "--bootstrap-samples", "200", "--output-dir", str(root / "analysis")]
            result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            analysis = json.loads((root / "analysis/ablation_analysis.json").read_text())
            effect = analysis["contrasts"]["final_output_matching"]["overall"]["nrmse"]
            self.assertAlmostEqual(effect["mean_delta_treatment_minus_control"], -0.1)
            self.assertEqual(len(analysis["inputs"]), 8)
            # A second invocation must not silently replace a previous analysis.
            repeat = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=60)
            self.assertNotEqual(repeat.returncode, 0)


if __name__ == "__main__":
    unittest.main()
