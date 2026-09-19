"""Check the commands a reader uses without any private data or weights."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ReleaseCliTests(unittest.TestCase):
    def test_paper_recipe_rejects_local_masks_before_reading_data(self):
        env = dict(os.environ, FLOWVN_MASK_BACKEND="local")
        env.pop("FLOWVN_CHALLENGE_MASK_ARCHIVE", None)
        result = subprocess.run(
            [sys.executable, "main.py", "--config", "configs/s8_full.yaml"],
            cwd=ROOT, env=env, text=True, capture_output=True, timeout=60,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires the challenge mask backend", result.stderr)

    def test_synthetic_benchmark_creates_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "new_directory/timing.json"
            result = subprocess.run(
                [sys.executable, "scripts/benchmark_flowvn.py", "--device", "cpu",
                 "--num-stages", "1", "--features-out", "2", "--kernel-size", "3",
                 "--depth", "3", "--time-frames", "3", "--spatial-size", "5",
                 "--coils", "2", "--num-warmup", "1", "--num-iters", "1",
                 "--output-json", str(output)],
                cwd=ROOT, text=True, capture_output=True, timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(output.is_file())

    def test_real_data_benchmarks_reject_wrong_mask_backend(self):
        env = dict(os.environ, FLOWVN_MASK_BACKEND="local")
        env.pop("FLOWVN_CHALLENGE_MASK_ARCHIVE", None)
        for script in ("benchmark_flowvn.py", "benchmark_flowvn_e2e.py"):
            with self.subTest(script=script):
                result = subprocess.run(
                    [sys.executable, f"scripts/{script}", "--config", "configs/validate.yaml",
                     "--case-dir", "missing_synthetic_case", "--real-mode", "val", "--device", "cpu"],
                    cwd=ROOT, env=env, text=True, capture_output=True, timeout=60,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("requires the challenge mask backend", result.stderr)


if __name__ == "__main__":
    unittest.main()
