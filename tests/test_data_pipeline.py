"""Exercise the documented HDF5 input format using fictional complex data."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np

from utils.dataloader_CMRx4DFlow import CMRx4DFlowDataSet


ROOT = Path(__file__).resolve().parents[1]


def write_case(path):
    rng = np.random.default_rng(7)
    # Non-square spatial dimensions catch accidental axis swaps.
    kspace = rng.standard_normal((4, 3, 2, 6, 7, 5)).astype(np.complex64)
    kspace += 1j * rng.standard_normal(kspace.shape).astype(np.float32)
    coils = np.ones((2, 6, 7, 5), dtype=np.complex64) / np.sqrt(2)
    mask = np.ones((1, 3, 1, 6, 7, 1), dtype=np.float32)
    mask[:, :, :, 1::2] = 0
    for name, value in {"kdata_ktGaussian": kspace, "coilmap": coils,
                        "segmask": np.ones((6, 7, 5)), "usmask_ktGaussian": mask}.items():
        filename = name + ("10" if "ktGaussian" in name else "") + ".mat"
        if np.iscomplexobj(value):
            compound = np.empty(value.shape, dtype=[("real", "<f4"), ("imag", "<f4")])
            compound["real"], compound["imag"] = value.real, value.imag
            value = compound
        with h5py.File(path / filename, "w") as handle:
            handle[name] = value
    (path / "params.csv").write_text('VENC\n"100;150;200"\n')


class DataPipelineTests(unittest.TestCase):
    def test_stored_masks_axes_and_velocity_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            write_case(path)
            dataset = CMRx4DFlowDataSet(mode="test", input=str(path), loss="supervised",
                                       network="FlowVN", D_size=5, T_size=3, usrate=[10])
            self.assertEqual(len(dataset), 4)
            sample = dataset[0]
            self.assertEqual(sample["imdata_p1"].shape, (1, 3, 5, 7, 6))
            self.assertEqual(sample["kdata_p1"].shape, (1, 2, 3, 5, 7, 6))
            self.assertTrue(np.iscomplexobj(sample["imdata_p1"]))
            self.assertTrue(np.isfinite(sample["imdata_p1"]).all())
            self.assertTrue(np.all(sample["kdata_p1"][..., 1::2] == 0))
            np.testing.assert_array_equal(sample["VENC"], [100, 150, 200])

    def test_real_case_model_benchmark_uses_default_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            write_case(path)
            result = subprocess.run(
                [sys.executable, "scripts/benchmark_flowvn.py", "--case-dir", str(path),
                 "--device", "cpu", "--usrate", "10", "--num-stages", "1",
                 "--features-out", "2", "--kernel-size", "3",
                 "--num-warmup", "1", "--num-iters", "1"],
                cwd=ROOT, text=True, capture_output=True, timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertTrue(np.isfinite(report["max_abs_diff"]))


if __name__ == "__main__":
    unittest.main()
