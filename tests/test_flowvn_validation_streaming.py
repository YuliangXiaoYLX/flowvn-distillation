import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.flowvn_validation import (
    StreamingValidationAccumulator,
    build_qualitative_source,
    select_validation_filename_indices,
    stable_case_id,
)


class StreamingValidationAccumulatorTests(unittest.TestCase):
    def test_physical_velocity_metric_is_persisted_in_group_outputs(self):
        def evaluate_group(key, pred, gt, seg, venc, want_visualization):
            return {
                "nrmse": 0.1,
                "ssim": 0.9,
                "relerr": 0.2,
                "angerr": 10.0,
                "velocity_vector_rmse_cm_s": 4.25,
            }, None

        accumulator = StreamingValidationAccumulator(evaluate_group)
        for encoding in range(4):
            row = accumulator.add_encoding(
                key=("/synthetic/site/case_a", 0, 10),
                encoding=encoding,
                pred=np.zeros((1, 1), dtype=np.complex64),
                gt=np.zeros((1, 1), dtype=np.complex64),
                segmentation=np.ones((1, 1), dtype=bool),
                venc=np.array([150.0, 150.0, 150.0], dtype=np.float32),
                normalized_l1=0.2,
            )

        self.assertAlmostEqual(row["velocity_vector_rmse_cm_s"], 4.25)
        self.assertAlmostEqual(
            accumulator.summary()["velocity_vector_rmse_cm_s"], 4.25
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path, json_path = accumulator.write_outputs(
                Path(tmpdir) / "validation_metrics.csv"
            )
            with csv_path.open(newline="") as f:
                csv_rows = list(csv.DictReader(f))
            report = json.loads(json_path.read_text())

        self.assertAlmostEqual(
            float(csv_rows[0]["velocity_vector_rmse_cm_s"]), 4.25
        )
        self.assertAlmostEqual(
            report["summary"]["velocity_vector_rmse_cm_s"], 4.25
        )

    def test_semantic_validation_filter_resolves_exact_four_encoding_group(self):
        case_a = "/synthetic/site_a/scanner_a/case_a"
        case_b = "/synthetic/site_b/scanner_b/case_a"
        filenames = []
        for case_dir in (case_a, case_b):
            for usrate in (10, 40):
                for encoding in range(4):
                    filenames.append([case_dir, 0, usrate, encoding, "/output"])

        indices = select_validation_filename_indices(
            filenames,
            case_id=stable_case_id(case_b),
            usrate=40,
        )

        self.assertEqual(indices, [12, 13, 14, 15])

    def test_qualitative_source_uses_one_time_and_slice_with_roi_mask(self):
        mag_pred = np.arange(2 * 2 * 3 * 4 * 5, dtype=np.float32).reshape(
            2, 2, 3, 4, 5
        )
        mag_gt = mag_pred + 1
        flow_pred = -mag_pred[:1]
        flow_gt = flow_pred - 1
        seg = np.zeros((3, 4, 5), dtype=bool)
        seg[1, 1:3, 2:5] = True

        source = build_qualitative_source(
            mag_pred=mag_pred,
            mag_gt=mag_gt,
            flow_pred=flow_pred,
            flow_gt=flow_gt,
            segmentation=seg,
            time_index=None,
        )

        expected_roi = seg[1].T.astype(np.uint8)
        self.assertTrue(np.array_equal(source["roi"], expected_roi))
        self.assertEqual(source["magnitude_pred"].shape, (2, 5, 4))
        self.assertEqual(source["velocity_pred"].shape, (1, 5, 4))
        self.assertTrue(
            np.array_equal(
                source["magnitude_pred"][0],
                np.abs(mag_pred[0, 1, 1]).T * expected_roi,
            )
        )
        self.assertEqual(int(source["time_index"]), 1)
        self.assertEqual(int(source["slice_index"]), 1)

    def test_completed_four_encoding_group_is_evaluated_and_released(self):
        calls = []

        def evaluate_group(key, pred, gt, seg, venc, want_visualization):
            calls.append((key, pred.shape, gt.shape, want_visualization))
            return {
                "nrmse": 0.1,
                "ssim": 0.9,
                "relerr": 0.2,
                "angerr": 10.0,
            }, np.zeros((3, 4), dtype=np.uint8)

        accumulator = StreamingValidationAccumulator(evaluate_group)
        key = ("/synthetic/site/case_a", 0, 10)

        for encoding in range(3):
            row = accumulator.add_encoding(
                key=key,
                encoding=encoding,
                pred=np.full((2, 3), encoding, dtype=np.complex64),
                gt=np.zeros((2, 3), dtype=np.complex64),
                segmentation=np.ones((2, 3), dtype=bool),
                venc=np.array([150.0, 150.0, 150.0], dtype=np.float32),
                normalized_l1=0.1 * (encoding + 1),
            )
            self.assertIsNone(row)

        self.assertEqual(accumulator.pending_group_count, 1)
        self.assertEqual(accumulator.completed_group_count, 0)

        row = accumulator.add_encoding(
            key=key,
            encoding=3,
            pred=np.full((2, 3), 3, dtype=np.complex64),
            gt=np.zeros((2, 3), dtype=np.complex64),
            segmentation=np.ones((2, 3), dtype=bool),
            venc=np.array([150.0, 150.0, 150.0], dtype=np.float32),
            normalized_l1=0.4,
        )

        self.assertEqual(accumulator.pending_group_count, 0)
        self.assertEqual(accumulator.completed_group_count, 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], (4, 2, 3))
        self.assertTrue(calls[0][3])
        self.assertRegex(row["case_id"], r"^case_a-[0-9a-f]{12}$")
        self.assertEqual(row["usrate"], 10)
        self.assertAlmostEqual(row["normalized_l1"], 0.25)
        self.assertEqual(len(accumulator.visualizations), 1)

    def test_summary_and_outputs_are_grouped_by_usrate_and_hide_source_paths(self):
        metric_values = iter(
            [
                {"nrmse": 0.1, "ssim": 0.9, "relerr": 0.2, "angerr": 10.0},
                {"nrmse": 0.3, "ssim": 0.7, "relerr": None, "angerr": np.nan},
            ]
        )

        def evaluate_group(key, pred, gt, seg, venc, want_visualization):
            return next(metric_values), None

        accumulator = StreamingValidationAccumulator(evaluate_group)
        for case_id, usrate, loss in (("case_a", 10, 0.2), ("case_b", 20, 0.4)):
            for encoding in range(4):
                accumulator.add_encoding(
                    key=(f"/private/source/{case_id}", 0, usrate),
                    encoding=encoding,
                    pred=np.zeros((1, 1), dtype=np.complex64),
                    gt=np.zeros((1, 1), dtype=np.complex64),
                    segmentation=np.ones((1, 1), dtype=bool),
                    normalized_l1=loss,
                )

        summary = accumulator.summary()
        self.assertEqual(summary["n_groups"], 2)
        self.assertEqual(summary["n_complete"], 2)
        self.assertEqual(summary["n_incomplete"], 0)
        self.assertAlmostEqual(summary["nrmse"], 0.2)
        self.assertAlmostEqual(summary["ssim"], 0.8)
        self.assertAlmostEqual(summary["relerr"], 0.2)
        self.assertAlmostEqual(summary["angerr"], 10.0)
        self.assertAlmostEqual(summary["normalized_l1"], 0.3)

        by_usrate = accumulator.summary_by_usrate()
        self.assertAlmostEqual(by_usrate[10]["nrmse"], 0.1)
        self.assertAlmostEqual(by_usrate[20]["nrmse"], 0.3)

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path, json_path = accumulator.write_outputs(
                Path(tmpdir) / "validation_metrics.csv"
            )
            with csv_path.open(newline="") as f:
                rows = list(csv.DictReader(f))
            with json_path.open() as f:
                report = json.load(f)

        self.assertTrue(rows[0]["case_id"].startswith("case_a-"))
        self.assertTrue(rows[1]["case_id"].startswith("case_b-"))
        self.assertNotIn("/private/source", json.dumps(rows))
        self.assertEqual(
            report["case_id_convention"],
            "basename-sha256_12(last_three_path_components)",
        )
        self.assertAlmostEqual(report["summary"]["nrmse"], 0.2)
        self.assertAlmostEqual(report["by_usrate"]["10"]["nrmse"], 0.1)

    def test_same_local_case_name_at_different_centers_has_distinct_ids(self):
        def evaluate_group(key, pred, gt, seg, venc, want_visualization):
            return {
                "nrmse": 0.1,
                "ssim": 0.9,
                "relerr": 0.2,
                "angerr": 10.0,
            }, None

        accumulator = StreamingValidationAccumulator(evaluate_group)
        case_dirs = (
            "/synthetic/site_a/scanner_a/case_a",
            "/synthetic/site_b/scanner_b/case_a",
        )
        for case_dir in case_dirs:
            for encoding in range(4):
                accumulator.add_encoding(
                    key=(case_dir, 0, 10),
                    encoding=encoding,
                    pred=np.zeros((1, 1), dtype=np.complex64),
                    gt=np.zeros((1, 1), dtype=np.complex64),
                    segmentation=np.ones((1, 1), dtype=bool),
                    normalized_l1=0.2,
                )

        case_ids = [row["case_id"] for row in accumulator.rows]
        self.assertEqual(len(case_ids), 2)
        self.assertEqual(len(set(case_ids)), 2)
        self.assertTrue(all(case_id.startswith("case_a-") for case_id in case_ids))

    def test_visualization_sources_are_written_with_a_hash_manifest(self):
        panel = np.arange(12, dtype=np.uint8).reshape(3, 4)
        magnitude = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        velocity = np.linspace(-1.0, 1.0, 12, dtype=np.float32).reshape(1, 3, 4)

        def evaluate_group(key, pred, gt, seg, venc, want_visualization):
            return {
                "nrmse": 0.1,
                "ssim": 0.9,
                "relerr": 0.2,
                "angerr": 10.0,
            }, {
                "panel": panel,
                "source": {
                    "magnitude_pred": magnitude,
                    "magnitude_gt": magnitude + 1,
                    "velocity_pred": velocity,
                    "velocity_gt": velocity + 1,
                    "roi": np.ones((3, 4), dtype=np.uint8),
                },
            }

        accumulator = StreamingValidationAccumulator(evaluate_group)
        for encoding in range(4):
            accumulator.add_encoding(
                key=("/synthetic/site/case_a", 0, 40),
                encoding=encoding,
                pred=np.zeros((2, 3), dtype=np.complex64),
                gt=np.zeros((2, 3), dtype=np.complex64),
                segmentation=np.ones((2, 3), dtype=bool),
                normalized_l1=0.2,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path, manifest_path = accumulator.write_visualizations(
                Path(tmpdir) / "validation_visualizations.npz"
            )
            archive_bytes = npz_path.read_bytes()
            manifest = json.loads(manifest_path.read_text())
            with np.load(npz_path, allow_pickle=False) as archive:
                self.assertTrue(np.array_equal(archive["sample_000_panel"], panel))
                self.assertTrue(
                    np.array_equal(archive["sample_000_magnitude_pred"], magnitude)
                )
                self.assertTrue(
                    np.array_equal(archive["sample_000_velocity_gt"], velocity + 1)
                )

        self.assertEqual(manifest["count"], 1)
        self.assertEqual(manifest["samples"][0]["case_id"].split("-")[0], "case_a")
        self.assertEqual(manifest["samples"][0]["usrate"], 40)
        self.assertEqual(manifest["archive_sha256"], hashlib.sha256(archive_bytes).hexdigest())


if __name__ == "__main__":
    unittest.main()
