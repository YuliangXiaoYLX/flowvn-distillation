import unittest

import numpy as np

from main import UnrolledNetwork


class _GroupEvaluator:
    _evaluate_val_group = UnrolledNetwork._evaluate_val_group

    @staticmethod
    def _maybe_apply_phase_correction(pred, gt):
        return pred, gt


class VelocityGroupIntegrationTests(unittest.TestCase):
    def test_group_evaluation_uses_reference_relative_wrapped_phase(self):
        pred_phase = np.zeros((4, 1, 1, 1, 1), dtype=np.float64)
        gt_phase = np.zeros_like(pred_phase)
        pred_phase[0] = 0.2
        gt_phase[0] = 0.4
        pred_phase[1] = 0.2 + np.pi / 2.0
        pred_phase[2] = 0.2 - np.pi / 3.0
        pred_phase[3] = 0.2
        gt_phase[1:] = 0.4
        pred = np.exp(1j * pred_phase)
        gt = np.exp(1j * gt_phase)

        metrics, visualization = _GroupEvaluator()._evaluate_val_group(
            key=("/synthetic/site/case_a", 0, 10),
            pred_np=pred,
            gt_np=gt,
            seg_np=np.ones((1, 1, 1), dtype=bool),
            venc=np.array([200.0, 150.0, 150.0]),
            want_visualization=False,
        )

        self.assertIsNone(visualization)
        self.assertAlmostEqual(
            metrics["velocity_vector_rmse_cm_s"],
            np.sqrt(100.0**2 + 50.0**2),
            places=10,
        )

    def test_group_evaluation_leaves_physical_metric_missing_without_venc(self):
        image = np.ones((4, 1, 1, 1, 1), dtype=np.complex64)

        metrics, _ = _GroupEvaluator()._evaluate_val_group(
            key=("/synthetic/site/case_a", 0, 10),
            pred_np=image,
            gt_np=image,
            seg_np=np.ones((1, 1, 1), dtype=bool),
            venc=None,
            want_visualization=False,
        )

        self.assertIsNone(metrics["velocity_vector_rmse_cm_s"])


if __name__ == "__main__":
    unittest.main()
