import unittest

import numpy as np

from utils.utils_metrics import VelocityVectorRMSE


class VelocityVectorRMSETests(unittest.TestCase):
    def test_identical_wrapped_phase_fields_have_zero_error(self):
        phase = np.linspace(
            -np.pi,
            np.pi,
            num=3 * 2 * 2 * 2 * 2,
            endpoint=False,
            dtype=np.float64,
        ).reshape(3, 2, 2, 2, 2)
        roi = np.ones((2, 2, 2), dtype=bool)

        value = VelocityVectorRMSE(
            phase,
            phase.copy(),
            venc_cm_s=np.array([200.0, 150.0, 150.0]),
            segmask=roi,
        )

        self.assertEqual(value, 0.0)

    def test_known_directional_phase_errors_use_each_venc(self):
        pred = np.zeros((3, 2, 1, 1, 1), dtype=np.float64)
        pred[0] = np.pi / 2.0
        pred[1] = -np.pi / 3.0
        gt = np.zeros_like(pred)

        value = VelocityVectorRMSE(
            pred,
            gt,
            venc_cm_s=np.array([200.0, 150.0, 150.0]),
            segmask=np.ones((1, 1, 1), dtype=bool),
        )

        self.assertAlmostEqual(value, np.sqrt(100.0**2 + 50.0**2), places=10)

    def test_error_is_circular_across_the_phase_wrap_boundary(self):
        offset = 0.01
        pred = np.zeros((3, 1, 1, 1, 1), dtype=np.float64)
        gt = np.zeros_like(pred)
        pred[0] = np.pi - offset
        gt[0] = -np.pi + offset

        value = VelocityVectorRMSE(
            pred,
            gt,
            venc_cm_s=np.array([150.0, 150.0, 150.0]),
            segmask=np.ones((1, 1, 1), dtype=bool),
        )

        self.assertAlmostEqual(value, 2.0 * offset * 150.0 / np.pi, places=10)

    def test_scalar_venc_is_broadcast_to_all_directions(self):
        pred = np.full((3, 1, 1, 1, 1), np.pi / 3.0, dtype=np.float64)
        gt = np.zeros_like(pred)

        value = VelocityVectorRMSE(
            pred,
            gt,
            venc_cm_s=150.0,
            segmask=np.ones((1, 1, 1), dtype=bool),
        )

        self.assertAlmostEqual(value, np.sqrt(3.0 * 50.0**2), places=10)

    def test_nonfinite_phase_or_venc_is_rejected(self):
        phase = np.zeros((3, 1, 1, 1, 1), dtype=np.float64)
        roi = np.ones((1, 1, 1), dtype=bool)

        with self.subTest(field="phase"):
            invalid_phase = phase.copy()
            invalid_phase[0, 0, 0, 0, 0] = np.nan
            with self.assertRaisesRegex(ValueError, "finite"):
                VelocityVectorRMSE(
                    invalid_phase,
                    phase,
                    venc_cm_s=[150.0, 150.0, 150.0],
                    segmask=roi,
                )

        with self.subTest(field="venc"):
            with self.assertRaisesRegex(ValueError, "finite"):
                VelocityVectorRMSE(
                    phase,
                    phase,
                    venc_cm_s=[150.0, np.inf, 150.0],
                    segmask=roi,
                )

    def test_nonpositive_venc_is_rejected(self):
        phase = np.zeros((3, 1, 1, 1, 1), dtype=np.float64)

        with self.assertRaisesRegex(ValueError, "positive"):
            VelocityVectorRMSE(
                phase,
                phase,
                venc_cm_s=[150.0, 0.0, 150.0],
                segmask=np.ones((1, 1, 1), dtype=bool),
            )

    def test_phase_fields_require_direction_time_and_three_spatial_axes(self):
        phase_without_time = np.zeros((3, 2, 2, 2), dtype=np.float64)

        with self.assertRaisesRegex(ValueError, "direction,time,SPE,PE,FE"):
            VelocityVectorRMSE(
                phase_without_time,
                phase_without_time,
                venc_cm_s=[150.0, 150.0, 150.0],
                segmask=np.ones((2, 2, 2), dtype=bool),
            )

    def test_segmentation_must_match_the_three_spatial_axes(self):
        phase = np.zeros((3, 1, 2, 2, 2), dtype=np.float64)

        with self.assertRaisesRegex(ValueError, "segmask shape"):
            VelocityVectorRMSE(
                phase,
                phase,
                venc_cm_s=[150.0, 150.0, 150.0],
                segmask=np.ones((2, 2), dtype=bool),
            )

    def test_roi_is_averaged_over_included_voxels_and_all_times(self):
        pred = np.zeros((3, 2, 1, 1, 2), dtype=np.float64)
        gt = np.zeros_like(pred)
        pred[0, 0, 0, 0, 0] = 3.0 * np.pi / 100.0
        pred[1, 0, 0, 0, 0] = 4.0 * np.pi / 100.0
        pred[:, :, 0, 0, 1] = np.pi / 2.0
        roi = np.array([[[True, False]]])

        value = VelocityVectorRMSE(
            pred,
            gt,
            venc_cm_s=100.0,
            segmask=roi,
        )

        self.assertAlmostEqual(value, np.sqrt(25.0 / 2.0), places=10)

    def test_empty_roi_is_rejected(self):
        phase = np.zeros((3, 1, 1, 1, 1), dtype=np.float64)

        with self.assertRaisesRegex(ValueError, "no True"):
            VelocityVectorRMSE(
                phase,
                phase,
                venc_cm_s=150.0,
                segmask=np.zeros((1, 1, 1), dtype=bool),
            )


if __name__ == "__main__":
    unittest.main()
