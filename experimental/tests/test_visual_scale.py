import sys
import unittest
from pathlib import Path

import numpy as np


_EXPERIMENTAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXPERIMENTAL_ROOT))

from e88_autopilot.reference_detection import ReferenceDetectionResult
from e88_autopilot.reference_store import ReferenceRecord
from e88_autopilot.visual_scale import VisualScaleEstimator, compute_altitude_est_m, compute_m_per_px


class TestVisualScaleMath(unittest.TestCase):
    def test_compute_m_per_px(self) -> None:
        mx, my = compute_m_per_px(pad_width_m=0.4, pad_height_m=0.3, ref_width_px=200.0, ref_height_px=150.0)
        self.assertAlmostEqual(float(mx), 0.002, places=9)
        self.assertAlmostEqual(float(my), 0.002, places=9)

    def test_compute_altitude_est_m(self) -> None:
        # For a pinhole-like model, ref_size_px scales ~ 1/z.
        # If the ref appears half as large, altitude doubles.
        z = compute_altitude_est_m(calibration_height_m=0.5, calibration_ref_size_px=200.0, ref_size_px=100.0)
        self.assertAlmostEqual(float(z), 1.0, places=6)


class _FakeDetector:
    def __init__(self, seq):
        self._seq = list(seq)
        self._i = 0

    def detect(self, _frame_bgr):
        if self._i >= len(self._seq):
            return self._seq[-1]
        v = self._seq[self._i]
        self._i += 1
        return v


class TestVisualScaleEstimator(unittest.TestCase):
    def test_accepts_detection_and_tracks_stability_and_last_known(self) -> None:
        rec = ReferenceRecord(
            reference_id="r1",
            created_at_ts=0.0,
            pad_type="pad",
            pad_width_m=0.4,
            pad_height_m=0.3,
            reference_capture_height_m=0.5,
            markers_present=False,
            image_filename="ref.png",
            calibration_height_m=0.5,
            calibration_ref_size_px=200.0,
        )

        ok = ReferenceDetectionResult(
            detected=True,
            mode="orb",
            quad_xy=np.zeros((4, 2), dtype=np.float32),
            ref_width_px=200.0,
            ref_height_px=150.0,
            ref_size_px=175.0,
            n_kp_frame=100,
            n_matches=80,
            n_inliers=60,
            inlier_ratio=0.75,
            reproj_error_px=1.0,
        )

        bad = ReferenceDetectionResult(
            detected=False,
            mode="orb",
            quad_xy=None,
            ref_width_px=0.0,
            ref_height_px=0.0,
            ref_size_px=0.0,
            n_kp_frame=0,
            n_matches=0,
            n_inliers=0,
            inlier_ratio=0.0,
            reproj_error_px=0.0,
        )

        est = VisualScaleEstimator(record=rec, reference_image_bgr=np.zeros((10, 10, 3), dtype=np.uint8), stable_required_frames=2)
        # Patch detector with a fake sequence.
        est._detector = _FakeDetector([ok, ok, bad])  # type: ignore[attr-defined]

        out1 = est.update(frame_bgr=np.zeros((10, 10, 3), dtype=np.uint8), timestamp=0.0, vx_px_s=10.0, vy_px_s=-5.0)
        self.assertTrue(out1.ref_detected)
        self.assertFalse(out1.stable)
        self.assertEqual(out1.altitude_source, "reference_object")
        self.assertIsNotNone(out1.vx_m_s)

        out2 = est.update(frame_bgr=np.zeros((10, 10, 3), dtype=np.uint8), timestamp=0.1, vx_px_s=10.0, vy_px_s=-5.0)
        self.assertTrue(out2.ref_detected)
        self.assertTrue(out2.stable)

        out3 = est.update(frame_bgr=np.zeros((10, 10, 3), dtype=np.uint8), timestamp=0.2, vx_px_s=10.0, vy_px_s=-5.0)
        self.assertFalse(out3.ref_detected)
        self.assertEqual(out3.altitude_source, "last_known")
        self.assertIsNotNone(out3.altitude_est_m)


if __name__ == "__main__":
    unittest.main()
