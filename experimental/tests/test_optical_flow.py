import sys
import unittest
from pathlib import Path

import cv2
import numpy as np
from unittest import mock


_EXPERIMENTAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXPERIMENTAL_ROOT))

from e88_autopilot.optical_flow import LucasKanadeDriftEstimator


class TestLucasKanadeDriftEstimator(unittest.TestCase):
    def test_detects_known_translation(self):
        h, w = 240, 320
        base = np.zeros((h, w, 3), dtype=np.uint8)

        rng = np.random.default_rng(0)
        for _ in range(200):
            x = int(rng.integers(10, w - 10))
            y = int(rng.integers(10, h - 10))
            cv2.circle(base, (x, y), 2, (255, 255, 255), -1)

        dx, dy = 6.0, -4.0
        m = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        shifted = cv2.warpAffine(base, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        est = LucasKanadeDriftEstimator(downscale=1.0, reinit_every_n_frames=0, min_tracked_features=10)

        self.assertIsNone(est.update(base, timestamp=0.0))
        out = est.update(shifted, timestamp=0.1)
        self.assertIsNotNone(out)
        assert out is not None

        self.assertAlmostEqual(out.dx_px, dx, delta=2.0)
        self.assertAlmostEqual(out.dy_px, dy, delta=2.0)
        self.assertGreater(out.n_tracked, 10)

    def test_downscale_rescales_output_to_full_resolution(self):
        h, w = 240, 320
        base = np.zeros((h, w, 3), dtype=np.uint8)

        rng = np.random.default_rng(0)
        for _ in range(200):
            x = int(rng.integers(10, w - 10))
            y = int(rng.integers(10, h - 10))
            cv2.circle(base, (x, y), 2, (255, 255, 255), -1)

        dx, dy = 10.0, -6.0
        m = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        shifted = cv2.warpAffine(base, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        est = LucasKanadeDriftEstimator(downscale=0.5, reinit_every_n_frames=0, min_tracked_features=10)

        self.assertIsNone(est.update(base, timestamp=0.0))
        out = est.update(shifted, timestamp=0.1)
        self.assertIsNotNone(out)
        assert out is not None

        # dx/dy should be reported in full-resolution pixels even when tracking is downscaled.
        self.assertAlmostEqual(out.dx_px, dx, delta=3.0)
        self.assertAlmostEqual(out.dy_px, dy, delta=3.0)

    def test_phase_correlation_fallback_path(self):
        # Force LK to fail and ensure phase correlation provides the estimate.
        h, w = 120, 160
        base = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.circle(base, (50, 50), 5, (255, 255, 255), -1)

        shifted = base.copy()

        est = LucasKanadeDriftEstimator(downscale=1.0, reinit_every_n_frames=0, min_tracked_features=10)

        self.assertIsNone(est.update(base, timestamp=0.0))

        with mock.patch("cv2.calcOpticalFlowPyrLK", return_value=(None, None, None)):
            with mock.patch("cv2.phaseCorrelate", return_value=((3.0, -2.0), 0.9)):
                out = est.update(shifted, timestamp=0.1)

        self.assertIsNotNone(out)
        assert out is not None
        self.assertAlmostEqual(out.dx_px, 3.0, delta=1e-6)
        self.assertAlmostEqual(out.dy_px, -2.0, delta=1e-6)
        self.assertGreater(out.quality, 0.0)


if __name__ == "__main__":
    unittest.main()
