import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


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


if __name__ == "__main__":
    unittest.main()
