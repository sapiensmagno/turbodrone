import sys
import unittest
from pathlib import Path

import numpy as np


_EXPERIMENTAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXPERIMENTAL_ROOT))

from e88_autopilot.calibration import run_stationary_calibration
from e88_autopilot.optical_flow import FlowEstimate


class _FakeFrameSource:
    def __init__(self, frames):
        self._frames = list(frames)
        self._i = 0

    def get_frame_with_timestamp(self, timeout: float):
        if self._i >= len(self._frames):
            return None
        item = self._frames[self._i]
        self._i += 1
        return item


class TestStationaryCalibration(unittest.TestCase):
    def test_stationary_calibration_computes_deadband_and_sigma(self):
        # Provide enough frames with increasing timestamps.
        # We'll patch the drift estimator to return a deterministic velocity sample.
        frames = [
            (np.zeros((64, 64, 3), dtype=np.uint8), 0.01 * i)
            for i in range(60)
        ]
        src = _FakeFrameSource(frames)

        # Import inside the test so patching is isolated.
        import e88_autopilot.calibration as calib

        class _FakeEstimator:
            def __init__(self):
                self._n = 0

            def update(self, frame_bgr, timestamp=None):
                self._n += 1
                # First update returns None to match real behavior.
                if self._n == 1:
                    return None
                # Small stationary jitter around 0.
                vx = 0.2
                vy = -0.1
                return FlowEstimate(
                    dt_sec=0.01,
                    dx_px=vx * 0.01,
                    dy_px=vy * 0.01,
                    vx_px_s=vx,
                    vy_px_s=vy,
                    quality=1.0,
                    n_features=100,
                    n_tracked=100,
                    inlier_ratio=1.0,
                    fallback_used=False,
                )

        orig = calib.LucasKanadeDriftEstimator
        try:
            calib.LucasKanadeDriftEstimator = _FakeEstimator
            r = run_stationary_calibration(src, duration_sec=0.1, min_quality=0.0, percentile=99.0)
        finally:
            calib.LucasKanadeDriftEstimator = orig

        self.assertGreaterEqual(r.n_samples, 20)
        self.assertGreater(r.estimator_deadband_px_s, 0.0)
        self.assertGreater(r.kalman_sigma_v, 0.0)


if __name__ == "__main__":
    unittest.main()
