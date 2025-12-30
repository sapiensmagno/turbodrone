import sys
import unittest
from pathlib import Path

import numpy as np


_EXPERIMENTAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXPERIMENTAL_ROOT))

from e88_autopilot.kalman import VelocityKalman2D


class TestVelocityKalman2D(unittest.TestCase):
    def test_converges_on_constant_velocity(self):
        rng = np.random.default_rng(0)

        true_vx, true_vy = 12.0, -7.0
        sigma_meas = 8.0

        kf = VelocityKalman2D(sigma_a=1.0, sigma_v_meas=sigma_meas)

        t = 0.0
        for _ in range(120):
            dt = float(rng.uniform(0.07, 0.13))
            t += dt
            meas_vx = float(true_vx + rng.normal(0.0, sigma_meas))
            meas_vy = float(true_vy + rng.normal(0.0, sigma_meas))
            est = kf.update_velocity(t=t, vx_px_s=meas_vx, vy_px_s=meas_vy, quality=1.0)

        self.assertAlmostEqual(est.vx_px_s, true_vx, delta=2.0)
        self.assertAlmostEqual(est.vy_px_s, true_vy, delta=2.0)

    def test_low_quality_reduces_measurement_influence(self):
        kf = VelocityKalman2D(sigma_a=1.0, sigma_v_meas=5.0)

        kf.update_velocity(t=0.0, vx_px_s=0.0, vy_px_s=0.0, quality=1.0)
        est1 = kf.update_velocity(t=0.1, vx_px_s=50.0, vy_px_s=0.0, quality=1.0)

        kf.reset()
        kf.update_velocity(t=0.0, vx_px_s=0.0, vy_px_s=0.0, quality=1.0)
        est2 = kf.update_velocity(t=0.1, vx_px_s=50.0, vy_px_s=0.0, quality=0.05)

        self.assertGreater(abs(est1.vx_px_s), abs(est2.vx_px_s))


if __name__ == "__main__":
    unittest.main()
