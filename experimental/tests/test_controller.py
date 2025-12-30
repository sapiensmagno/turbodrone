import sys
import unittest
from pathlib import Path


_EXPERIMENTAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXPERIMENTAL_ROOT))

from e88_autopilot.controller import VelocityHoldController


class TestVelocityHoldController(unittest.TestCase):
    def test_returns_none_below_quality_threshold(self):
        c = VelocityHoldController(min_quality=0.5)
        out = c.update(dt_sec=0.1, vx_px_s=20.0, vy_px_s=0.0, quality=0.1)
        self.assertIsNone(out)

    def test_deadband_zeroes_small_velocities(self):
        c = VelocityHoldController(deadband_px_s=5.0, min_quality=0.0)
        out = c.update(dt_sec=0.1, vx_px_s=4.9, vy_px_s=-4.9, quality=1.0)
        self.assertIsNotNone(out)
        assert out is not None
        self.assertAlmostEqual(out.roll, 0.0, delta=1e-6)
        self.assertAlmostEqual(out.pitch, 0.0, delta=1e-6)

    def test_output_clamps(self):
        c = VelocityHoldController(max_cmd=0.1, min_quality=0.0, kp_vx=1.0, kp_vy=1.0, ki_vx=0.0, ki_vy=0.0)
        out = c.update(dt_sec=0.1, vx_px_s=1000.0, vy_px_s=-1000.0, quality=1.0)
        self.assertIsNotNone(out)
        assert out is not None
        self.assertLessEqual(abs(out.roll), 0.1 + 1e-9)
        self.assertLessEqual(abs(out.pitch), 0.1 + 1e-9)


if __name__ == "__main__":
    unittest.main()
