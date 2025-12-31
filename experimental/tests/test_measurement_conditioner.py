import sys
import unittest
from pathlib import Path


_EXPERIMENTAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXPERIMENTAL_ROOT))

from e88_autopilot.measurement_conditioner import VelocityMeasurementConditioner


class TestVelocityMeasurementConditioner(unittest.TestCase):
    def test_deadband_gates_small_velocities(self):
        c = VelocityMeasurementConditioner(deadband_px_s=1.0)
        out = c.apply(vx_px_s=0.5, vy_px_s=-0.2)
        self.assertEqual(out.vx_px_s, 0.0)
        self.assertEqual(out.vy_px_s, 0.0)
        self.assertTrue(out.gated)

    def test_deadband_passes_large_velocities(self):
        c = VelocityMeasurementConditioner(deadband_px_s=1.0)
        out = c.apply(vx_px_s=1.5, vy_px_s=-2.0)
        self.assertEqual(out.vx_px_s, 1.5)
        self.assertEqual(out.vy_px_s, -2.0)
        self.assertFalse(out.gated)


if __name__ == "__main__":
    unittest.main()
