import sys
import unittest
from pathlib import Path


_EXPERIMENTAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXPERIMENTAL_ROOT))

from e88_autopilot.controller import PositionLeashController
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


class TestPositionLeashController(unittest.TestCase):
    def test_commands_when_position_error_exists_even_if_velocity_is_zero(self):
        inner = VelocityHoldController(
            min_quality=0.0,
            kp_vx=1.0,
            kp_vy=1.0,
            ki_vx=0.0,
            ki_vy=0.0,
            deadband_px_s=0.0,
            max_cmd=10.0,
            roll_sign=1.0,
            pitch_sign=1.0,
        )
        leash = PositionLeashController(
            inner=inner,
            kp_pos=0.5,
            ki_pos=0.0,
            pos_deadband=0.0,
            max_v_sp=100.0,
            pos_integrator_limit=1000.0,
            min_quality=0.0,
        )

        for _ in range(10):
            out = leash.update(dt_sec=0.1, vx_px_s=1.0, vy_px_s=0.0, quality=1.0)
            self.assertIsNotNone(out)

        out = leash.update(dt_sec=0.1, vx_px_s=0.0, vy_px_s=0.0, quality=1.0)
        self.assertIsNotNone(out)
        assert out is not None
        self.assertGreater(abs(out.roll), 1e-6)

    def test_can_use_external_position_estimate(self):
        inner = VelocityHoldController(
            min_quality=0.0,
            kp_vx=1.0,
            kp_vy=1.0,
            ki_vx=0.0,
            ki_vy=0.0,
            deadband_px_s=0.0,
            max_cmd=10.0,
            roll_sign=1.0,
            pitch_sign=1.0,
        )
        leash = PositionLeashController(
            inner=inner,
            kp_pos=0.5,
            ki_pos=0.0,
            pos_deadband=0.0,
            max_v_sp=100.0,
            pos_integrator_limit=1000.0,
            min_quality=0.0,
        )

        out = leash.update(dt_sec=0.1, vx_px_s=0.0, vy_px_s=0.0, pos_x=10.0, pos_y=0.0, quality=1.0)
        self.assertIsNotNone(out)
        assert out is not None
        self.assertGreater(abs(out.roll), 1e-6)

    def test_position_converges_toward_zero_under_perfect_velocity_tracking(self):
        inner = VelocityHoldController(
            min_quality=0.0,
            kp_vx=0.0,
            kp_vy=0.0,
            ki_vx=0.0,
            ki_vy=0.0,
            deadband_px_s=0.0,
            max_cmd=10.0,
        )
        leash = PositionLeashController(
            inner=inner,
            kp_pos=0.5,
            ki_pos=0.0,
            pos_deadband=0.0,
            max_v_sp=100.0,
            pos_integrator_limit=1000.0,
            min_quality=0.0,
        )

        leash.update(dt_sec=1.0, vx_px_s=1.0, vy_px_s=0.0, quality=1.0)
        start_pos_x = leash.state().pos_x
        self.assertGreater(start_pos_x, 0.5)

        vx = 0.0
        for _ in range(50):
            leash.update(dt_sec=0.1, vx_px_s=vx, vy_px_s=0.0, quality=1.0)
            vx = leash.state().v_sp_x

        end_pos_x = leash.state().pos_x
        self.assertLess(abs(end_pos_x), abs(start_pos_x))


if __name__ == "__main__":
    unittest.main()
