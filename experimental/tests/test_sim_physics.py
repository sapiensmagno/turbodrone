import math
import sys
import unittest
from pathlib import Path

import numpy as np

_EXPERIMENTAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXPERIMENTAL_ROOT))

from e88_sim.config import (
    GRAVITY_M_S2,
    AirframeParams,
    EnvironmentParams,
    SimConfig,
)
from e88_sim.physics import QuadPhysics, axis_to_byte, body_to_world_rotation, byte_to_axis


def _quiet_cfg(**airframe_overrides) -> SimConfig:
    """A drone with no disturbances, so a test measures one thing at a time."""
    af = dict(yaw_drift_deg_s=0.0, vibration_sigma_deg=0.0, altitude_hold_sigma_m=0.0)
    af.update(airframe_overrides)
    return SimConfig(
        airframe=AirframeParams(**af),
        environment=EnvironmentParams(trim_accel_x_m_s2=0.0, trim_accel_y_m_s2=0.0, gust_sigma_m_s2=0.0),
    )


class TestAttitudeAndAcceleration(unittest.TestCase):
    def test_steady_tilt_produces_g_tan_theta(self):
        """The defining physical relation of the model: a level-flight quad
        banked by theta accelerates at g*tan(theta). If this is wrong, every
        gain conclusion drawn from the simulator is wrong by the same factor."""
        cfg = _quiet_cfg(drag_linear_1_s=0.0, drag_quadratic_1_m=0.0)
        p = QuadPhysics(cfg)
        p.submit_command(roll=0.5, pitch=0.0, throttle=50.0)
        p.step(3.0)  # let the attitude loop settle

        s = p.state()
        v0 = s.vx_m_s
        p.step(0.1)
        measured_ax = (p.state().vx_m_s - v0) / 0.1

        expected_ax = GRAVITY_M_S2 * math.tan(s.roll_rad)
        self.assertAlmostEqual(measured_ax, expected_ax, delta=0.02)

    def test_commanded_angle_reached_within_max_tilt(self):
        cfg = _quiet_cfg()
        p = QuadPhysics(cfg)
        p.submit_command(roll=1.0, throttle=50.0)
        p.step(2.0)
        self.assertAlmostEqual(
            math.degrees(p.state().roll_rad), cfg.airframe.max_tilt_deg, delta=0.3
        )

    def test_attitude_settles_within_expected_time(self):
        """omega_n = 15 rad/s with zeta = 0.8 should be within 5% in ~0.4 s."""
        cfg = _quiet_cfg()
        p = QuadPhysics(cfg)
        target = math.radians(0.5 * cfg.airframe.max_tilt_deg)
        p.submit_command(roll=0.5, throttle=50.0)
        p.step(0.4)
        self.assertLess(abs(p.state().roll_rad - target), 0.05 * abs(target))

    def test_roll_moves_right_and_pitch_moves_forward(self):
        """Sign convention. A mistake here would silently invert every
        conclusion the simulator produces about roll_sign / pitch_sign."""
        for cmd, axis, sign in (("roll", "vx_m_s", 1.0), ("pitch", "vy_m_s", 1.0)):
            p = QuadPhysics(_quiet_cfg())
            p.submit_command(**{cmd: 0.5}, throttle=50.0)
            p.step(1.0)
            self.assertGreater(sign * getattr(p.state(), axis), 0.05, f"{cmd} drove the wrong axis/sign")

    def test_command_sign_flags_invert_the_response(self):
        cfg = _quiet_cfg(cmd_roll_to_right=-1.0)
        p = QuadPhysics(cfg)
        p.submit_command(roll=0.5, throttle=50.0)
        p.step(1.0)
        self.assertLess(p.state().vx_m_s, -0.05)

    def test_cross_axis_coupling_is_negligible_at_hover_tilts(self):
        p = QuadPhysics(_quiet_cfg())
        p.submit_command(roll=0.2, throttle=50.0)
        p.step(1.0)
        s = p.state()
        self.assertGreater(abs(s.vx_m_s), 0.1)
        self.assertLess(abs(s.vy_m_s), 0.02 * abs(s.vx_m_s))


class TestRotationMatrix(unittest.TestCase):
    def test_level_rotation_is_identity(self):
        np.testing.assert_allclose(body_to_world_rotation(0.0, 0.0, 0.0), np.eye(3), atol=1e-12)

    def test_positive_roll_tips_thrust_right(self):
        r = body_to_world_rotation(math.radians(10.0), 0.0, 0.0)
        thrust = r @ np.array([0.0, 0.0, 1.0])
        self.assertGreater(thrust[0], 0.0)
        self.assertAlmostEqual(thrust[1], 0.0, places=12)

    def test_positive_pitch_tips_thrust_forward(self):
        r = body_to_world_rotation(0.0, math.radians(10.0), 0.0)
        thrust = r @ np.array([0.0, 0.0, 1.0])
        self.assertGreater(thrust[1], 0.0)
        self.assertAlmostEqual(thrust[0], 0.0, places=12)

    def test_yaw_rotates_body_axes_counter_clockwise(self):
        r = body_to_world_rotation(0.0, 0.0, math.radians(90.0))
        body_forward_in_world = r @ np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(body_forward_in_world, [-1.0, 0.0, 0.0], atol=1e-9)

    def test_rotation_is_orthonormal(self):
        r = body_to_world_rotation(0.3, -0.2, 1.1)
        np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(r)), 1.0, places=12)


class TestCommandPath(unittest.TestCase):
    def test_quantization_matches_the_e88_protocol(self):
        """Commands are quantized to a byte before transmission, so the drone
        cannot act on a correction finer than 1/127. With kp_vx = 0.003 that is
        a 2.6 px/s dead zone -- the same order as the measured noise floor."""
        p = QuadPhysics(_quiet_cfg())
        p.submit_command(roll=0.0001, throttle=50.0)
        p.step(0.2)
        self.assertEqual(p.state().applied_roll, byte_to_axis(axis_to_byte(0.0001)))
        self.assertEqual(axis_to_byte(0.0001), 128)

    def test_quantization_can_be_disabled(self):
        p = QuadPhysics(_quiet_cfg(quantize_commands=False))
        p.submit_command(roll=0.0001, throttle=50.0)
        p.step(0.2)
        self.assertAlmostEqual(p.state().applied_roll, 0.0001)

    def test_command_is_not_applied_before_the_latency_elapses(self):
        cfg = _quiet_cfg(command_latency_sec=0.05)
        p = QuadPhysics(cfg)
        p.submit_command(roll=0.5, throttle=50.0)
        p.step(0.02)
        self.assertEqual(p.state().applied_roll, 0.0)
        p.step(0.10)
        self.assertGreater(p.state().applied_roll, 0.4)

    def test_commands_are_held_between_transmissions(self):
        """The RC state is re-sent at 33 Hz, so a command issued between
        transmissions does not reach the drone until the next one."""
        cfg = _quiet_cfg(control_interval_sec=0.03, command_latency_sec=0.0)
        p = QuadPhysics(cfg)
        p.step(0.001)
        p.submit_command(roll=0.5, throttle=50.0)
        p.step(0.01)
        self.assertEqual(p.state().applied_roll, 0.0, "command applied before the next transmit slot")
        p.step(0.03)
        self.assertGreater(p.state().applied_roll, 0.4)

    def test_stick_deadzone_suppresses_small_commands(self):
        p = QuadPhysics(_quiet_cfg(stick_deadzone_counts=5))
        p.submit_command(roll=0.02, throttle=50.0)  # ~2.5 counts
        p.step(0.2)
        self.assertEqual(p.state().applied_roll, 0.0)


class TestDragAndAltitude(unittest.TestCase):
    def test_drag_bounds_terminal_velocity(self):
        p = QuadPhysics(_quiet_cfg())
        p.submit_command(roll=1.0, throttle=50.0)
        p.step(30.0)
        speed = p.state().horizontal_speed_m_s
        self.assertGreater(speed, 1.0)
        self.assertLess(speed, 8.0, "terminal velocity is implausible for a light indoor quad")

    def test_neutral_throttle_holds_altitude(self):
        p = QuadPhysics(_quiet_cfg())
        p.submit_command(throttle=50.0)
        p.step(10.0)
        self.assertAlmostEqual(p.state().z_m, 0.7, delta=0.01)

    def test_throttle_above_neutral_climbs(self):
        p = QuadPhysics(_quiet_cfg())
        p.submit_command(throttle=80.0)
        p.step(2.0)
        self.assertGreater(p.state().z_m, 0.9)

    def test_takeoff_climbs_to_the_configured_altitude(self):
        cfg = _quiet_cfg()
        env = EnvironmentParams(
            trim_accel_x_m_s2=0.0, trim_accel_y_m_s2=0.0, gust_sigma_m_s2=0.0, initial_altitude_m=0.0
        )
        p = QuadPhysics(SimConfig(airframe=cfg.airframe, environment=env))
        p.request_takeoff()
        p.step(8.0)
        self.assertAlmostEqual(p.state().z_m, cfg.airframe.takeoff_altitude_m, delta=0.1)

    def test_land_returns_to_the_ground_and_stops(self):
        p = QuadPhysics(_quiet_cfg())
        p.request_land()
        p.step(10.0)
        s = p.state()
        self.assertLess(s.z_m, 0.02)
        self.assertTrue(s.landed)
        self.assertAlmostEqual(s.horizontal_speed_m_s, 0.0, places=6)

    def test_altitude_never_goes_below_ground(self):
        p = QuadPhysics(_quiet_cfg())
        p.submit_command(throttle=0.0)
        p.step(20.0)
        self.assertGreaterEqual(p.state().z_m, 0.0)


class TestDisturbances(unittest.TestCase):
    def test_trim_bias_drives_a_steady_drift(self):
        cfg = SimConfig(
            airframe=AirframeParams(yaw_drift_deg_s=0.0, vibration_sigma_deg=0.0, altitude_hold_sigma_m=0.0),
            environment=EnvironmentParams(
                trim_accel_x_m_s2=0.05, trim_accel_y_m_s2=0.0, gust_sigma_m_s2=0.0
            ),
        )
        p = QuadPhysics(cfg)
        p.step(30.0)
        # Terminal velocity where linear drag balances the bias: 0.05 / 0.6.
        self.assertAlmostEqual(p.state().vx_m_s, 0.05 / 0.6, delta=0.01)

    def test_turbulence_has_the_configured_standard_deviation(self):
        """The OU process is discretized exactly, so its steady-state spread
        must not depend on the integration step."""
        cfg = SimConfig(
            airframe=AirframeParams(yaw_drift_deg_s=0.0, vibration_sigma_deg=0.0, altitude_hold_sigma_m=0.0),
            environment=EnvironmentParams(
                trim_accel_x_m_s2=0.0, trim_accel_y_m_s2=0.0, gust_sigma_m_s2=0.2, gust_tau_sec=0.5
            ),
            seed=4,
        )
        p = QuadPhysics(cfg)
        p.step(5.0)  # burn-in
        samples = []
        for _ in range(4000):
            p.step(0.01)
            samples.append(p._gust[0])  # noqa: SLF001 - asserting on the noise process itself
        self.assertAlmostEqual(float(np.std(samples)), 0.2, delta=0.04)

    def test_gust_pulse_applies_only_inside_its_window(self):
        cfg = SimConfig(
            airframe=AirframeParams(yaw_drift_deg_s=0.0, vibration_sigma_deg=0.0, altitude_hold_sigma_m=0.0),
            environment=EnvironmentParams(
                trim_accel_x_m_s2=0.0,
                trim_accel_y_m_s2=0.0,
                gust_sigma_m_s2=0.0,
                gust_pulse_accel_m_s2=(1.0, 0.0),
                gust_pulse_start_sec=1.0,
                gust_pulse_duration_sec=0.5,
            ),
        )
        p = QuadPhysics(cfg)
        p.step(0.9)
        self.assertAlmostEqual(p.state().vx_m_s, 0.0, places=6)
        p.step(0.6)
        self.assertGreater(p.state().vx_m_s, 0.3)

    def test_vibration_shakes_the_camera_but_not_the_trajectory(self):
        """Vibration must reach the reported attitude (the camera rides on it)
        without contributing net acceleration -- a fast oscillation averages out
        in the dynamics but not in a per-frame image snapshot."""
        cfg = SimConfig(
            airframe=AirframeParams(
                yaw_drift_deg_s=0.0, altitude_hold_sigma_m=0.0, vibration_sigma_deg=0.5
            ),
            environment=EnvironmentParams(
                trim_accel_x_m_s2=0.0, trim_accel_y_m_s2=0.0, gust_sigma_m_s2=0.0
            ),
            seed=2,
        )
        p = QuadPhysics(cfg)
        attitudes = []
        for _ in range(2000):
            p.step(0.005)
            attitudes.append(p.state().roll_rad)

        self.assertGreater(float(np.std(attitudes)), math.radians(0.2))
        self.assertLess(abs(p.state().vx_m_s), 0.02, "vibration leaked into the trajectory")

    def test_seeded_runs_are_reproducible(self):
        def run(seed):
            p = QuadPhysics(SimConfig(seed=seed))
            p.submit_command(roll=0.1, throttle=50.0)
            p.step(5.0)
            return p.state()

        a, b, c = run(1), run(1), run(2)
        self.assertEqual(a.x_m, b.x_m)
        self.assertNotEqual(a.x_m, c.x_m)


class TestIntegrationAccuracy(unittest.TestCase):
    def test_result_is_insensitive_to_the_integration_step(self):
        """If halving the step changes the answer, the step is too big and the
        speed/accuracy tradeoff in the fast loop is unsafe."""
        results = []
        for dt in (0.004, 0.002, 0.001):
            cfg = _quiet_cfg()
            p = QuadPhysics(SimConfig(airframe=cfg.airframe, environment=cfg.environment, physics_dt_sec=dt))
            p.submit_command(roll=0.4, pitch=-0.2, throttle=50.0)
            p.step(4.0)
            results.append(p.state().x_m)

        self.assertAlmostEqual(results[0], results[-1], delta=0.005 * abs(results[-1]))


if __name__ == "__main__":
    unittest.main()
