"""``SimulatedDrone`` must be usable wherever ``turbodrone.Drone`` is.

If the interface diverges, the vision-in-the-loop simulation stops exercising
the real autopilot and starts exercising an adapter, which is exactly the sort
of gap that lets a sign error survive to a real flight.
"""

import inspect
import sys
import time
import unittest
from pathlib import Path

import numpy as np

_EXPERIMENTAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXPERIMENTAL_ROOT))

from e88_sim.config import AirframeParams, EnvironmentParams, GroundParams, SimConfig, VideoLinkParams
from e88_sim.sim_drone import SimulatedDrone
from turbodrone import Drone

# A small ground keeps these tests quick; they are about plumbing, not texture.
_FAST_GROUND = GroundParams(extent_m=2.0, m_per_px=0.004, relief_fraction=0.0)


def _cfg(**kwargs) -> SimConfig:
    kwargs.setdefault("ground", _FAST_GROUND)
    kwargs.setdefault("seed", 1)
    return SimConfig(**kwargs)


class TestDroneInterfaceCompatibility(unittest.TestCase):
    def test_implements_every_method_the_autopilot_calls(self):
        for name in ("connect", "close", "get_frame", "get_frame_with_timestamp", "send_cmd", "takeoff", "land"):
            self.assertTrue(callable(getattr(SimulatedDrone, name, None)), f"missing {name}")

    def test_send_cmd_signature_matches_the_real_drone(self):
        real = inspect.signature(Drone.send_cmd)
        sim = inspect.signature(SimulatedDrone.send_cmd)
        self.assertEqual(list(real.parameters), list(sim.parameters))

    def test_get_frame_with_timestamp_signature_matches(self):
        real = inspect.signature(Drone.get_frame_with_timestamp)
        sim = inspect.signature(SimulatedDrone.get_frame_with_timestamp)
        self.assertEqual(list(real.parameters), list(sim.parameters))


class TestFrameDelivery(unittest.TestCase):
    def test_delivers_bgr_frames_of_the_configured_size(self):
        drone = SimulatedDrone(_cfg())
        drone.connect()
        try:
            item = drone.get_frame_with_timestamp(timeout=2.0)
            self.assertIsNotNone(item, "no frame delivered")
            frame, ts = item
            self.assertEqual(frame.shape, (480, 640, 3))
            self.assertEqual(frame.dtype, np.uint8)
            self.assertIsInstance(ts, float)
        finally:
            drone.close()

    def test_delivers_at_roughly_the_configured_frame_rate(self):
        drone = SimulatedDrone(_cfg(video=VideoLinkParams(drop_probability=0.0, tear_probability=0.0)))
        drone.connect()
        try:
            time.sleep(0.4)  # let the latency pipeline fill
            count = 0
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                if drone.get_frame_with_timestamp(timeout=0.3) is not None:
                    count += 1
            self.assertGreater(count, 25, f"only {count} frames in 2 s, expected ~40")
            self.assertLess(count, 55)
        finally:
            drone.close()

    def test_get_frame_returns_none_on_timeout_when_the_feed_is_frozen(self):
        drone = SimulatedDrone(_cfg(video=VideoLinkParams(freeze_start_sec=0.0, freeze_duration_sec=30.0)))
        drone.connect()
        try:
            self.assertIsNone(drone.get_frame_with_timestamp(timeout=0.3))
        finally:
            drone.close()

    def test_close_is_idempotent_and_stops_the_threads(self):
        drone = SimulatedDrone(_cfg())
        drone.connect()
        drone.close()
        drone.close()
        self.assertIsNone(drone.get_frame_with_timestamp(timeout=0.1))


class TestCommandsAndTruth(unittest.TestCase):
    def test_send_cmd_moves_the_drone_and_is_recorded(self):
        drone = SimulatedDrone(
            _cfg(
                airframe=AirframeParams(yaw_drift_deg_s=0.0),
                environment=EnvironmentParams(trim_accel_x_m_s2=0.0, trim_accel_y_m_s2=0.0, gust_sigma_m_s2=0.0),
            )
        )
        drone.connect()
        try:
            deadline = time.monotonic() + 1.5
            while time.monotonic() < deadline:
                drone.send_cmd(roll=0.5, pitch=0.0, throttle=50.0)
                time.sleep(0.05)
            state = drone.state()
            self.assertGreater(state.vx_m_s, 0.1, "positive roll should accelerate right")
            self.assertTrue(drone.truth)
            self.assertAlmostEqual(drone.truth[-1].cmd_roll, 0.5)
        finally:
            drone.close()

    def test_takeoff_climbs(self):
        drone = SimulatedDrone(_cfg(environment=EnvironmentParams(initial_altitude_m=0.0)))
        drone.connect()
        try:
            drone.takeoff()
            time.sleep(2.0)
            self.assertGreater(drone.state().z_m, 0.3)
        finally:
            drone.close()

    def test_land_descends(self):
        drone = SimulatedDrone(_cfg(environment=EnvironmentParams(initial_altitude_m=0.6)))
        drone.connect()
        try:
            drone.land()
            time.sleep(3.0)
            self.assertLess(drone.state().z_m, 0.2)
        finally:
            drone.close()

    def test_stats_report_link_health(self):
        drone = SimulatedDrone(_cfg(video=VideoLinkParams(drop_probability=0.5, tear_probability=0.5)))
        drone.connect()
        try:
            time.sleep(1.5)
            stats = drone.stats
            self.assertGreater(stats["frames_rendered"], 0)
            self.assertGreater(stats["frames_dropped_link"], 0)
        finally:
            drone.close()


if __name__ == "__main__":
    unittest.main()
