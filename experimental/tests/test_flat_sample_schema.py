import sys
import unittest
from pathlib import Path


_EXPERIMENTAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXPERIMENTAL_ROOT))

from e88_autopilot.autostabilizer import StabilizerTelemetry
from e88_autopilot.kalman import KalmanEstimate
from e88_autopilot.optical_flow import FlowEstimate
from e88_autopilot.session_recorder import telemetry_to_flat_sample


class TestFlatSampleSchema(unittest.TestCase):
    def test_schema_is_flat_and_key_stable(self) -> None:
        t0 = StabilizerTelemetry(
            phase="settle",
            frame_bgr=None,
            timestamp=1.0,
            pos_x_px=0.0,
            pos_y_px=0.0,
            flow=None,
            tracks=None,
            kalman=None,
            used_vx_px_s=0.0,
            used_vy_px_s=0.0,
            cmd_roll=0.0,
            cmd_pitch=0.0,
            cmd_throttle=50.0,
        )

        s0 = telemetry_to_flat_sample(t0)
        self.assertNotIn("flow", s0)
        self.assertNotIn("kalman", s0)
        self.assertIn("flow_valid", s0)
        self.assertIn("kalman_valid", s0)

        flow = FlowEstimate(
            dt_sec=0.1,
            dx_px=1.0,
            dy_px=2.0,
            vx_px_s=10.0,
            vy_px_s=20.0,
            quality=0.5,
            n_features=100,
            n_tracked=80,
            inlier_ratio=0.75,
            fallback_used=False,
        )
        kal = KalmanEstimate(t=1.0, x_px=0.0, y_px=0.0, vx_px_s=1.0, vy_px_s=2.0, p_vx=3.0, p_vy=4.0)

        t1 = StabilizerTelemetry(
            phase="hold",
            frame_bgr=None,
            timestamp=2.0,
            pos_x_px=1.0,
            pos_y_px=2.0,
            flow=flow,
            tracks=None,
            kalman=kal,
            used_vx_px_s=10.0,
            used_vy_px_s=20.0,
            cmd_roll=0.1,
            cmd_pitch=-0.2,
            cmd_throttle=50.0,
        )

        s1 = telemetry_to_flat_sample(t1)
        self.assertNotIn("flow", s1)
        self.assertNotIn("kalman", s1)

        self.assertEqual(set(s0.keys()), set(s1.keys()))

        # When flow is missing, numeric flow fields are None (except validity + bool/string fields).
        self.assertFalse(bool(s0["flow_valid"]))
        self.assertIsNone(s0["flow_dx_px"])
        self.assertIsNone(s0["flow_quality"])
        self.assertFalse(bool(s0["flow_fallback_used"]))

        # When flow is present, values are populated.
        self.assertTrue(bool(s1["flow_valid"]))
        self.assertAlmostEqual(float(s1["flow_dx_px"]), 1.0, places=6)
        self.assertAlmostEqual(float(s1["flow_quality"]), 0.5, places=6)

        # Kalman behaves similarly.
        self.assertFalse(bool(s0["kalman_valid"]))
        self.assertIsNone(s0["kalman_x_px"])
        self.assertTrue(bool(s1["kalman_valid"]))
        self.assertAlmostEqual(float(s1["kalman_x_px"]), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
