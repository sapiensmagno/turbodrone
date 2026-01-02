import sys
import time
import unittest
from pathlib import Path
from queue import Empty, Queue

import numpy as np


_EXPERIMENTAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXPERIMENTAL_ROOT))

from e88_autopilot.autostabilizer import AutoStabilizer, StabilizerConfig
from e88_autopilot.optical_flow import FlowEstimate


class _FakeDrone:
    def __init__(self) -> None:
        self._q: "Queue[tuple[np.ndarray, float]]" = Queue()
        self.sent = []

    def push(self, frame: np.ndarray, ts: float) -> None:
        self._q.put((frame, float(ts)))

    def get_frame_with_timestamp(self, timeout: float):
        try:
            return self._q.get(timeout=float(timeout))
        except Empty:
            return None

    def send_cmd(self, *, roll: float, pitch: float, throttle: float, yaw: float = 0.0):
        self.sent.append((float(roll), float(pitch), float(throttle), float(yaw)))


class TestAutoStabilizerRuntime(unittest.TestCase):
    def test_run_does_not_crash_on_update_rate_ema(self) -> None:
        # This reproduces the prior regression where AutoStabilizer referenced
        # self._update_rate_ema but the method did not exist.
        drone = _FakeDrone()

        cfg = StabilizerConfig(
            enable_takeoff=False,
            cmd_rate_hz=60.0,
            use_kalman=False,
            enable_visual_scale=False,
            min_quality=0.0,
        )

        s = AutoStabilizer(drone, cfg=cfg)

        # Build a frame with trackable features so optical flow produces an estimate
        # (and hits the _update_rate_ema code path).
        rng = np.random.default_rng(0)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        for _ in range(250):
            x = int(rng.integers(10, 310))
            y = int(rng.integers(10, 230))
            frame[y - 1 : y + 1, x - 1 : x + 1] = 255

        t0 = float(time.monotonic())
        drone.push(frame.copy(), t0)
        drone.push(frame.copy(), t0 + 0.05)

        s.activate()

        # Should complete without AttributeError.
        s.run(duration_sec=0.15)

        self.assertGreaterEqual(len(drone.sent), 1)

    def test_visual_scale_exceptions_are_reported_in_telemetry(self) -> None:
        class _FakeFlow:
            def reset(self):
                return None

            def last_tracks(self):
                return None

            def update(self, _frame_bgr, timestamp=None):
                return FlowEstimate(
                    dt_sec=0.05,
                    dx_px=0.0,
                    dy_px=0.0,
                    vx_px_s=0.0,
                    vy_px_s=0.0,
                    quality=1.0,
                    n_features=100,
                    n_tracked=100,
                    inlier_ratio=1.0,
                    fallback_used=False,
                )

        class _RaisingVS:
            def update(self, *args, **kwargs):
                raise RuntimeError("boom")

        drone = _FakeDrone()

        cfg = StabilizerConfig(
            enable_takeoff=False,
            cmd_rate_hz=60.0,
            use_kalman=False,
            enable_visual_scale=True,
            min_quality=0.0,
        )

        latest = []

        def sink(t):
            latest.append(t)

        s = AutoStabilizer(drone, cfg=cfg, telemetry_sink=sink)
        s._flow = _FakeFlow()  # type: ignore[attr-defined]

        rng = np.random.default_rng(0)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        for _ in range(250):
            x = int(rng.integers(10, 310))
            y = int(rng.integers(10, 230))
            frame[y - 1 : y + 1, x - 1 : x + 1] = 255

        t0 = float(time.monotonic())
        drone.push(frame.copy(), t0)
        drone.push(frame.copy(), t0 + 0.05)

        s.activate()
        # Force visual scale object to exist without depending on the reference store.
        s._visual_scale = _RaisingVS()  # type: ignore[attr-defined]
        s._visual_scale_error = ""  # type: ignore[attr-defined]

        s.run(duration_sec=0.15)

        self.assertGreaterEqual(len(latest), 1)
        any_err = any("boom" in str(getattr(t, "visual_scale_error", "")) for t in latest)
        self.assertTrue(any_err)


if __name__ == "__main__":
    unittest.main()
