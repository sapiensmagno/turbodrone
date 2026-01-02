import sys
import time
import unittest
from pathlib import Path
from queue import Empty, Queue

import numpy as np


_EXPERIMENTAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXPERIMENTAL_ROOT))

from e88_autopilot.autostabilizer import AutoStabilizer, StabilizerConfig


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


if __name__ == "__main__":
    unittest.main()
