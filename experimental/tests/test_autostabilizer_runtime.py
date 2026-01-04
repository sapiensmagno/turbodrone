import sys
import time
import threading
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

    def test_visual_scale_ms_hold_hysteresis(self) -> None:
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
                    vx_px_s=1.0,
                    vy_px_s=2.0,
                    quality=1.0,
                    n_features=100,
                    n_tracked=100,
                    inlier_ratio=1.0,
                    fallback_used=False,
                )

        class _Det:
            def __init__(self) -> None:
                self.quad_xy = None
                self.mode = "test"
                self.n_matches = 0
                self.n_inliers = 0
                self.inlier_ratio = 0.0
                self.reproj_error_px = 0.0

        class _VSStable:
            def __init__(self) -> None:
                self._i = 0

            def update(self, *, frame_bgr, timestamp: float, vx_px_s: float, vy_px_s: float):
                self._i += 1

                # 1st call: stable with known m/px.
                if self._i == 1:
                    m_per_px = 0.01
                    return type(
                        "_Scale",
                        (),
                        {
                            "altitude_est_m": 1.0,
                            "altitude_source": "test",
                            "ref_detected": True,
                            "ref_width_px": 100.0,
                            "ref_height_px": 100.0,
                            "ref_size_px": 100.0,
                            "vx_m_s": float(vx_px_s) * m_per_px,
                            "vy_m_s": float(vy_px_s) * m_per_px,
                            "m_per_px_x": m_per_px,
                            "m_per_px_y": m_per_px,
                            "stable": True,
                            "detection": _Det(),
                        },
                    )()

                # Later calls: unstable/no m/s info.
                return type(
                    "_Scale",
                    (),
                    {
                        "altitude_est_m": 1.0,
                        "altitude_source": "test",
                        "ref_detected": False,
                        "ref_width_px": 0.0,
                        "ref_height_px": 0.0,
                        "ref_size_px": 0.0,
                        "vx_m_s": None,
                        "vy_m_s": None,
                        "m_per_px_x": None,
                        "m_per_px_y": None,
                        "stable": False,
                        "detection": _Det(),
                    },
                )()

        drone = _FakeDrone()
        cfg = StabilizerConfig(
            enable_takeoff=False,
            cmd_rate_hz=60.0,
            use_kalman=False,
            enable_visual_scale=True,
            use_m_s_control=True,
            visual_scale_ms_hold_sec=1.0,
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

        pushed = {}

        def _push_frames() -> None:
            t0 = float(time.monotonic())
            pushed["t0"] = t0
            drone.push(frame.copy(), t0)
            time.sleep(0.20)
            t1 = float(time.monotonic())
            pushed["t1"] = t1
            drone.push(frame.copy(), t1)
            time.sleep(1.20)
            t2 = float(time.monotonic())
            pushed["t2"] = t2
            drone.push(frame.copy(), t2)

        s.activate()
        # Force visual scale object to exist without depending on the reference store.
        s._visual_scale = _VSStable()  # type: ignore[attr-defined]
        s._visual_scale_error = ""  # type: ignore[attr-defined]

        th = threading.Thread(target=_push_frames)
        th.daemon = True
        th.start()
        s.run(duration_sec=1.8)
        th.join(timeout=2.0)

        # Find samples closest to our injected timestamps.
        t1 = float(pushed.get("t1", 0.0))
        t2 = float(pushed.get("t2", 0.0))
        self.assertGreater(t1, 0.0)
        self.assertGreater(t2, 0.0)

        by_ts = sorted(latest, key=lambda t: float(t.timestamp))
        t_hold = min(by_ts, key=lambda t: abs(float(t.timestamp) - t1))
        t_after = min(by_ts, key=lambda t: abs(float(t.timestamp) - t2))

        # Within the 1.0s hold window after the last stable detection, we should still report m/s usage.
        self.assertFalse(bool(getattr(t_hold, "scale_stable", False)))
        self.assertIsNotNone(getattr(t_hold, "used_vx_m_s", None))

        # After the hold window expires, it should fall back to px/s and stop reporting used m/s.
        self.assertFalse(bool(getattr(t_after, "scale_stable", False)))
        self.assertIsNone(getattr(t_after, "used_vx_m_s", None))


if __name__ == "__main__":
    unittest.main()
