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


class _PassthroughFlow:
    """Flow estimator stub that returns a fixed estimate once it has seen two
    frames, deriving dt from whatever timestamp it is handed."""

    def __init__(self, *, vx_px_s: float = 10.0, vy_px_s: float = 0.0) -> None:
        self._last_t = None
        self.calls = 0
        self._vx = float(vx_px_s)
        self._vy = float(vy_px_s)

    def reset(self):
        self._last_t = None

    def last_tracks(self):
        return None

    def update(self, _frame_bgr, timestamp=None):
        self.calls += 1
        t = float(0.0 if timestamp is None else timestamp)
        if self._last_t is None:
            self._last_t = t
            return None
        dt = t - float(self._last_t)
        self._last_t = t
        if dt <= 1e-6:
            return None
        return FlowEstimate(
            dt_sec=float(dt),
            dx_px=float(self._vx) * dt,
            dy_px=float(self._vy) * dt,
            vx_px_s=float(self._vx),
            vy_px_s=float(self._vy),
            quality=1.0,
            n_features=100,
            n_tracked=100,
            inlier_ratio=1.0,
            fallback_used=False,
        )


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

    def test_hold_flow_receives_monotonic_time_not_frame_timestamp(self) -> None:
        """The hold loop must derive flow dt from a monotonic clock, not from the
        frame's source timestamp (regression guard for commit 2e49441).

        This used to be tested indirectly, by pushing duplicate frames and relying
        on the loop reprocessing them. Stale frames are now skipped outright
        (skip_stale_frames), so the timestamp contract is asserted directly."""

        seen_timestamps = []

        class _RecordingFlow(_PassthroughFlow):
            def update(self, _frame_bgr, timestamp=None):
                seen_timestamps.append(None if timestamp is None else float(timestamp))
                return super().update(_frame_bgr, timestamp=timestamp)

        drone = _FakeDrone()
        cfg = StabilizerConfig(
            enable_takeoff=False,
            cmd_rate_hz=60.0,
            use_kalman=False,
            enable_visual_scale=False,
            min_quality=0.0,
        )

        s = AutoStabilizer(drone, cfg=cfg)
        s._flow = _RecordingFlow()  # type: ignore[attr-defined]

        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        # Source timestamps far from any plausible monotonic clock value.
        drone.push(frame.copy(), 1000.0)
        drone.push(frame.copy(), 1000.1)

        s.activate()
        s.run(duration_sec=0.25)

        self.assertTrue(seen_timestamps)
        for t in seen_timestamps:
            self.assertIsNotNone(t)
            self.assertNotIn(t, (1000.0, 1000.1))

    @staticmethod
    def _paced_pusher(drone, *, n: int, interval_sec: float):
        """Feed frames at a fixed cadence, slower than the control loop, so the
        loop genuinely outruns the camera. The size-1 LatestFrameBuffer drops
        frames pushed faster than they are read, so pacing is required to get
        more than one distinct frame into the loop."""
        frame = np.zeros((120, 160, 3), dtype=np.uint8)

        def run():
            for i in range(n):
                drone.push(frame.copy(), 1000.0 + 0.1 * i)
                time.sleep(interval_sec)

        return threading.Thread(target=run, daemon=True)

    def test_stale_frames_are_not_reprocessed(self) -> None:
        """When the control loop outruns the camera, get_latest() returns the frame
        it already returned. Re-running optical flow on an identical image yields
        dx ~ 0 with dt > 0 -- a fabricated zero-velocity measurement. The loop must
        skip those iterations instead."""

        drone = _FakeDrone()
        cfg = StabilizerConfig(
            enable_takeoff=False,
            cmd_rate_hz=200.0,  # far faster than frames arrive
            use_kalman=False,
            enable_visual_scale=False,
            min_quality=0.0,
            settle_good_frames=0,
            climb_duration_sec=0.0,
        )

        latest = []
        s = AutoStabilizer(drone, cfg=cfg, telemetry_sink=latest.append)
        flow = _PassthroughFlow()
        s._flow = flow  # type: ignore[attr-defined]

        n_frames = 5
        pusher = self._paced_pusher(drone, n=n_frames, interval_sec=0.05)
        s.activate()
        pusher.start()
        s.run(duration_sec=0.4)
        pusher.join(timeout=1.0)

        hold = [t for t in latest if getattr(t, "phase", "") == "hold"]
        fresh = [t for t in hold if getattr(t, "frame_is_new", False)]
        stale = [t for t in hold if not getattr(t, "frame_is_new", True)]

        # The loop spun many more times than frames arrived.
        self.assertTrue(stale)
        self.assertGreater(len(stale), len(fresh))
        # Flow ran only on genuinely new frames, never on the repeats.
        self.assertLessEqual(flow.calls, n_frames)
        self.assertEqual(flow.calls, len(fresh))
        self.assertTrue(all(getattr(t, "flow", None) is None for t in stale))

    def test_stale_frame_iterations_hold_the_last_command(self) -> None:
        """Skipping a stale frame must re-latch the previous command, not drop to
        neutral -- otherwise the effective command rate is chopped by the ratio of
        loop rate to frame rate."""

        drone = _FakeDrone()
        cfg = StabilizerConfig(
            enable_takeoff=False,
            cmd_rate_hz=200.0,
            use_kalman=False,
            enable_visual_scale=False,
            min_quality=0.0,
            settle_good_frames=0,
            climb_duration_sec=0.0,
            kp_vx=0.01,
            kp_vy=0.01,
            deadband_px_s=0.0,
            estimator_deadband_px_s=0.0,
        )

        latest = []
        s = AutoStabilizer(drone, cfg=cfg, telemetry_sink=latest.append)
        s._flow = _PassthroughFlow(vx_px_s=50.0, vy_px_s=0.0)  # type: ignore[attr-defined]

        pusher = self._paced_pusher(drone, n=4, interval_sec=0.05)
        s.activate()
        pusher.start()
        s.run(duration_sec=0.4)
        pusher.join(timeout=1.0)

        hold = [t for t in latest if getattr(t, "phase", "") == "hold"]
        # Find the last iteration that had a real flow estimate, then confirm the
        # stale iterations after it repeat that command rather than zeroing it.
        idx = max((i for i, t in enumerate(hold) if getattr(t, "flow", None) is not None), default=None)
        self.assertIsNotNone(idx)
        assert idx is not None
        after = hold[idx + 1 :]
        self.assertTrue(after, "expected stale iterations after the last real frame")
        for t in after:
            self.assertAlmostEqual(float(t.cmd_roll), float(hold[idx].cmd_roll), places=9)
            self.assertAlmostEqual(float(t.cmd_pitch), float(hold[idx].cmd_pitch), places=9)

    def test_frozen_video_feed_neutralizes_instead_of_holding_forever(self) -> None:
        """Holding the last command is only safe across the gap between consecutive
        frames. If the feed freezes, frame_seq stops advancing forever -- and because
        the E88 control thread re-transmits the latched RC state at 33 Hz, holding a
        nonzero correction would fly the aircraft away."""

        drone = _FakeDrone()
        cfg = StabilizerConfig(
            enable_takeoff=False,
            cmd_rate_hz=200.0,
            use_kalman=False,
            enable_visual_scale=False,
            min_quality=0.0,
            settle_good_frames=0,
            climb_duration_sec=0.0,
            max_stale_frame_hold_sec=0.1,
            kp_vx=0.01,
            kp_vy=0.01,
            deadband_px_s=0.0,
            estimator_deadband_px_s=0.0,
        )

        latest = []
        s = AutoStabilizer(drone, cfg=cfg, telemetry_sink=latest.append)
        # Both axes nonzero: the controller call swaps vx/vy for the 90-degree
        # camera mount, so a single-axis velocity would leave one command at zero.
        s._flow = _PassthroughFlow(vx_px_s=50.0, vy_px_s=30.0)  # type: ignore[attr-defined]

        # Two frames, then the feed goes silent for the rest of the run.
        pusher = self._paced_pusher(drone, n=2, interval_sec=0.05)
        s.activate()
        pusher.start()
        s.run(duration_sec=0.6)
        pusher.join(timeout=1.0)

        hold = [t for t in latest if getattr(t, "phase", "") == "hold"]
        idx = max((i for i, t in enumerate(hold) if getattr(t, "flow", None) is not None), default=None)
        self.assertIsNotNone(idx)
        assert idx is not None

        # A nonzero correction was in flight when the feed died.
        self.assertNotEqual(float(hold[idx].cmd_roll), 0.0)
        self.assertNotEqual(float(hold[idx].cmd_pitch), 0.0)

        # The tail of the run must have neutralized rather than latched it.
        tail = hold[-5:]
        self.assertTrue(tail)
        for t in tail:
            self.assertEqual(float(t.cmd_roll), 0.0)
            self.assertEqual(float(t.cmd_pitch), 0.0)

        # ...and the same must reach the drone, not just telemetry.
        self.assertTrue(drone.sent)
        for roll, pitch, _throttle, _yaw in drone.sent[-5:]:
            self.assertEqual(float(roll), 0.0)
            self.assertEqual(float(pitch), 0.0)

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
