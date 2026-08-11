"""A ``Drone``-shaped object backed by the simulator.

``SimulatedDrone`` implements the surface ``AutoStabilizer`` actually uses --
``get_frame_with_timestamp``, ``send_cmd``, ``takeoff``, ``land``, ``close`` --
so the **unmodified** autopilot can fly it. That matters more than it sounds:
the thing most likely to crash a real E88 is not a subtly mistuned gain, it is a
sign convention or an axis swap, and those live in the wiring of
``autostabilizer.py``, not in any component you could test in isolation. Flying
the real loop is the only way to test the real wiring.

The video path deliberately reproduces the awkward parts of the real one:

- frames arrive at ~20 fps with jitter, not on demand;
- each frame is delivered *late*, carrying the timestamp of the moment it was
  decoded, so the pose it shows is already stale by the time control sees it;
- frames are occasionally lost, torn, or stop arriving entirely.

Latency is the parameter most likely to decide whether the real hover is stable,
and it is the one nobody has measured on this drone -- see
``VideoLinkParams.pipeline_latency_sec``.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from e88_sim.camera import DownwardCamera
from e88_sim.config import SimConfig
from e88_sim.ground import make_ground
from e88_sim.physics import PhysicsState, QuadPhysics


@dataclass
class TruthSample:
    """Ground truth recorded at the moment a command was issued."""

    t: float
    x_m: float
    y_m: float
    z_m: float
    vx_m_s: float
    vy_m_s: float
    roll_rad: float
    pitch_rad: float
    yaw_rad: float
    cmd_roll: float
    cmd_pitch: float
    cmd_throttle: float


@dataclass
class _PendingFrame:
    deliver_at: float
    frame_bgr: np.ndarray
    capture_pose: PhysicsState


class SimulatedDrone:
    """Simulated E88, interface-compatible with ``turbodrone.Drone``.

    Runs two background threads once :meth:`connect` is called: one integrating
    physics at the configured step, one rendering and delivering frames. Both
    are paced against the wall clock, because the autopilot paces itself against
    the wall clock too -- the point is to reproduce its real timing, including
    the jitter.
    """

    def __init__(
        self,
        cfg: Optional[SimConfig] = None,
        *,
        record_truth: bool = True,
        on_frame: Optional[Callable[[np.ndarray, float], None]] = None,
    ) -> None:
        self._cfg = cfg or SimConfig()
        rng_seed = int(self._cfg.seed)

        self._physics = QuadPhysics(self._cfg, rng=np.random.default_rng(rng_seed))
        self._camera = DownwardCamera(self._cfg.camera, ground=make_ground(self._cfg.ground))
        self._camera.seed(rng_seed + 1)
        self._link_rng = np.random.default_rng(rng_seed + 2)

        self._record_truth = bool(record_truth)
        self._on_frame = on_frame

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []

        self._pending: List[_PendingFrame] = []
        self._ready: List[Tuple[np.ndarray, float]] = []
        self._frame_available = threading.Condition(self._lock)

        self._truth: List[TruthSample] = []
        self._last_cmd = (0.0, 0.0, 50.0)
        self._last_frame_bgr: Optional[np.ndarray] = None

        self._t0: Optional[float] = None
        self._frames_rendered = 0
        self._frames_dropped_link = 0
        self._frames_torn = 0
        self._out_of_bounds_frames = 0

    # --------------------------------------------------------------- lifecycle

    def connect(self, *, start_video: bool = True, start_control: bool = True) -> None:
        if self._threads:
            return
        self._stop.clear()
        self._t0 = time.monotonic()
        self._threads = [
            threading.Thread(target=self._physics_loop, name="SimPhysics", daemon=True),
            threading.Thread(target=self._video_loop, name="SimVideo", daemon=True),
        ]
        for t in self._threads:
            t.start()

    def close(self) -> None:
        self._stop.set()
        with self._lock:
            self._frame_available.notify_all()
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads = []

    # ------------------------------------------------------------------ frames

    def get_frame_with_timestamp(self, timeout: Optional[float] = None) -> Optional[Tuple[np.ndarray, float]]:
        deadline = None if timeout is None else (time.monotonic() + float(timeout))
        with self._frame_available:
            while not self._ready:
                if self._stop.is_set():
                    return None
                remaining = None if deadline is None else (deadline - time.monotonic())
                if remaining is not None and remaining <= 0.0:
                    return None
                self._frame_available.wait(timeout=0.02 if remaining is None else min(0.02, remaining))
            frame, ts = self._ready.pop(0)
        return frame, ts

    def get_frame(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        item = self.get_frame_with_timestamp(timeout=timeout)
        return None if item is None else item[0]

    # ---------------------------------------------------------------- commands

    def send_cmd(self, *, roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0, throttle: float = 50.0) -> None:
        with self._lock:
            self._physics.submit_command(roll=roll, pitch=pitch, yaw=yaw, throttle=throttle)
            self._last_cmd = (float(roll), float(pitch), float(throttle))
            if self._record_truth:
                self._truth.append(self._truth_sample(self._physics.state()))

    def set_sticks_raw(self, *, roll=None, pitch=None, throttle=None, yaw=None) -> None:
        from e88_sim.physics import byte_to_axis, byte_to_throttle

        with self._lock:
            state = self._physics.state()
            self._physics.submit_command(
                roll=byte_to_axis(roll) if roll is not None else float(state.applied_roll),
                pitch=byte_to_axis(pitch) if pitch is not None else float(state.applied_pitch),
                yaw=byte_to_axis(yaw) if yaw is not None else float(state.applied_yaw),
                throttle=byte_to_throttle(throttle) if throttle is not None else float(state.applied_throttle),
            )

    def takeoff(self) -> None:
        with self._lock:
            self._physics.request_takeoff()

    def land(self) -> None:
        with self._lock:
            self._physics.request_land()

    def calibrate(self) -> None:  # pragma: no cover - no simulated effect
        pass

    def flip(self) -> None:  # pragma: no cover - not simulated
        pass

    def toggle_headless(self) -> None:  # pragma: no cover - not simulated
        pass

    def switch_camera(self, cam: int) -> None:  # pragma: no cover - one camera
        pass

    # ------------------------------------------------------------------- truth

    @property
    def truth(self) -> List[TruthSample]:
        with self._lock:
            return list(self._truth)

    def state(self) -> PhysicsState:
        with self._lock:
            return self._physics.state()

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "frames_rendered": int(self._frames_rendered),
                "frames_dropped_link": int(self._frames_dropped_link),
                "frames_torn": int(self._frames_torn),
                "frames_out_of_bounds": int(self._out_of_bounds_frames),
            }

    @property
    def camera(self) -> DownwardCamera:
        return self._camera

    @property
    def cfg(self) -> SimConfig:
        return self._cfg

    def _truth_sample(self, s: PhysicsState) -> TruthSample:
        return TruthSample(
            t=float(s.t),
            x_m=float(s.x_m),
            y_m=float(s.y_m),
            z_m=float(s.z_m),
            vx_m_s=float(s.vx_m_s),
            vy_m_s=float(s.vy_m_s),
            roll_rad=float(s.roll_rad),
            pitch_rad=float(s.pitch_rad),
            yaw_rad=float(s.yaw_rad),
            cmd_roll=float(self._last_cmd[0]),
            cmd_pitch=float(self._last_cmd[1]),
            cmd_throttle=float(self._last_cmd[2]),
        )

    # ------------------------------------------------------------------ threads

    def _physics_loop(self) -> None:
        step = max(0.002, float(self._cfg.physics_dt_sec))
        last = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            elapsed = now - last
            if elapsed < step:
                time.sleep(step - elapsed)
                now = time.monotonic()
                elapsed = now - last
            last = now
            # Cap the catch-up so a scheduling hiccup cannot make the drone
            # teleport; the sim falls behind wall clock instead, which is
            # visible in the results rather than silently corrupting them.
            with self._lock:
                self._physics.step(min(elapsed, 0.05))

    def _video_loop(self) -> None:
        video = self._cfg.video
        period = 1.0 / max(1.0, float(video.fps))
        next_capture = time.monotonic()

        while not self._stop.is_set():
            now = time.monotonic()
            self._deliver_due_frames(now)

            if now < next_capture:
                time.sleep(min(0.005, next_capture - now))
                continue

            jitter = float(self._link_rng.normal(0.0, max(0.0, float(video.fps_jitter_sec))))
            next_capture = now + max(0.001, period + jitter)

            if self._is_frozen(now):
                continue
            if float(video.drop_probability) > 0.0 and self._link_rng.random() < float(video.drop_probability):
                with self._lock:
                    self._frames_dropped_link += 1
                continue

            with self._lock:
                pose = self._physics.state()
            result = self._camera.render(pose)
            frame = result.frame_bgr

            with self._lock:
                self._frames_rendered += 1
                if result.out_of_bounds:
                    self._out_of_bounds_frames += 1

            if float(video.tear_probability) > 0.0 and self._link_rng.random() < float(video.tear_probability):
                frame = self._tear(frame)
                with self._lock:
                    self._frames_torn += 1

            latency = max(
                0.0,
                float(video.pipeline_latency_sec)
                + float(self._link_rng.normal(0.0, max(0.0, float(video.latency_jitter_sec)))),
            )
            with self._lock:
                self._pending.append(_PendingFrame(deliver_at=now + latency, frame_bgr=frame, capture_pose=pose))
                self._last_frame_bgr = frame

    def _deliver_due_frames(self, now: float) -> None:
        with self._frame_available:
            if not self._pending:
                return
            # Latency jitter can reorder frames; the real decoder emits them in
            # order, so sort before delivering.
            self._pending.sort(key=lambda p: p.deliver_at)
            due = [p for p in self._pending if p.deliver_at <= now]
            if not due:
                return
            self._pending = [p for p in self._pending if p.deliver_at > now]
            for p in due:
                # The timestamp the autopilot sees is the *decode* time, exactly
                # as in e88/video.py -- not the capture time. The gap between
                # them is the unmeasured latency the drone actually flies with.
                self._ready.append((p.frame_bgr, float(p.deliver_at)))
                if self._on_frame is not None:
                    try:
                        self._on_frame(p.frame_bgr, float(p.deliver_at))
                    except Exception:
                        pass
            # A size-1 buffer sits downstream in LatestFrameBuffer; keeping a
            # short queue here prevents unbounded growth if nobody reads.
            if len(self._ready) > 4:
                self._ready = self._ready[-4:]
            self._frame_available.notify_all()

    def _is_frozen(self, now: float) -> bool:
        video = self._cfg.video
        dur = float(video.freeze_duration_sec)
        if dur <= 0.0 or self._t0 is None:
            return False
        elapsed = now - float(self._t0)
        start = float(video.freeze_start_sec)
        return bool(start <= elapsed < (start + dur))

    def _tear(self, frame: np.ndarray) -> np.ndarray:
        """Splice the top of the previous frame onto the bottom of this one.

        README 8.2: lost UDP packets leave a decoded frame whose halves show
        different moments, and optical flow reads the seam as a huge jump. The
        estimator has gates for exactly this, so the sim has to be able to
        trigger them.
        """
        prev = self._last_frame_bgr
        if prev is None or prev.shape != frame.shape:
            return frame
        h = frame.shape[0]
        split = int(self._link_rng.integers(int(0.2 * h), int(0.8 * h)))
        out = frame.copy()
        out[:split] = prev[:split]
        return out
