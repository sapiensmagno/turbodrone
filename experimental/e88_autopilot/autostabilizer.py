from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

from e88_autopilot.controller import VelocityHoldController
from e88_autopilot.kalman import VelocityKalman2D
from e88_autopilot.optical_flow import LucasKanadeDriftEstimator
from turbodrone import Drone


@dataclass(frozen=True)
class StabilizerConfig:
    enable_takeoff: bool = False
    takeoff_throttle: float = 100.0
    takeoff_duration_sec: float = 0.5
    climb_throttle: float = 70.0
    climb_duration_sec: float = 1.5
    settle_good_frames: int = 5

    base_throttle: float = 50.0
    cmd_rate_hz: float = 20.0

    use_kalman: bool = True
    kalman_sigma_a: float = 25.0
    kalman_sigma_v: float = 60.0

    min_quality: float = 0.15
    max_cmd: float = 0.35

    kp_vx: float = 0.003
    kp_vy: float = 0.003
    ki_vx: float = 0.0005
    ki_vy: float = 0.0005
    deadband_px_s: float = 3.0
    roll_sign: float = -1.0
    pitch_sign: float = -1.0


class AutoStabilizer:
    def __init__(self, drone: Drone, *, cfg: Optional[StabilizerConfig] = None) -> None:
        self._drone = drone
        self._cfg = cfg or StabilizerConfig()

        self._flow = LucasKanadeDriftEstimator()
        self._kf = None
        if self._cfg.use_kalman:
            self._kf = VelocityKalman2D(sigma_a=self._cfg.kalman_sigma_a, sigma_v_meas=self._cfg.kalman_sigma_v)

        self._ctl = VelocityHoldController(
            min_quality=self._cfg.min_quality,
            max_cmd=self._cfg.max_cmd,
            kp_vx=self._cfg.kp_vx,
            kp_vy=self._cfg.kp_vy,
            ki_vx=self._cfg.ki_vx,
            ki_vy=self._cfg.ki_vy,
            deadband_px_s=self._cfg.deadband_px_s,
            roll_sign=self._cfg.roll_sign,
            pitch_sign=self._cfg.pitch_sign,
        )

        self._active = False
        self._last_loop_t: Optional[float] = None
        self._stop = threading.Event()

    def activate(self) -> None:
        self._flow.reset()
        if self._kf is not None:
            self._kf.reset()
        self._ctl.reset()
        self._last_loop_t = None
        self._active = True
        self._stop.clear()

    def deactivate(self) -> None:
        self._active = False

    def request_stop(self) -> None:
        self._stop.set()

    def run(self, *, duration_sec: Optional[float] = None) -> None:
        if not self._active:
            self.activate()

        start = time.monotonic()

        if self._cfg.enable_takeoff:
            t0 = time.monotonic()
            while not self._stop.is_set() and (time.monotonic() - t0) < self._cfg.takeoff_duration_sec:
                self._drone.takeoff()
                self._drone.send_cmd(roll=0.0, pitch=0.0, throttle=self._cfg.takeoff_throttle)
                time.sleep(0.05)

            t1 = time.monotonic()
            while not self._stop.is_set() and (time.monotonic() - t1) < self._cfg.climb_duration_sec:
                self._drone.send_cmd(roll=0.0, pitch=0.0, throttle=self._cfg.climb_throttle)
                time.sleep(0.05)

            good_needed = max(0, int(self._cfg.settle_good_frames))
            good_seen = 0
            while not self._stop.is_set() and good_seen < good_needed:
                item = self._drone.get_frame_with_timestamp(timeout=2.0)
                if item is None:
                    self._drone.send_cmd(roll=0.0, pitch=0.0, throttle=self._cfg.climb_throttle)
                    continue

                frame, ts = item
                est = self._flow.update(frame, timestamp=ts)
                if est is None:
                    self._drone.send_cmd(roll=0.0, pitch=0.0, throttle=self._cfg.climb_throttle)
                    continue

                if est.quality >= self._cfg.min_quality:
                    good_seen += 1
                else:
                    good_seen = 0

                self._drone.send_cmd(roll=0.0, pitch=0.0, throttle=self._cfg.climb_throttle)

            start = time.monotonic()

        period = 1.0 / max(1.0, float(self._cfg.cmd_rate_hz))

        while True:
            now = time.monotonic()
            if self._stop.is_set():
                break
            if duration_sec is not None and (now - start) >= float(duration_sec):
                break

            item = self._drone.get_frame_with_timestamp(timeout=2.0)
            if item is None:
                self._drone.send_cmd(roll=0.0, pitch=0.0, throttle=self._cfg.base_throttle)
                time.sleep(period)
                continue

            frame, ts = item

            est = self._flow.update(frame, timestamp=ts)
            if est is None:
                self._drone.send_cmd(roll=0.0, pitch=0.0, throttle=self._cfg.base_throttle)
                time.sleep(period)
                continue

            vx, vy = est.vx_px_s, est.vy_px_s
            q = est.quality
            if self._kf is not None:
                k = self._kf.update_velocity(t=ts, vx_px_s=vx, vy_px_s=vy, quality=q)
                vx, vy = k.vx_px_s, k.vy_px_s

            out = self._ctl.update(dt_sec=est.dt_sec, vx_px_s=vx, vy_px_s=vy, quality=q)
            if out is None:
                self._drone.send_cmd(roll=0.0, pitch=0.0, throttle=self._cfg.base_throttle)
            else:
                self._drone.send_cmd(roll=out.roll, pitch=out.pitch, throttle=self._cfg.base_throttle)

            elapsed = time.monotonic() - now
            if elapsed < period:
                time.sleep(period - elapsed)
