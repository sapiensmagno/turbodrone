from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from e88_autopilot.controller import VelocityHoldController
from e88_autopilot.kalman import VelocityKalman2D
from e88_autopilot.kalman import KalmanEstimate
from e88_autopilot.measurement_conditioner import VelocityMeasurementConditioner
from e88_autopilot.optical_flow import FlowEstimate, FlowTracks, LucasKanadeDriftEstimator
from turbodrone import Drone


@dataclass(frozen=True)
class StabilizerConfig:
    enable_takeoff: bool = False
    takeoff_throttle: float = 100.0
    takeoff_duration_sec: float = 2.0
    climb_throttle: float = 70.0
    climb_duration_sec: float = 1.5
    settle_good_frames: int = 5

    base_throttle: float = 50.0
    cmd_rate_hz: float = 20.0

    use_kalman: bool = True
    kalman_sigma_a: float = 25.0
    kalman_sigma_v: float = 460.0

    min_quality: float = 0.15
    max_cmd: float = 0.35

    kp_vx: float = 0.003
    kp_vy: float = 0.003
    ki_vx: float = 0.0005
    ki_vy: float = 0.0005
    deadband_px_s: float = 3.0
    estimator_deadband_px_s: float = 3.0
    roll_sign: float = -1.0
    pitch_sign: float = -1.0


@dataclass(frozen=True)
class StabilizerTelemetry:
    phase: str
    frame_bgr: Optional[np.ndarray]
    timestamp: float
    pos_x_px: float
    pos_y_px: float
    flow: Optional[FlowEstimate]
    tracks: Optional[FlowTracks]
    kalman: Optional[KalmanEstimate]
    used_vx_px_s: float
    used_vy_px_s: float
    cmd_roll: float
    cmd_pitch: float
    cmd_throttle: float
    kf_input_vx_px_s: float = 0.0
    kf_input_vy_px_s: float = 0.0
    kf_gated: bool = False
    t_loop_start: float = 0.0
    t_frame_received: float = 0.0
    t_flow_start: float = 0.0
    t_flow_end: float = 0.0
    t_kf_end: float = 0.0
    t_ctrl_end: float = 0.0
    t_cmd_sent: float = 0.0
    dt_flow_ms: float = 0.0
    dt_total_ms: float = 0.0
    frame_age_ms: float = 0.0
    estimated_latency_ms: float = 0.0
    loop_rate_hz: float = 0.0
    frame_rate_hz: float = 0.0


class AutoStabilizer:
    def __init__(
        self,
        drone: Drone,
        *,
        cfg: Optional[StabilizerConfig] = None,
        telemetry_sink: Optional[Callable[[StabilizerTelemetry], None]] = None,
    ) -> None:
        self._drone = drone
        self._cfg = cfg or StabilizerConfig()
        self._telemetry_sink = telemetry_sink

        self._flow = LucasKanadeDriftEstimator()
        self._kf = None
        if self._cfg.use_kalman:
            self._kf = VelocityKalman2D(sigma_a=self._cfg.kalman_sigma_a, sigma_v_meas=self._cfg.kalman_sigma_v)

        self._meas_cond = VelocityMeasurementConditioner(deadband_px_s=self._cfg.estimator_deadband_px_s)

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
        self._loop_rate_hz_ema = 0.0
        self._frame_rate_hz_ema = 0.0
        self._stop = threading.Event()

        self._viz_x_px = 0.0
        self._viz_y_px = 0.0

    def activate(self) -> None:
        self._flow.reset()
        if self._kf is not None:
            self._kf.reset()
        self._ctl.reset()
        self._last_loop_t = None
        self._loop_rate_hz_ema = 0.0
        self._frame_rate_hz_ema = 0.0
        self._active = True
        self._stop.clear()
        self._viz_x_px = 0.0
        self._viz_y_px = 0.0

    def _update_rate_ema(self, *, prev: float, inst: float, alpha: float = 0.1) -> float:
        i = float(inst)
        if i <= 0.0 or not np.isfinite(i):
            return float(prev)
        if prev <= 0.0:
            return float(i)
        a = float(min(1.0, max(0.0, alpha)))
        return float((1.0 - a) * float(prev) + a * i)

    def deactivate(self) -> None:
        self._active = False

    def request_stop(self) -> None:
        self._stop.set()

    def set_telemetry_sink(self, sink: Optional[Callable[[StabilizerTelemetry], None]]) -> None:
        self._telemetry_sink = sink

    def _emit(self, t: StabilizerTelemetry) -> None:
        sink = self._telemetry_sink
        if sink is None:
            return
        try:
            sink(t)
        except Exception:
            pass

    def run(self, *, duration_sec: Optional[float] = None) -> None:
        if not self._active:
            self.activate()

        start = time.monotonic()

        if self._cfg.enable_takeoff:
            t0 = time.monotonic()
            while not self._stop.is_set() and (time.monotonic() - t0) < self._cfg.takeoff_duration_sec:
                t_loop_start = float(time.monotonic())
                self._drone.takeoff()
                self._drone.send_cmd(roll=0.0, pitch=0.0, throttle=self._cfg.takeoff_throttle)
                t_cmd_sent = float(time.monotonic())
                self._emit(
                    StabilizerTelemetry(
                        phase="takeoff",
                        frame_bgr=None,
                        timestamp=float(time.monotonic()),
                        pos_x_px=float(self._viz_x_px),
                        pos_y_px=float(self._viz_y_px),
                        flow=None,
                        tracks=None,
                        kalman=None,
                        used_vx_px_s=0.0,
                        used_vy_px_s=0.0,
                        cmd_roll=0.0,
                        cmd_pitch=0.0,
                        cmd_throttle=float(self._cfg.takeoff_throttle),
                        t_loop_start=float(t_loop_start),
                        t_cmd_sent=float(t_cmd_sent),
                        dt_total_ms=float((t_cmd_sent - t_loop_start) * 1000.0),
                    )
                )
                time.sleep(0.05)

            t1 = time.monotonic()
            while not self._stop.is_set() and (time.monotonic() - t1) < self._cfg.climb_duration_sec:
                t_loop_start = float(time.monotonic())
                self._drone.send_cmd(roll=0.0, pitch=0.0, throttle=self._cfg.climb_throttle)
                t_cmd_sent = float(time.monotonic())
                self._emit(
                    StabilizerTelemetry(
                        phase="climb",
                        frame_bgr=None,
                        timestamp=float(time.monotonic()),
                        pos_x_px=float(self._viz_x_px),
                        pos_y_px=float(self._viz_y_px),
                        flow=None,
                        tracks=None,
                        kalman=None,
                        used_vx_px_s=0.0,
                        used_vy_px_s=0.0,
                        cmd_roll=0.0,
                        cmd_pitch=0.0,
                        cmd_throttle=float(self._cfg.climb_throttle),
                        t_loop_start=float(t_loop_start),
                        t_cmd_sent=float(t_cmd_sent),
                        dt_total_ms=float((t_cmd_sent - t_loop_start) * 1000.0),
                    )
                )
                time.sleep(0.05)

            good_needed = max(0, int(self._cfg.settle_good_frames))
            good_seen = 0
            while not self._stop.is_set() and good_seen < good_needed:
                t_loop_start = float(time.monotonic())
                item = self._drone.get_frame_with_timestamp(timeout=2.0)
                if item is None:
                    self._drone.send_cmd(roll=0.0, pitch=0.0, throttle=self._cfg.climb_throttle)
                    t_cmd_sent = float(time.monotonic())
                    self._emit(
                        StabilizerTelemetry(
                            phase="settle",
                            frame_bgr=None,
                            timestamp=float(time.monotonic()),
                            pos_x_px=float(self._viz_x_px),
                            pos_y_px=float(self._viz_y_px),
                            flow=None,
                            tracks=None,
                            kalman=None,
                            used_vx_px_s=0.0,
                            used_vy_px_s=0.0,
                            cmd_roll=0.0,
                            cmd_pitch=0.0,
                            cmd_throttle=float(self._cfg.climb_throttle),
                            t_loop_start=float(t_loop_start),
                            t_cmd_sent=float(t_cmd_sent),
                            dt_total_ms=float((t_cmd_sent - t_loop_start) * 1000.0),
                        )
                    )
                    continue

                frame, ts = item

                t_frame_received = float(time.monotonic())
                frame_age_ms = float(max(0.0, (t_frame_received - float(ts)) * 1000.0))

                t_flow_start = float(time.monotonic())
                est = self._flow.update(frame, timestamp=ts)
                t_flow_end = float(time.monotonic())

                if est is None:
                    self._drone.send_cmd(roll=0.0, pitch=0.0, throttle=self._cfg.climb_throttle)
                    t_cmd_sent = float(time.monotonic())
                    self._emit(
                        StabilizerTelemetry(
                            phase="settle",
                            frame_bgr=frame,
                            timestamp=float(ts),
                            pos_x_px=float(self._viz_x_px),
                            pos_y_px=float(self._viz_y_px),
                            flow=None,
                            tracks=self._flow.last_tracks(),
                            kalman=None,
                            used_vx_px_s=0.0,
                            used_vy_px_s=0.0,
                            cmd_roll=0.0,
                            cmd_pitch=0.0,
                            cmd_throttle=float(self._cfg.climb_throttle),
                            t_loop_start=float(t_loop_start),
                            t_frame_received=float(t_frame_received),
                            t_flow_start=float(t_flow_start),
                            t_flow_end=float(t_flow_end),
                            t_kf_end=float(t_flow_end),
                            t_ctrl_end=float(t_flow_end),
                            t_cmd_sent=float(t_cmd_sent),
                            dt_flow_ms=float((t_flow_end - t_flow_start) * 1000.0),
                            dt_total_ms=float((t_cmd_sent - t_loop_start) * 1000.0),
                            frame_age_ms=float(frame_age_ms),
                            estimated_latency_ms=float(max(0.0, (t_cmd_sent - float(ts)) * 1000.0)),
                            loop_rate_hz=float(self._loop_rate_hz_ema),
                            frame_rate_hz=float(self._frame_rate_hz_ema),
                        )
                    )
                    continue

                if est.quality >= self._cfg.min_quality:
                    good_seen += 1
                else:
                    good_seen = 0

                self._drone.send_cmd(roll=0.0, pitch=0.0, throttle=self._cfg.climb_throttle)

                cond = self._meas_cond.apply(vx_px_s=float(est.vx_px_s), vy_px_s=float(est.vy_px_s))
                self._viz_x_px += float(cond.vx_px_s) * float(est.dt_sec)
                self._viz_y_px += float(cond.vy_px_s) * float(est.dt_sec)

                t_ctrl_end = float(time.monotonic())
                self._frame_rate_hz_ema = self._update_rate_ema(prev=self._frame_rate_hz_ema, inst=(1.0 / float(est.dt_sec)))
                t_cmd_sent = float(time.monotonic())

                self._emit(
                    StabilizerTelemetry(
                        phase="settle",
                        frame_bgr=frame,
                        timestamp=float(ts),
                        pos_x_px=float(self._viz_x_px),
                        pos_y_px=float(self._viz_y_px),
                        flow=est,
                        tracks=self._flow.last_tracks(),
                        kalman=None,
                        used_vx_px_s=float(cond.vx_px_s),
                        used_vy_px_s=float(cond.vy_px_s),
                        kf_input_vx_px_s=float(cond.vx_px_s),
                        kf_input_vy_px_s=float(cond.vy_px_s),
                        kf_gated=bool(cond.gated),
                        cmd_roll=0.0,
                        cmd_pitch=0.0,
                        cmd_throttle=float(self._cfg.climb_throttle),
                        t_loop_start=float(t_loop_start),
                        t_frame_received=float(t_frame_received),
                        t_flow_start=float(t_flow_start),
                        t_flow_end=float(t_flow_end),
                        t_kf_end=float(t_flow_end),
                        t_ctrl_end=float(t_ctrl_end),
                        t_cmd_sent=float(t_cmd_sent),
                        dt_flow_ms=float((t_flow_end - t_flow_start) * 1000.0),
                        dt_total_ms=float((t_cmd_sent - t_loop_start) * 1000.0),
                        frame_age_ms=float(frame_age_ms),
                        estimated_latency_ms=float(max(0.0, (t_cmd_sent - float(ts)) * 1000.0)),
                        loop_rate_hz=float(self._loop_rate_hz_ema),
                        frame_rate_hz=float(self._frame_rate_hz_ema),
                    )
                )

            start = time.monotonic()

        period = 1.0 / max(1.0, float(self._cfg.cmd_rate_hz))

        while True:
            now = time.monotonic()
            if self._last_loop_t is not None:
                dt_loop = float(now - float(self._last_loop_t))
                if dt_loop > 1e-6:
                    self._loop_rate_hz_ema = self._update_rate_ema(prev=self._loop_rate_hz_ema, inst=(1.0 / dt_loop))
            self._last_loop_t = float(now)

            t_loop_start = float(now)
            if self._stop.is_set():
                break
            if duration_sec is not None and (now - start) >= float(duration_sec):
                break

            item = self._drone.get_frame_with_timestamp(timeout=2.0)
            if item is None:
                self._drone.send_cmd(roll=0.0, pitch=0.0, throttle=self._cfg.base_throttle)
                t_cmd_sent = float(time.monotonic())
                self._emit(
                    StabilizerTelemetry(
                        phase="hold",
                        frame_bgr=None,
                        timestamp=float(time.monotonic()),
                        pos_x_px=float(self._viz_x_px),
                        pos_y_px=float(self._viz_y_px),
                        flow=None,
                        tracks=None,
                        kalman=None,
                        used_vx_px_s=0.0,
                        used_vy_px_s=0.0,
                        cmd_roll=0.0,
                        cmd_pitch=0.0,
                        cmd_throttle=float(self._cfg.base_throttle),
                        t_loop_start=float(t_loop_start),
                        t_cmd_sent=float(t_cmd_sent),
                        dt_total_ms=float((t_cmd_sent - t_loop_start) * 1000.0),
                        loop_rate_hz=float(self._loop_rate_hz_ema),
                        frame_rate_hz=float(self._frame_rate_hz_ema),
                    )
                )
                time.sleep(period)
                continue

            frame, ts = item

            t_frame_received = float(time.monotonic())
            frame_age_ms = float(max(0.0, (t_frame_received - float(ts)) * 1000.0))

            t_flow_start = float(time.monotonic())
            est = self._flow.update(frame, timestamp=ts)
            t_flow_end = float(time.monotonic())
            if est is None:
                self._drone.send_cmd(roll=0.0, pitch=0.0, throttle=self._cfg.base_throttle)
                t_cmd_sent = float(time.monotonic())
                self._emit(
                    StabilizerTelemetry(
                        phase="hold",
                        frame_bgr=frame,
                        timestamp=float(ts),
                        pos_x_px=float(self._viz_x_px),
                        pos_y_px=float(self._viz_y_px),
                        flow=None,
                        tracks=self._flow.last_tracks(),
                        kalman=None,
                        used_vx_px_s=0.0,
                        used_vy_px_s=0.0,
                        cmd_roll=0.0,
                        cmd_pitch=0.0,
                        cmd_throttle=float(self._cfg.base_throttle),
                        t_loop_start=float(t_loop_start),
                        t_frame_received=float(t_frame_received),
                        t_flow_start=float(t_flow_start),
                        t_flow_end=float(t_flow_end),
                        t_kf_end=float(t_flow_end),
                        t_ctrl_end=float(t_flow_end),
                        t_cmd_sent=float(t_cmd_sent),
                        dt_flow_ms=float((t_flow_end - t_flow_start) * 1000.0),
                        dt_total_ms=float((t_cmd_sent - t_loop_start) * 1000.0),
                        frame_age_ms=float(frame_age_ms),
                        estimated_latency_ms=float(max(0.0, (t_cmd_sent - float(ts)) * 1000.0)),
                        loop_rate_hz=float(self._loop_rate_hz_ema),
                        frame_rate_hz=float(self._frame_rate_hz_ema),
                    )
                )
                time.sleep(period)
                continue

            flow_vx, flow_vy = float(est.vx_px_s), float(est.vy_px_s)
            q = est.quality

            cond = self._meas_cond.apply(vx_px_s=flow_vx, vy_px_s=flow_vy)
            vx, vy = float(cond.vx_px_s), float(cond.vy_px_s)
            k_est = None
            if self._kf is not None:
                k_est = self._kf.update_velocity(t=ts, vx_px_s=vx, vy_px_s=vy, quality=q)
                vx, vy = k_est.vx_px_s, k_est.vy_px_s
                self._viz_x_px = float(k_est.x_px)
                self._viz_y_px = float(k_est.y_px)
            else:
                self._viz_x_px += float(vx) * float(est.dt_sec)
                self._viz_y_px += float(vy) * float(est.dt_sec)

            t_kf_end = float(time.monotonic())
            self._frame_rate_hz_ema = self._update_rate_ema(prev=self._frame_rate_hz_ema, inst=(1.0 / float(est.dt_sec)))

            out = self._ctl.update(dt_sec=est.dt_sec, vx_px_s=vx, vy_px_s=vy, quality=q)
            t_ctrl_end = float(time.monotonic())
            if out is None:
                self._drone.send_cmd(roll=0.0, pitch=0.0, throttle=self._cfg.base_throttle)
                t_cmd_sent = float(time.monotonic())
                self._emit(
                    StabilizerTelemetry(
                        phase="hold",
                        frame_bgr=frame,
                        timestamp=float(ts),
                        pos_x_px=float(self._viz_x_px),
                        pos_y_px=float(self._viz_y_px),
                        flow=est,
                        tracks=self._flow.last_tracks(),
                        kalman=k_est,
                        used_vx_px_s=float(vx),
                        used_vy_px_s=float(vy),
                        kf_input_vx_px_s=float(cond.vx_px_s),
                        kf_input_vy_px_s=float(cond.vy_px_s),
                        kf_gated=bool(cond.gated),
                        cmd_roll=0.0,
                        cmd_pitch=0.0,
                        cmd_throttle=float(self._cfg.base_throttle),
                        t_loop_start=float(t_loop_start),
                        t_frame_received=float(t_frame_received),
                        t_flow_start=float(t_flow_start),
                        t_flow_end=float(t_flow_end),
                        t_kf_end=float(t_kf_end),
                        t_ctrl_end=float(t_ctrl_end),
                        t_cmd_sent=float(t_cmd_sent),
                        dt_flow_ms=float((t_flow_end - t_flow_start) * 1000.0),
                        dt_total_ms=float((t_cmd_sent - t_loop_start) * 1000.0),
                        frame_age_ms=float(frame_age_ms),
                        estimated_latency_ms=float(max(0.0, (t_cmd_sent - float(ts)) * 1000.0)),
                        loop_rate_hz=float(self._loop_rate_hz_ema),
                        frame_rate_hz=float(self._frame_rate_hz_ema),
                    )
                )
            else:
                self._drone.send_cmd(roll=out.roll, pitch=out.pitch, throttle=self._cfg.base_throttle)
                t_cmd_sent = float(time.monotonic())

                self._emit(
                    StabilizerTelemetry(
                        phase="hold",
                        frame_bgr=frame,
                        timestamp=float(ts),
                        pos_x_px=float(self._viz_x_px),
                        pos_y_px=float(self._viz_y_px),
                        flow=est,
                        tracks=self._flow.last_tracks(),
                        kalman=k_est,
                        used_vx_px_s=float(vx),
                        used_vy_px_s=float(vy),
                        kf_input_vx_px_s=float(cond.vx_px_s),
                        kf_input_vy_px_s=float(cond.vy_px_s),
                        kf_gated=bool(cond.gated),
                        cmd_roll=float(out.roll),
                        cmd_pitch=float(out.pitch),
                        cmd_throttle=float(self._cfg.base_throttle),
                        t_loop_start=float(t_loop_start),
                        t_frame_received=float(t_frame_received),
                        t_flow_start=float(t_flow_start),
                        t_flow_end=float(t_flow_end),
                        t_kf_end=float(t_kf_end),
                        t_ctrl_end=float(t_ctrl_end),
                        t_cmd_sent=float(t_cmd_sent),
                        dt_flow_ms=float((t_flow_end - t_flow_start) * 1000.0),
                        dt_total_ms=float((t_cmd_sent - t_loop_start) * 1000.0),
                        frame_age_ms=float(frame_age_ms),
                        estimated_latency_ms=float(max(0.0, (t_cmd_sent - float(ts)) * 1000.0)),
                        loop_rate_hz=float(self._loop_rate_hz_ema),
                        frame_rate_hz=float(self._frame_rate_hz_ema),
                    )
                )

            elapsed = time.monotonic() - now
            if elapsed < period:
                time.sleep(period - elapsed)
