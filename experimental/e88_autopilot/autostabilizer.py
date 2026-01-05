from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from e88_autopilot.controller import VelocityHoldController
from e88_autopilot.kalman import VelocityKalman2D
from e88_autopilot.kalman import KalmanEstimate
from e88_autopilot.measurement_conditioner import VelocityMeasurementConditioner
from e88_autopilot.optical_flow import FlowEstimate, FlowTracks, LucasKanadeDriftEstimator
from e88_autopilot.reference_store import ReferenceStore
from e88_autopilot.visual_scale import VisualScaleEstimator
from turbodrone import Drone


@dataclass(frozen=True)
class StabilizerConfig:
    enable_takeoff: bool = False
    takeoff_throttle: float = 100.0
    takeoff_duration_sec: float = 0.5
    climb_throttle: float = 90.0
    climb_duration_sec: float = 0.5
    settle_good_frames: int = 2

    base_throttle: float = 50.0
    cmd_rate_hz: float = 20.0

    use_kalman: bool = True
    kalman_sigma_a: float = 25.0
    kalman_sigma_v: float = 1.0

    min_quality: float = 0.15
    max_cmd: float = 0.9

    flow_motion_model: str = "affine_translation"
    flow_tr_residual_thresh_px: float = 3.0
    flow_tr_min_points: int = 20

    kp_vx: float = 0.003
    kp_vy: float = 0.003
    ki_vx: float = 0.0005
    ki_vy: float = 0.0005
    deadband_px_s: float = 3.0
    estimator_deadband_px_s: float = 3.0
    roll_sign: float = -1.0
    pitch_sign: float = -1.0

    enable_visual_scale: bool = False
    reference_id: Optional[str] = None
    visual_scale_detection_method: str = "orb_contours"
    use_m_s_control: bool = True
    kp_vx_m_s: float = 1.5
    kp_vy_m_s: float = 1.5
    ki_vx_m_s: float = 0.25
    ki_vy_m_s: float = 0.25
    deadband_m_s: float = 0.02
    visual_scale_unstable_cmd_scale: float = 1.0
    visual_scale_ms_hold_sec: float = 1.0
    visual_scale_stable_frames: int = 8
    visual_scale_max_ref_size_frac_per_sec: float = 2.0


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
    frame_stale_ms: float = 0.0
    frame_seq: int = 0
    frame_is_new: bool = False
    frames_dropped: int = 0
    estimated_latency_ms: float = 0.0
    loop_rate_hz: float = 0.0
    frame_rate_hz: float = 0.0

    visual_scale_enabled: bool = False
    visual_scale_error: str = ""

    altitude_est_m: Optional[float] = None
    altitude_source: str = "unknown"
    ref_detected: bool = False
    ref_width_px: float = 0.0
    ref_height_px: float = 0.0
    ref_size_px: float = 0.0
    ref_quad_xy: Optional[np.ndarray] = None
    vx_m_s: Optional[float] = None
    vy_m_s: Optional[float] = None
    used_vx_m_s: Optional[float] = None
    used_vy_m_s: Optional[float] = None
    scale_stable: bool = False
    ref_mode: str = ""
    ref_n_matches: int = 0
    ref_n_inliers: int = 0
    ref_inlier_ratio: float = 0.0
    ref_reproj_error_px: float = 0.0


class LatestFrameBuffer:
    def __init__(self, drone: Drone, *, timeout_sec: float = 0.25) -> None:
        self._drone = drone
        self._timeout_sec = float(timeout_sec)

        self._lock = threading.Lock()
        self._latest: Optional[Tuple[np.ndarray, float, float, int]] = None
        self._seq = 0
        self._last_read_seq = 0
        self._dropped_total = 0
        self._last_source_ts: Optional[float] = None

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="LatestFrameBuffer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        self._thread = None
        if t is not None:
            t.join(timeout=1.0)

    @property
    def dropped_total(self) -> int:
        with self._lock:
            return int(self._dropped_total)

    def get_latest(self) -> Optional[Tuple[np.ndarray, float, float, int, int]]:
        with self._lock:
            item = self._latest
            if item is None:
                return None

            frame, ts, t_received, seq = item
            if int(seq) != int(self._last_read_seq):
                dropped = int(max(0, int(seq) - int(self._last_read_seq) - 1))
                self._dropped_total += dropped
                self._last_read_seq = int(seq)
            return frame.copy(), float(ts), float(t_received), int(seq), int(self._dropped_total)

    def _run(self) -> None:
        while not self._stop.is_set():
            item = self._drone.get_frame_with_timestamp(timeout=self._timeout_sec)
            if item is None:
                continue

            frame, ts = item
            ts_f = float(ts)
            last_ts = self._last_source_ts
            if last_ts is not None and ts_f <= float(last_ts) + 1e-9:
                time.sleep(0.001)
                continue
            self._last_source_ts = float(ts_f)
            t_received = float(time.monotonic())
            with self._lock:
                self._seq += 1
                self._latest = (frame, float(ts_f), float(t_received), int(self._seq))


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

        self._flow = LucasKanadeDriftEstimator(
            motion_model=str(self._cfg.flow_motion_model),
            tr_residual_thresh_px=float(self._cfg.flow_tr_residual_thresh_px),
            tr_min_points=int(self._cfg.flow_tr_min_points),
        )
        self._kf = None
        if self._cfg.use_kalman:
            self._kf = VelocityKalman2D(sigma_a=self._cfg.kalman_sigma_a, sigma_v_meas=self._cfg.kalman_sigma_v)

        self._meas_cond = VelocityMeasurementConditioner(deadband_px_s=self._cfg.estimator_deadband_px_s)

        self._ctl_px = VelocityHoldController(
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

        self._ctl_m = VelocityHoldController(
            min_quality=self._cfg.min_quality,
            max_cmd=self._cfg.max_cmd,
            kp_vx=self._cfg.kp_vx_m_s,
            kp_vy=self._cfg.kp_vy_m_s,
            ki_vx=self._cfg.ki_vx_m_s,
            ki_vy=self._cfg.ki_vy_m_s,
            deadband_px_s=self._cfg.deadband_m_s,
            roll_sign=self._cfg.roll_sign,
            pitch_sign=self._cfg.pitch_sign,
        )

        self._visual_scale: Optional[VisualScaleEstimator] = None
        self._visual_scale_error: str = ""

        self._last_stable_scale_t: Optional[float] = None
        self._last_stable_m_per_px_x: Optional[float] = None
        self._last_stable_m_per_px_y: Optional[float] = None

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
        self._ctl_px.reset()
        self._ctl_m.reset()
        self._init_visual_scale()
        self._last_stable_scale_t = None
        self._last_stable_m_per_px_x = None
        self._last_stable_m_per_px_y = None
        self._last_loop_t = None
        self._loop_rate_hz_ema = 0.0
        self._frame_rate_hz_ema = 0.0
        self._active = True
        self._stop.clear()
        self._viz_x_px = 0.0
        self._viz_y_px = 0.0

    def _init_visual_scale(self) -> None:
        self._visual_scale = None
        self._visual_scale_error = ""
        if not bool(self._cfg.enable_visual_scale):
            return
        ref_id = self._cfg.reference_id
        if ref_id is None or not str(ref_id).strip():
            self._visual_scale_error = "no_reference_id"
            return

        store = ReferenceStore()
        record = store.load(str(ref_id))
        if record is None:
            self._visual_scale_error = "reference_record_not_found"
            return
        img = store.load_image_bgr(str(ref_id))
        if img is None:
            self._visual_scale_error = "reference_image_not_found"
            return

        self._visual_scale = VisualScaleEstimator(
            record=record,
            reference_image_bgr=img,
            detection_method=str(self._cfg.visual_scale_detection_method),
            stable_required_frames=int(self._cfg.visual_scale_stable_frames),
            max_ref_size_frac_per_sec=float(self._cfg.visual_scale_max_ref_size_frac_per_sec),
        )

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

    @staticmethod
    def _update_rate_ema(*, prev: float, inst: float, alpha: float = 0.15) -> float:
        p = float(prev)
        x = float(inst)
        a = float(alpha)
        if not np.isfinite(x) or x <= 0.0:
            return float(p)
        if (not np.isfinite(p)) or p <= 0.0:
            return float(x)
        if (not np.isfinite(a)) or a <= 0.0:
            return float(p)
        if a >= 1.0:
            return float(x)
        return float((1.0 - a) * p + a * x)

    def run(self, *, duration_sec: Optional[float] = None) -> None:
        if not self._active:
            self.activate()

        start = time.monotonic()
        frame_buf = LatestFrameBuffer(self._drone)
        frame_buf.start()
        last_used_frame_seq = -1
        try:
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
                            frames_dropped=int(frame_buf.dropped_total),
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
                            frames_dropped=int(frame_buf.dropped_total),
                        )
                    )
                    time.sleep(0.05)

                good_needed = max(0, int(self._cfg.settle_good_frames))
                good_seen = 0
                while not self._stop.is_set() and good_seen < good_needed:
                    t_loop_start = float(time.monotonic())
                    item = frame_buf.get_latest()
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
                                frames_dropped=int(frame_buf.dropped_total),
                                visual_scale_enabled=bool(self._cfg.enable_visual_scale),
                                visual_scale_error=str(self._visual_scale_error),
                            )
                        )
                        time.sleep(0.01)
                        continue

                    frame, ts, t_frame_received, frame_seq, dropped = item
                    frame_is_new = bool(int(frame_seq) != int(last_used_frame_seq))
                    last_used_frame_seq = int(frame_seq)

                    t_flow_start = float(time.monotonic())
                    frame_age_ms = float(max(0.0, (t_flow_start - float(ts)) * 1000.0))
                    frame_stale_ms = float(max(0.0, (t_flow_start - float(t_frame_received)) * 1000.0))
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
                                frame_stale_ms=float(frame_stale_ms),
                                frame_seq=int(frame_seq),
                                frame_is_new=bool(frame_is_new),
                                frames_dropped=int(dropped),
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

                    scale = None
                    vs_error = str(self._visual_scale_error)
                    if self._visual_scale is not None:
                        try:
                            scale = self._visual_scale.update(
                                frame_bgr=frame,
                                timestamp=float(ts),
                                vx_px_s=float(cond.vx_px_s),
                                vy_px_s=float(cond.vy_px_s),
                            )
                        except Exception as e:
                            scale = None
                            vs_error = f"{type(e).__name__}: {e}"

                    altitude_est_m = None if scale is None else scale.altitude_est_m
                    altitude_source = str("unknown" if scale is None else scale.altitude_source)
                    ref_detected = bool(False if scale is None else scale.ref_detected)
                    ref_width_px = float(0.0 if scale is None else scale.ref_width_px)
                    ref_height_px = float(0.0 if scale is None else scale.ref_height_px)
                    ref_size_px = float(0.0 if scale is None else scale.ref_size_px)
                    ref_quad_xy = None
                    if scale is not None and bool(scale.ref_detected) and scale.detection.quad_xy is not None:
                        ref_quad_xy = np.asarray(scale.detection.quad_xy, dtype=np.float32).copy()
                    vx_m_s = None if scale is None else scale.vx_m_s
                    vy_m_s = None if scale is None else scale.vy_m_s
                    scale_stable = bool(False if scale is None else scale.stable)
                    ref_mode = str("") if scale is None else str(scale.detection.mode)
                    ref_n_matches = int(0 if scale is None else scale.detection.n_matches)
                    ref_n_inliers = int(0 if scale is None else scale.detection.n_inliers)
                    ref_inlier_ratio = float(0.0 if scale is None else scale.detection.inlier_ratio)
                    ref_reproj_error_px = float(0.0 if scale is None else scale.detection.reproj_error_px)

                    t_ctrl_end = float(time.monotonic())
                    self._frame_rate_hz_ema = self._update_rate_ema(
                        prev=self._frame_rate_hz_ema, inst=(1.0 / float(est.dt_sec))
                    )
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
                            frame_stale_ms=float(frame_stale_ms),
                            frame_seq=int(frame_seq),
                            frame_is_new=bool(frame_is_new),
                            frames_dropped=int(dropped),
                            estimated_latency_ms=float(max(0.0, (t_cmd_sent - float(ts)) * 1000.0)),
                            loop_rate_hz=float(self._loop_rate_hz_ema),
                            frame_rate_hz=float(self._frame_rate_hz_ema),
                            visual_scale_enabled=bool(self._cfg.enable_visual_scale),
                            visual_scale_error=str(vs_error),
                            altitude_est_m=None if altitude_est_m is None else float(altitude_est_m),
                            altitude_source=str(altitude_source),
                            ref_detected=bool(ref_detected),
                            ref_width_px=float(ref_width_px),
                            ref_height_px=float(ref_height_px),
                            ref_size_px=float(ref_size_px),
                            ref_quad_xy=ref_quad_xy,
                            vx_m_s=None if vx_m_s is None else float(vx_m_s),
                            vy_m_s=None if vy_m_s is None else float(vy_m_s),
                            used_vx_m_s=None,
                            used_vy_m_s=None,
                            scale_stable=bool(scale_stable),
                            ref_mode=str(ref_mode),
                            ref_n_matches=int(ref_n_matches),
                            ref_n_inliers=int(ref_n_inliers),
                            ref_inlier_ratio=float(ref_inlier_ratio),
                            ref_reproj_error_px=float(ref_reproj_error_px),
                        )
                    )

            start = time.monotonic()

            period = 1.0 / max(1.0, float(self._cfg.cmd_rate_hz))

            while True:
                now = time.monotonic()
                if self._last_loop_t is not None:
                    dt_loop = float(now - float(self._last_loop_t))
                    if dt_loop > 1e-6:
                        self._loop_rate_hz_ema = self._update_rate_ema(
                            prev=self._loop_rate_hz_ema, inst=(1.0 / dt_loop)
                        )
                self._last_loop_t = float(now)

                t_loop_start = float(now)
                if self._stop.is_set():
                    break
                if duration_sec is not None and (now - start) >= float(duration_sec):
                    break

                item = frame_buf.get_latest()
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
                            frames_dropped=int(frame_buf.dropped_total),
                            visual_scale_enabled=bool(self._cfg.enable_visual_scale),
                            visual_scale_error=str(self._visual_scale_error),
                        )
                    )
                    time.sleep(period)
                    continue

                frame, ts, t_frame_received, frame_seq, dropped = item
                frame_is_new = bool(int(frame_seq) != int(last_used_frame_seq))
                last_used_frame_seq = int(frame_seq)

                t_flow_start = float(time.monotonic())
                frame_age_ms = float(max(0.0, (t_flow_start - float(ts)) * 1000.0))
                frame_stale_ms = float(max(0.0, (t_flow_start - float(t_frame_received)) * 1000.0))
                est = self._flow.update(frame, timestamp=float(time.monotonic()))
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
                            frame_stale_ms=float(frame_stale_ms),
                            frame_seq=int(frame_seq),
                            frame_is_new=bool(frame_is_new),
                            frames_dropped=int(dropped),
                            estimated_latency_ms=float(max(0.0, (t_cmd_sent - float(ts)) * 1000.0)),
                            loop_rate_hz=float(self._loop_rate_hz_ema),
                            frame_rate_hz=float(self._frame_rate_hz_ema),
                            visual_scale_enabled=bool(self._cfg.enable_visual_scale),
                            visual_scale_error=str(self._visual_scale_error),
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
                self._frame_rate_hz_ema = self._update_rate_ema(
                    prev=self._frame_rate_hz_ema, inst=(1.0 / float(est.dt_sec))
                )

                scale = None
                vs_error = str(self._visual_scale_error)
                if self._visual_scale is not None:
                    try:
                        scale = self._visual_scale.update(
                            frame_bgr=frame,
                            timestamp=float(ts),
                            vx_px_s=float(vx),
                            vy_px_s=float(vy),
                        )
                    except Exception as e:
                        scale = None
                        vs_error = f"{type(e).__name__}: {e}"

                altitude_est_m = None if scale is None else scale.altitude_est_m
                altitude_source = str("unknown" if scale is None else scale.altitude_source)
                ref_detected = bool(False if scale is None else scale.ref_detected)
                ref_width_px = float(0.0 if scale is None else scale.ref_width_px)
                ref_height_px = float(0.0 if scale is None else scale.ref_height_px)
                ref_size_px = float(0.0 if scale is None else scale.ref_size_px)
                ref_quad_xy = None
                if scale is not None and bool(scale.ref_detected) and scale.detection.quad_xy is not None:
                    ref_quad_xy = np.asarray(scale.detection.quad_xy, dtype=np.float32).copy()
                vx_m_s = None if scale is None else scale.vx_m_s
                vy_m_s = None if scale is None else scale.vy_m_s
                m_per_px_x = None if scale is None else scale.m_per_px_x
                m_per_px_y = None if scale is None else scale.m_per_px_y
                scale_stable = bool(False if scale is None else scale.stable)
                ref_mode = str("") if scale is None else str(scale.detection.mode)
                ref_n_matches = int(0 if scale is None else scale.detection.n_matches)
                ref_n_inliers = int(0 if scale is None else scale.detection.n_inliers)
                ref_inlier_ratio = float(0.0 if scale is None else scale.detection.inlier_ratio)
                ref_reproj_error_px = float(0.0 if scale is None else scale.detection.reproj_error_px)

                if bool(scale_stable) and (m_per_px_x is not None) and (m_per_px_y is not None):
                    self._last_stable_scale_t = float(ts)
                    self._last_stable_m_per_px_x = float(m_per_px_x)
                    self._last_stable_m_per_px_y = float(m_per_px_y)

                use_ms_hold = False
                vx_m_s_eff = vx_m_s
                vy_m_s_eff = vy_m_s
                hold_sec = float(self._cfg.visual_scale_ms_hold_sec)
                if hold_sec > 0.0 and (not bool(scale_stable)):
                    if (
                        self._last_stable_scale_t is not None
                        and self._last_stable_m_per_px_x is not None
                        and self._last_stable_m_per_px_y is not None
                        and (float(ts) - float(self._last_stable_scale_t)) <= hold_sec
                    ):
                        vx_m_s_eff = float(vx) * float(self._last_stable_m_per_px_x)
                        vy_m_s_eff = float(vy) * float(self._last_stable_m_per_px_y)
                        use_ms_hold = True

                use_m_s = (
                    bool(self._cfg.enable_visual_scale)
                    and bool(self._cfg.use_m_s_control)
                    and (bool(scale_stable) or bool(use_ms_hold))
                    and (vx_m_s_eff is not None)
                    and (vy_m_s_eff is not None)
                )

                out = None
                if bool(self._cfg.enable_visual_scale) and bool(self._cfg.use_m_s_control) and use_m_s:
                    out = self._ctl_m.update(
                        dt_sec=est.dt_sec,
                        vx_px_s=float(vy_m_s_eff),
                        vy_px_s=float(vx_m_s_eff),
                        quality=q,
                    )
                else:
                    out = self._ctl_px.update(dt_sec=est.dt_sec, vx_px_s=vy, vy_px_s=vx, quality=q)

                cmd_scale = 1.0
                if bool(self._cfg.enable_visual_scale) and bool(self._cfg.use_m_s_control) and (not use_m_s):
                    cmd_scale = float(self._cfg.visual_scale_unstable_cmd_scale)
                t_ctrl_end = float(time.monotonic())
                if out is None:
                    cmd_roll = 0.0
                    cmd_pitch = 0.0
                    self._drone.send_cmd(roll=float(cmd_roll), pitch=float(cmd_pitch), throttle=self._cfg.base_throttle)
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
                            cmd_roll=float(cmd_roll),
                            cmd_pitch=float(cmd_pitch),
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
                            frame_stale_ms=float(frame_stale_ms),
                            frame_seq=int(frame_seq),
                            frame_is_new=bool(frame_is_new),
                            frames_dropped=int(dropped),
                            estimated_latency_ms=float(max(0.0, (t_cmd_sent - float(ts)) * 1000.0)),
                            loop_rate_hz=float(self._loop_rate_hz_ema),
                            frame_rate_hz=float(self._frame_rate_hz_ema),
                            visual_scale_enabled=bool(self._cfg.enable_visual_scale),
                            visual_scale_error=str(vs_error),
                            altitude_est_m=None if altitude_est_m is None else float(altitude_est_m),
                            altitude_source=str(altitude_source),
                            ref_detected=bool(ref_detected),
                            ref_width_px=float(ref_width_px),
                            ref_height_px=float(ref_height_px),
                            ref_size_px=float(ref_size_px),
                            ref_quad_xy=ref_quad_xy,
                            vx_m_s=None if vx_m_s is None else float(vx_m_s),
                            vy_m_s=None if vy_m_s is None else float(vy_m_s),
                            used_vx_m_s=None,
                            used_vy_m_s=None,
                            scale_stable=bool(scale_stable),
                            ref_mode=str(ref_mode),
                            ref_n_matches=int(ref_n_matches),
                            ref_n_inliers=int(ref_n_inliers),
                            ref_inlier_ratio=float(ref_inlier_ratio),
                            ref_reproj_error_px=float(ref_reproj_error_px),
                        )
                    )
                else:
                    cmd_roll = float(out.roll) * float(cmd_scale)
                    cmd_pitch = float(out.pitch) * float(cmd_scale)
                    self._drone.send_cmd(roll=float(cmd_roll), pitch=float(cmd_pitch), throttle=self._cfg.base_throttle)
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
                            cmd_roll=float(cmd_roll),
                            cmd_pitch=float(cmd_pitch),
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
                            frame_stale_ms=float(frame_stale_ms),
                            frame_seq=int(frame_seq),
                            frame_is_new=bool(frame_is_new),
                            frames_dropped=int(dropped),
                            estimated_latency_ms=float(max(0.0, (t_cmd_sent - float(ts)) * 1000.0)),
                            loop_rate_hz=float(self._loop_rate_hz_ema),
                            frame_rate_hz=float(self._frame_rate_hz_ema),
                            altitude_est_m=None if altitude_est_m is None else float(altitude_est_m),
                            altitude_source=str(altitude_source),
                            ref_detected=bool(ref_detected),
                            ref_width_px=float(ref_width_px),
                            ref_height_px=float(ref_height_px),
                            ref_size_px=float(ref_size_px),
                            ref_quad_xy=ref_quad_xy,
                            vx_m_s=None if vx_m_s is None else float(vx_m_s),
                            vy_m_s=None if vy_m_s is None else float(vy_m_s),
                            used_vx_m_s=None if (not use_m_s or vx_m_s_eff is None) else float(vx_m_s_eff),
                            used_vy_m_s=None if (not use_m_s or vy_m_s_eff is None) else float(vy_m_s_eff),
                            scale_stable=bool(scale_stable),
                            ref_mode=str(ref_mode),
                            ref_n_matches=int(ref_n_matches),
                            ref_n_inliers=int(ref_n_inliers),
                            ref_inlier_ratio=float(ref_inlier_ratio),
                            ref_reproj_error_px=float(ref_reproj_error_px),
                            visual_scale_enabled=bool(self._cfg.enable_visual_scale),
                            visual_scale_error=str(vs_error),
                        )
                    )

                elapsed = time.monotonic() - now
                if elapsed < period:
                    time.sleep(period - elapsed)
        finally:
            frame_buf.stop()
