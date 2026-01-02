from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from e88_autopilot.reference_detection import ReferenceDetectionResult, ReferenceDetector
from e88_autopilot.reference_store import ReferenceRecord


@dataclass(frozen=True)
class VisualScaleEstimate:
    timestamp: float
    ref_detected: bool
    ref_width_px: float
    ref_height_px: float
    ref_size_px: float
    altitude_est_m: Optional[float]
    altitude_source: str
    vx_m_s: Optional[float]
    vy_m_s: Optional[float]
    m_per_px_x: Optional[float]
    m_per_px_y: Optional[float]
    stable: bool
    detection: ReferenceDetectionResult


def compute_m_per_px(*, pad_width_m: float, pad_height_m: float, ref_width_px: float, ref_height_px: float) -> tuple[Optional[float], Optional[float]]:
    wpx = float(ref_width_px)
    hpx = float(ref_height_px)
    if wpx <= 1e-6 or hpx <= 1e-6:
        return None, None
    mw = float(pad_width_m) / wpx
    mh = float(pad_height_m) / hpx
    if not np.isfinite(mw) or not np.isfinite(mh) or mw <= 0.0 or mh <= 0.0:
        return None, None
    return float(mw), float(mh)


def compute_altitude_est_m(
    *, calibration_height_m: Optional[float], calibration_ref_size_px: Optional[float], ref_size_px: float
) -> Optional[float]:
    if calibration_height_m is None or calibration_ref_size_px is None:
        return None
    h = float(calibration_height_m)
    s0 = float(calibration_ref_size_px)
    s = float(ref_size_px)
    if h <= 1e-6 or s0 <= 1e-6 or s <= 1e-6:
        return None
    z = h * (s0 / s)
    if not np.isfinite(z) or z <= 0.0:
        return None
    return float(z)


@dataclass
class AltitudeSmootherConfig:
    tau_sec: float = 0.7
    dt_cap_sec: float = 0.35
    max_rate_m_s: float = 0.8
    min_m: float = 0.7
    max_m: float = 2.0


class AltitudeSmoother:
    def __init__(self, *, cfg: Optional[AltitudeSmootherConfig] = None) -> None:
        self._cfg = AltitudeSmootherConfig() if cfg is None else cfg
        self._t_last: Optional[float] = None
        self._z_ema: Optional[float] = None

    def reset(self) -> None:
        self._t_last = None
        self._z_ema = None

    @property
    def value(self) -> Optional[float]:
        return self._z_ema

    def update(self, *, timestamp: float, altitude_meas_m: Optional[float]) -> Optional[float]:
        ts = float(timestamp)
        z = None if altitude_meas_m is None else float(altitude_meas_m)

        if z is None or (not np.isfinite(z)):
            self._t_last = float(ts) if self._t_last is None else float(self._t_last)
            return self._z_ema

        z = float(np.clip(z, float(self._cfg.min_m), float(self._cfg.max_m)))

        if self._z_ema is None:
            self._z_ema = float(z)
            self._t_last = float(ts)
            return self._z_ema

        dt = 0.0
        if self._t_last is not None:
            dt = float(ts - float(self._t_last))
        self._t_last = float(ts)

        if (not np.isfinite(dt)) or dt <= 0.0:
            dt = 0.0

        dt_eff = float(min(dt, float(self._cfg.dt_cap_sec)))

        dz = float(z - float(self._z_ema))
        max_dz = float(self._cfg.max_rate_m_s) * float(dt_eff)
        if max_dz > 0.0:
            dz = float(np.clip(dz, -max_dz, +max_dz))
        z_rl = float(float(self._z_ema) + dz)

        tau = float(self._cfg.tau_sec)
        if tau <= 1e-6:
            alpha = 1.0
        else:
            alpha = float(1.0 - float(np.exp(-float(dt_eff) / tau)))
        alpha = float(np.clip(alpha, 0.0, 1.0))

        self._z_ema = float((1.0 - alpha) * float(self._z_ema) + alpha * float(z_rl))
        return self._z_ema


class VisualScaleEstimator:
    def __init__(
        self,
        *,
        record: ReferenceRecord,
        reference_image_bgr: np.ndarray,
        stable_required_frames: int = 8,
        max_ref_size_frac_per_sec: float = 2.0,
        altitude_smoother_cfg: Optional[AltitudeSmootherConfig] = None,
    ) -> None:
        self._record = record
        self._detector = ReferenceDetector(reference_bgr=reference_image_bgr, use_markers=bool(record.markers_present))

        self._stable_required_frames = int(max(1, stable_required_frames))
        self._max_ref_size_frac_per_sec = float(max(0.01, max_ref_size_frac_per_sec))

        self._last_t: Optional[float] = None
        self._last_ref_size_px: Optional[float] = None
        self._last_altitude_est_m: Optional[float] = None
        self._altitude_smoother = AltitudeSmoother(cfg=altitude_smoother_cfg)
        self._stable_seen = 0

    @property
    def record(self) -> ReferenceRecord:
        return self._record

    def update(self, *, frame_bgr: np.ndarray, timestamp: float, vx_px_s: float, vy_px_s: float) -> VisualScaleEstimate:
        ts = float(timestamp)
        det = self._detector.detect(frame_bgr)

        accepted = bool(det.detected) and float(det.ref_size_px) > 1e-6

        if accepted and self._last_t is not None and self._last_ref_size_px is not None:
            dt = float(ts - float(self._last_t))
            if dt > 1e-6:
                prev = float(self._last_ref_size_px)
                cur = float(det.ref_size_px)
                frac_per_sec = abs(cur - prev) / max(1e-6, prev) / dt
                if frac_per_sec > self._max_ref_size_frac_per_sec:
                    accepted = False

        if accepted:
            self._stable_seen = min(self._stable_required_frames, int(self._stable_seen) + 1)
            self._last_t = float(ts)
            self._last_ref_size_px = float(det.ref_size_px)

            alt = compute_altitude_est_m(
                calibration_height_m=self._record.calibration_height_m,
                calibration_ref_size_px=self._record.calibration_ref_size_px,
                ref_size_px=float(det.ref_size_px),
            )
            alt_smoothed = self._altitude_smoother.update(timestamp=float(ts), altitude_meas_m=alt)
            if alt_smoothed is not None:
                self._last_altitude_est_m = float(alt_smoothed)

            m_per_px_x, m_per_px_y = compute_m_per_px(
                pad_width_m=self._record.pad_width_m,
                pad_height_m=self._record.pad_height_m,
                ref_width_px=float(det.ref_width_px),
                ref_height_px=float(det.ref_height_px),
            )

            vx_m_s = None if m_per_px_x is None else float(vx_px_s) * float(m_per_px_x)
            vy_m_s = None if m_per_px_y is None else float(vy_px_s) * float(m_per_px_y)

            alt_source = "reference_object" if alt is not None else "unknown"

            return VisualScaleEstimate(
                timestamp=float(ts),
                ref_detected=True,
                ref_width_px=float(det.ref_width_px),
                ref_height_px=float(det.ref_height_px),
                ref_size_px=float(det.ref_size_px),
                altitude_est_m=None if self._last_altitude_est_m is None else float(self._last_altitude_est_m),
                altitude_source=str(alt_source),
                vx_m_s=vx_m_s,
                vy_m_s=vy_m_s,
                m_per_px_x=m_per_px_x,
                m_per_px_y=m_per_px_y,
                stable=bool(self._stable_seen >= self._stable_required_frames),
                detection=det,
            )

        self._stable_seen = 0

        if self._last_altitude_est_m is not None:
            alt_source = "last_known"
        else:
            alt_source = "unknown"

        return VisualScaleEstimate(
            timestamp=float(ts),
            ref_detected=False,
            ref_width_px=0.0,
            ref_height_px=0.0,
            ref_size_px=0.0,
            altitude_est_m=None if self._last_altitude_est_m is None else float(self._last_altitude_est_m),
            altitude_source=str(alt_source),
            vx_m_s=None,
            vy_m_s=None,
            m_per_px_x=None,
            m_per_px_y=None,
            stable=False,
            detection=det,
        )
