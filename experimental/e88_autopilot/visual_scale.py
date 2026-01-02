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


class VisualScaleEstimator:
    def __init__(
        self,
        *,
        record: ReferenceRecord,
        reference_image_bgr: np.ndarray,
        stable_required_frames: int = 8,
        max_ref_size_frac_per_sec: float = 2.0,
    ) -> None:
        self._record = record
        self._detector = ReferenceDetector(reference_bgr=reference_image_bgr, use_markers=bool(record.markers_present))

        self._stable_required_frames = int(max(1, stable_required_frames))
        self._max_ref_size_frac_per_sec = float(max(0.01, max_ref_size_frac_per_sec))

        self._last_t: Optional[float] = None
        self._last_ref_size_px: Optional[float] = None
        self._last_altitude_est_m: Optional[float] = None
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
            if alt is not None:
                self._last_altitude_est_m = float(alt)

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
                altitude_est_m=None if alt is None else float(alt),
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
