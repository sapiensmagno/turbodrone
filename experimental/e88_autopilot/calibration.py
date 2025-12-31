from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Protocol, Tuple

import numpy as np

from e88_autopilot.optical_flow import LucasKanadeDriftEstimator


class FrameSource(Protocol):
    def get_frame_with_timestamp(self, timeout: float) -> Optional[Tuple[np.ndarray, float]]: ...


@dataclass(frozen=True)
class StationaryCalibrationResult:
    created_at_ts: float
    duration_sec: float
    min_quality: float
    percentile: float
    n_samples: int
    estimator_deadband_px_s: float
    kalman_sigma_v: float


DEFAULT_CALIBRATION_PATH = Path(__file__).resolve().parent / "calibration.json"


def run_stationary_calibration(
    source: FrameSource,
    *,
    duration_sec: float,
    min_quality: float,
    percentile: float = 99.0,
    sigma_scale: float = 1.0,
) -> StationaryCalibrationResult:
    dur = float(max(0.1, duration_sec))
    qmin = float(min_quality)
    p = float(percentile)
    if not (50.0 <= p <= 100.0):
        raise ValueError(f"percentile must be in [50, 100], got {percentile}")

    print(
        f"[calibration] start stationary calibration: duration={dur:.1f}s min_quality={qmin:.2f} percentile={p:.1f} sigma_scale={float(sigma_scale):.2f}"
    )

    flow = LucasKanadeDriftEstimator()
    t0 = time.monotonic()

    last_report = t0
    n_frames = 0
    n_flow_ok = 0
    n_quality_ok = 0

    vx_s: list[float] = []
    vy_s: list[float] = []

    while (time.monotonic() - t0) < dur:
        item = source.get_frame_with_timestamp(timeout=2.0)
        if item is None:
            continue
        frame, ts = item
        n_frames += 1
        est = flow.update(frame, timestamp=ts)
        if est is None:
            continue
        n_flow_ok += 1
        if float(est.quality) < qmin:
            continue
        n_quality_ok += 1
        vx_s.append(float(est.vx_px_s))
        vy_s.append(float(est.vy_px_s))

        now = time.monotonic()
        if (now - last_report) >= 1.0:
            last_report = now
            print(
                f"[calibration] t={now - t0:.1f}s frames={n_frames} flow_ok={n_flow_ok} q_ok={n_quality_ok} samples={len(vx_s)}"
            )

    if len(vx_s) < 20:
        raise RuntimeError(f"not enough samples for calibration: got {len(vx_s)}")

    vx = np.asarray(vx_s, dtype=np.float64)
    vy = np.asarray(vy_s, dtype=np.float64)

    dbx = float(np.percentile(np.abs(vx), p))
    dby = float(np.percentile(np.abs(vy), p))
    deadband = float(max(dbx, dby))

    sx = _robust_sigma(vx)
    sy = _robust_sigma(vy)
    sigma_v = float(np.sqrt((sx * sx + sy * sy) / 2.0))
    sigma_v = float(max(1e-3, sigma_v * float(sigma_scale)))

    print(
        f"[calibration] done: samples={len(vx_s)} est_deadband={deadband:.4f} px/s sigma_v={sigma_v:.4f} px/s"
    )

    return StationaryCalibrationResult(
        created_at_ts=float(time.time()),
        duration_sec=float(dur),
        min_quality=float(qmin),
        percentile=float(p),
        n_samples=int(vx.shape[0]),
        estimator_deadband_px_s=float(deadband),
        kalman_sigma_v=float(sigma_v),
    )


def save_calibration(result: StationaryCalibrationResult, *, path: Path = DEFAULT_CALIBRATION_PATH) -> None:
    data = {
        "version": 1,
        "result": asdict(result),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def load_calibration(*, path: Path = DEFAULT_CALIBRATION_PATH) -> Optional[StationaryCalibrationResult]:
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return None
    if int(raw.get("version", 0)) != 1:
        return None
    r = raw.get("result")
    if not isinstance(r, dict):
        return None
    try:
        return StationaryCalibrationResult(
            created_at_ts=float(r["created_at_ts"]),
            duration_sec=float(r["duration_sec"]),
            min_quality=float(r["min_quality"]),
            percentile=float(r["percentile"]),
            n_samples=int(r["n_samples"]),
            estimator_deadband_px_s=float(r["estimator_deadband_px_s"]),
            kalman_sigma_v=float(r["kalman_sigma_v"]),
        )
    except Exception:
        return None


def _robust_sigma(v: np.ndarray) -> float:
    x = np.asarray(v, dtype=np.float64).reshape(-1)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return float(1.4826 * mad)
