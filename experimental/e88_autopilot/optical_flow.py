from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class FlowEstimate:
    dt_sec: float
    dx_px: float
    dy_px: float
    vx_px_s: float
    vy_px_s: float
    quality: float
    n_features: int
    n_tracked: int
    inlier_ratio: float
    fallback_used: bool
    motion_model: str = ""
    raw_dx_px: float = 0.0
    raw_dy_px: float = 0.0
    omega_rad: float = 0.0
    omega_rad_s: float = 0.0
    model_rmse_px: float = 0.0


@dataclass(frozen=True)
class FlowTracks:
    prev_xy: np.ndarray  # shape (N, 2)
    next_xy: np.ndarray  # shape (N, 2)
    inliers: np.ndarray  # shape (N,)


@dataclass(frozen=True)
class TranslationRotationFit:
    tx_px: float
    ty_px: float
    omega_rad: float
    rmse_px: float
    n_used: int
    inlier_ratio: float


class TranslationRotationMotionModel:
    def __init__(self, *, residual_thresh_px: float = 3.0, min_points: int = 20) -> None:
        self._residual_thresh_px = float(residual_thresh_px)
        self._min_points = int(min_points)

    def estimate(
        self, *, prev_xy: np.ndarray, next_xy: np.ndarray, inliers: Optional[np.ndarray]
    ) -> Optional[TranslationRotationFit]:
        if prev_xy.ndim != 2 or prev_xy.shape[1] != 2 or next_xy.ndim != 2 or next_xy.shape[1] != 2:
            return None

        if prev_xy.shape[0] != next_xy.shape[0]:
            return None

        if prev_xy.shape[0] < self._min_points:
            return None

        if inliers is None:
            mask = np.ones((prev_xy.shape[0],), dtype=np.bool_)
        else:
            mask = inliers.reshape(-1).astype(np.bool_)

        prev = prev_xy[mask]
        nxt = next_xy[mask]
        if prev.shape[0] < self._min_points:
            return None

        d = (nxt - prev).astype(np.float64)
        p = prev.astype(np.float64)

        cx = float(np.median(p[:, 0]))
        cy = float(np.median(p[:, 1]))
        x = p[:, 0] - cx
        y = p[:, 1] - cy

        n = int(p.shape[0])
        a = np.zeros((2 * n, 3), dtype=np.float64)
        b = np.zeros((2 * n,), dtype=np.float64)

        a[0::2, 0] = 1.0
        a[0::2, 2] = -y
        b[0::2] = d[:, 0]

        a[1::2, 1] = 1.0
        a[1::2, 2] = x
        b[1::2] = d[:, 1]

        sol, *_ = np.linalg.lstsq(a, b, rcond=None)
        tx, ty, omega = float(sol[0]), float(sol[1]), float(sol[2])

        pred_dx = tx - omega * y
        pred_dy = ty + omega * x
        r = np.sqrt((d[:, 0] - pred_dx) ** 2 + (d[:, 1] - pred_dy) ** 2)

        used = np.ones((n,), dtype=np.bool_)
        if self._residual_thresh_px > 0.0:
            used = r <= float(self._residual_thresh_px)
            if int(np.count_nonzero(used)) >= self._min_points and int(np.count_nonzero(used)) < n:
                d2 = d[used]
                p2 = p[used]
                cx2 = float(np.median(p2[:, 0]))
                cy2 = float(np.median(p2[:, 1]))
                x2 = p2[:, 0] - cx2
                y2 = p2[:, 1] - cy2

                n2 = int(p2.shape[0])
                a2 = np.zeros((2 * n2, 3), dtype=np.float64)
                b2 = np.zeros((2 * n2,), dtype=np.float64)

                a2[0::2, 0] = 1.0
                a2[0::2, 2] = -y2
                b2[0::2] = d2[:, 0]

                a2[1::2, 1] = 1.0
                a2[1::2, 2] = x2
                b2[1::2] = d2[:, 1]

                sol2, *_ = np.linalg.lstsq(a2, b2, rcond=None)
                tx, ty, omega = float(sol2[0]), float(sol2[1]), float(sol2[2])

                pred_dx2 = tx - omega * y2
                pred_dy2 = ty + omega * x2
                r2 = np.sqrt((d2[:, 0] - pred_dx2) ** 2 + (d2[:, 1] - pred_dy2) ** 2)
                rmse = float(np.sqrt(float(np.mean(r2**2))))
                return TranslationRotationFit(
                    tx_px=float(tx),
                    ty_px=float(ty),
                    omega_rad=float(omega),
                    rmse_px=float(rmse),
                    n_used=int(n2),
                    inlier_ratio=float(n2 / max(1, n)),
                )

        rmse = float(np.sqrt(float(np.mean((r[used]) ** 2))))
        return TranslationRotationFit(
            tx_px=float(tx),
            ty_px=float(ty),
            omega_rad=float(omega),
            rmse_px=float(rmse),
            n_used=int(np.count_nonzero(used)),
            inlier_ratio=float(np.count_nonzero(used) / max(1, n)),
        )


class LucasKanadeDriftEstimator:
    def __init__(
        self,
        *,
        max_corners: int = 250,
        quality_level: float = 0.01,
        min_distance: float = 7.0,
        block_size: int = 7,
        win_size: Tuple[int, int] = (21, 21),
        max_level: int = 3,
        min_tracked_features: int = 40,
        reinit_every_n_frames: int = 60,
        downscale: float = 0.5,
        max_translation_frac: float = 0.25,
        min_inlier_ratio: float = 0.6,
        enable_phase_corr_fallback: bool = True,
        phase_corr_min_response: float = 0.2,
        phase_corr_max_translation_frac: float = 0.6,
        motion_model: str = "affine_translation",
        tr_residual_thresh_px: float = 3.0,
        tr_min_points: int = 20,
    ) -> None:
        self._feature_params = dict(
            maxCorners=int(max_corners),
            qualityLevel=float(quality_level),
            minDistance=float(min_distance),
            blockSize=int(block_size),
        )
        self._lk_params = dict(
            winSize=tuple(map(int, win_size)),
            maxLevel=int(max_level),
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        self._min_tracked_features = int(min_tracked_features)
        self._reinit_every_n_frames = int(reinit_every_n_frames)
        self._downscale = float(downscale)
        self._max_translation_frac = float(max_translation_frac)
        self._min_inlier_ratio = float(min_inlier_ratio)

        self._enable_phase_corr_fallback = bool(enable_phase_corr_fallback)
        self._phase_corr_min_response = float(phase_corr_min_response)
        self._phase_corr_max_translation_frac = float(phase_corr_max_translation_frac)

        self._motion_model = str(motion_model)
        self._tr_model = TranslationRotationMotionModel(
            residual_thresh_px=float(tr_residual_thresh_px),
            min_points=int(tr_min_points),
        )

        self._prev_gray: Optional[np.ndarray] = None
        self._prev_pts: Optional[np.ndarray] = None
        self._prev_ts: Optional[float] = None
        self._frames_since_init = 0

        self._last_tracks: Optional[FlowTracks] = None

    @staticmethod
    def _rmse_affine_px(
        *,
        prev_xy: np.ndarray,
        next_xy: np.ndarray,
        m: Optional[np.ndarray],
        inliers: Optional[np.ndarray],
    ) -> float:
        if m is None:
            return 0.0
        if prev_xy is None or next_xy is None:
            return 0.0
        prev = prev_xy.reshape(-1, 2).astype(np.float64)
        nxt = next_xy.reshape(-1, 2).astype(np.float64)
        if prev.shape[0] != nxt.shape[0] or prev.shape[0] == 0:
            return 0.0

        if inliers is None:
            mask = np.ones((prev.shape[0],), dtype=np.bool_)
        else:
            mask = inliers.reshape(-1).astype(np.bool_)
            if mask.shape[0] != prev.shape[0]:
                mask = np.ones((prev.shape[0],), dtype=np.bool_)

        prev_u = prev[mask]
        nxt_u = nxt[mask]
        if prev_u.shape[0] == 0:
            return 0.0

        mm = np.asarray(m, dtype=np.float64)
        if mm.shape != (2, 3):
            return 0.0

        ones = np.ones((prev_u.shape[0], 1), dtype=np.float64)
        prev_h = np.concatenate([prev_u, ones], axis=1)
        pred = prev_h @ mm.T
        r = nxt_u - pred
        e2 = np.sum(r * r, axis=1)
        return float(np.sqrt(float(np.mean(e2))))

    def reset(self) -> None:
        self._prev_gray = None
        self._prev_pts = None
        self._prev_ts = None
        self._frames_since_init = 0
        self._last_tracks = None

    def last_tracks(self) -> Optional[FlowTracks]:
        t = self._last_tracks
        if t is None:
            return None
        return FlowTracks(prev_xy=t.prev_xy.copy(), next_xy=t.next_xy.copy(), inliers=t.inliers.copy())

    def update(self, frame_bgr: np.ndarray, timestamp: Optional[float] = None) -> Optional[FlowEstimate]:
        ts = float(time.monotonic() if timestamp is None else timestamp)

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self._downscale != 1.0:
            gray = cv2.resize(gray, (0, 0), fx=self._downscale, fy=self._downscale, interpolation=cv2.INTER_AREA)

        if self._prev_gray is None or self._prev_pts is None or self._prev_ts is None:
            self._initialize(gray)
            self._prev_ts = ts
            self._last_tracks = None
            return None

        dt = ts - self._prev_ts
        if dt <= 1e-6:
            return None

        scale = (1.0 / self._downscale) if self._downscale != 0.0 else 1.0

        next_pts, status, _err = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, self._prev_pts, None, **self._lk_params)
        if next_pts is None or status is None:
            fallback = self._fallback_phase_correlation(prev_gray=self._prev_gray, gray=gray, dt=float(dt))
            if fallback is not None:
                dx_px, dy_px, quality = fallback
                dx_px *= scale
                dy_px *= scale
                self._initialize(gray)
                self._prev_ts = ts
                self._last_tracks = None
                return FlowEstimate(
                    dt_sec=float(dt),
                    dx_px=float(dx_px),
                    dy_px=float(dy_px),
                    vx_px_s=float(dx_px / dt),
                    vy_px_s=float(dy_px / dt),
                    quality=float(quality),
                    n_features=0,
                    n_tracked=0,
                    inlier_ratio=0.0,
                    fallback_used=True,
                    motion_model=str("phase_corr"),
                    raw_dx_px=float(dx_px),
                    raw_dy_px=float(dy_px),
                )

            self._initialize(gray)
            self._prev_ts = ts
            self._last_tracks = None
            return None

        good = status.reshape(-1) == 1
        prev_good = self._prev_pts[good]
        next_good = next_pts[good]

        n_features = int(self._prev_pts.shape[0])
        n_tracked = int(prev_good.shape[0])

        dx_px = 0.0
        dy_px = 0.0

        inliers = np.ones((n_tracked,), dtype=np.bool_)

        inlier_ratio = 1.0

        affine_m: Optional[np.ndarray] = None
        if n_tracked >= 4:
            if self._motion_model == "affine_full":
                affine_m, inliers = cv2.estimateAffine2D(
                    prev_good,
                    next_good,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=3.0,
                )
            else:
                affine_m, inliers = cv2.estimateAffinePartial2D(
                    prev_good,
                    next_good,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=3.0,
                )

            if affine_m is not None:
                dx_px = float(affine_m[0, 2])
                dy_px = float(affine_m[1, 2])
            else:
                flow = (next_good - prev_good).reshape(-1, 2)
                dx_px = float(np.median(flow[:, 0]))
                dy_px = float(np.median(flow[:, 1]))
                inliers = np.ones((n_tracked,), dtype=np.bool_)
                affine_m = np.array([[1.0, 0.0, dx_px], [0.0, 1.0, dy_px]], dtype=np.float64)
        elif n_tracked > 0:
            flow = (next_good - prev_good).reshape(-1, 2)
            dx_px = float(np.median(flow[:, 0]))
            dy_px = float(np.median(flow[:, 1]))
            inliers = np.ones((n_tracked,), dtype=np.bool_)
            affine_m = np.array([[1.0, 0.0, dx_px], [0.0, 1.0, dy_px]], dtype=np.float64)

        if inliers is None:
            inliers = np.ones((n_tracked,), dtype=np.bool_)
        else:
            inliers = inliers.reshape(-1).astype(np.bool_)

        if n_tracked > 0:
            inlier_ratio = float(np.count_nonzero(inliers)) / float(n_tracked)

        max_frac = max(0.01, float(self._max_translation_frac))
        max_step_px = max_frac * float(min(gray.shape[0], gray.shape[1]))
        if abs(dx_px) > max_step_px or abs(dy_px) > max_step_px or inlier_ratio < float(self._min_inlier_ratio):
            fallback = self._fallback_phase_correlation(prev_gray=self._prev_gray, gray=gray, dt=float(dt))
            if fallback is not None:
                dx_px, dy_px, quality = fallback
                dx_px *= scale
                dy_px *= scale
                self._initialize(gray)
                self._prev_ts = ts
                self._last_tracks = None
                return FlowEstimate(
                    dt_sec=float(dt),
                    dx_px=float(dx_px),
                    dy_px=float(dy_px),
                    vx_px_s=float(dx_px / dt),
                    vy_px_s=float(dy_px / dt),
                    quality=float(quality),
                    n_features=n_features,
                    n_tracked=n_tracked,
                    inlier_ratio=float(inlier_ratio),
                    fallback_used=True,
                    motion_model=str("phase_corr"),
                    raw_dx_px=float(dx_px),
                    raw_dy_px=float(dy_px),
                )

            self._initialize(gray)
            self._prev_ts = ts
            self._last_tracks = None
            return None

        raw_dx_px = float(dx_px) * scale
        raw_dy_px = float(dy_px) * scale
        omega_rad = 0.0
        omega_rad_s = 0.0
        model_rmse_px = float(self._rmse_affine_px(prev_xy=prev_good, next_xy=next_good, m=affine_m, inliers=inliers)) * scale
        motion_model = str("affine_full" if self._motion_model == "affine_full" else "affine_translation")

        if self._motion_model == "translation_rotation" and n_tracked >= 4:
            fit = self._tr_model.estimate(
                prev_xy=prev_good.reshape(-1, 2).astype(np.float32),
                next_xy=next_good.reshape(-1, 2).astype(np.float32),
                inliers=inliers,
            )
            if fit is not None:
                dx_px = float(fit.tx_px)
                dy_px = float(fit.ty_px)
                omega_rad = float(fit.omega_rad)
                omega_rad_s = float(omega_rad / float(dt))
                model_rmse_px = float(fit.rmse_px) * scale
                motion_model = str("translation_rotation")

        dx_px *= scale
        dy_px *= scale
        prev_xy = prev_good.reshape(-1, 2).astype(np.float32) * scale
        next_xy = next_good.reshape(-1, 2).astype(np.float32) * scale
        self._last_tracks = FlowTracks(prev_xy=prev_xy, next_xy=next_xy, inliers=inliers)

        vx = dx_px / dt
        vy = dy_px / dt

        tracked_ratio = (n_tracked / max(1, n_features))
        quality = float(max(0.0, min(1.0, tracked_ratio * inlier_ratio)))

        self._prev_gray = gray
        self._prev_pts = next_good.reshape(-1, 1, 2)
        self._prev_ts = ts
        self._frames_since_init += 1

        needs_reinit = (
            n_tracked < self._min_tracked_features
            or (self._reinit_every_n_frames > 0 and self._frames_since_init >= self._reinit_every_n_frames)
        )
        if needs_reinit:
            self._initialize(gray)
            self._prev_ts = ts

        return FlowEstimate(
            dt_sec=float(dt),
            dx_px=float(dx_px),
            dy_px=float(dy_px),
            vx_px_s=float(vx),
            vy_px_s=float(vy),
            quality=quality,
            n_features=n_features,
            n_tracked=n_tracked,
            inlier_ratio=float(inlier_ratio),
            fallback_used=False,
            motion_model=str(motion_model),
            raw_dx_px=float(raw_dx_px),
            raw_dy_px=float(raw_dy_px),
            omega_rad=float(omega_rad),
            omega_rad_s=float(omega_rad_s),
            model_rmse_px=float(model_rmse_px),
        )

    def _initialize(self, gray: np.ndarray) -> None:
        pts = cv2.goodFeaturesToTrack(gray, mask=None, **self._feature_params)
        if pts is None:
            self._prev_pts = None
        else:
            self._prev_pts = pts.astype(np.float32)
        self._prev_gray = gray
        self._frames_since_init = 0

    def _fallback_phase_correlation(
        self, *, prev_gray: np.ndarray, gray: np.ndarray, dt: float
    ) -> Optional[Tuple[float, float, float]]:
        if not self._enable_phase_corr_fallback:
            return None
        if dt <= 1e-6:
            return None

        a = prev_gray.astype(np.float32)
        b = gray.astype(np.float32)

        (dx_px, dy_px), response = cv2.phaseCorrelate(a, b)
        q = float(response)
        if q < float(self._phase_corr_min_response):
            return None

        max_frac = max(0.01, float(self._phase_corr_max_translation_frac))
        max_step_px = max_frac * float(min(gray.shape[0], gray.shape[1]))
        if abs(float(dx_px)) > max_step_px or abs(float(dy_px)) > max_step_px:
            return None

        return float(dx_px), float(dy_px), float(max(0.0, min(1.0, q)))
