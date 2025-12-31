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


@dataclass(frozen=True)
class FlowTracks:
    prev_xy: np.ndarray  # shape (N, 2)
    next_xy: np.ndarray  # shape (N, 2)
    inliers: np.ndarray  # shape (N,)


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

        self._prev_gray: Optional[np.ndarray] = None
        self._prev_pts: Optional[np.ndarray] = None
        self._prev_ts: Optional[float] = None
        self._frames_since_init = 0

        self._last_tracks: Optional[FlowTracks] = None

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

        if n_tracked >= 4:
            m, inliers = cv2.estimateAffinePartial2D(prev_good, next_good, method=cv2.RANSAC, ransacReprojThreshold=3.0)
            if m is not None:
                dx_px = float(m[0, 2])
                dy_px = float(m[1, 2])
            else:
                flow = (next_good - prev_good).reshape(-1, 2)
                dx_px = float(np.median(flow[:, 0]))
                dy_px = float(np.median(flow[:, 1]))
                inliers = np.ones((n_tracked,), dtype=np.bool_)
        elif n_tracked > 0:
            flow = (next_good - prev_good).reshape(-1, 2)
            dx_px = float(np.median(flow[:, 0]))
            dy_px = float(np.median(flow[:, 1]))
            inliers = np.ones((n_tracked,), dtype=np.bool_)

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
                )

            self._initialize(gray)
            self._prev_ts = ts
            self._last_tracks = None
            return None

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
