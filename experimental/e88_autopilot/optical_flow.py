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
    omega_x_rad: float = 0.0
    omega_y_rad: float = 0.0
    omega_z_rad: float = 0.0


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


@dataclass(frozen=True)
class DerotationFit:
    vx_px: float
    vy_px: float
    omega_x_rad: float
    omega_y_rad: float
    omega_z_rad: float
    rmse_px: float
    n_used: int
    inlier_ratio: float


class DerotationMotionModel:
    def __init__(
        self,
        *,
        focal_length_px: float,
        residual_thresh_px: float = 3.0,
        min_points: int = 20,
        max_condition_number: float = 5000.0,
    ) -> None:
        self._f = float(focal_length_px)
        self._residual_thresh_px = float(residual_thresh_px)
        self._min_points = int(min_points)
        # Measured condition numbers for this design matrix, 200 features, at the
        # real downscaled geometry (320x240, f=183):
        #     full-frame support            ~1300
        #     features in central 50%       ~4150
        #     features in central 35%       ~8700
        #     features in central 25%      ~18400
        #     full frame but f 2x too large ~7700
        # The default admits full-frame and moderately concentrated support while
        # rejecting the regimes where translation and tilt stop being separable.
        # <= 0 disables the check (for tests that probe the raw algebra).
        self._max_condition_number = float(max_condition_number)

    @property
    def focal_length_px(self) -> float:
        return self._f

    def estimate(
        self,
        *,
        prev_xy: np.ndarray,
        next_xy: np.ndarray,
        inliers: Optional[np.ndarray],
        cx: float,
        cy: float,
    ) -> Optional[DerotationFit]:
        if prev_xy.ndim != 2 or prev_xy.shape[1] != 2:
            return None
        if next_xy.ndim != 2 or next_xy.shape[1] != 2:
            return None
        if prev_xy.shape[0] != next_xy.shape[0]:
            return None
        if prev_xy.shape[0] < self._min_points:
            return None

        if inliers is None:
            mask = np.ones((prev_xy.shape[0],), dtype=np.bool_)
        else:
            mask = inliers.reshape(-1).astype(np.bool_)

        prev = prev_xy[mask].astype(np.float64)
        nxt = next_xy[mask].astype(np.float64)
        if prev.shape[0] < self._min_points:
            return None

        return self._solve(prev, nxt, cx, cy, prev_xy.shape[0])

    def _fit_pass(
        self, prev: np.ndarray, nxt: np.ndarray, cx: float, cy: float
    ) -> Optional[Tuple[Tuple[float, float, float, float, float], np.ndarray]]:
        """Solve one least-squares pass. Returns ((Vx, Vy, wx, wy, wz), residuals)
        or None if the system is too ill-conditioned to trust.

        Translation and tilt are separated only by the quadratic radial term
        (1 + u'^2). When the tracked features sit in a small patch or along a line,
        the Vx and wy columns become near-collinear and lstsq still returns an
        arbitrary -- often enormous, or merely plausible but sign-flipped --
        translation with a tiny residual. RMSE cannot detect that, so conditioning
        has to be checked on every pass, including the post-residual refit: residual
        filtering can itself strip the peripheral tracks that were holding the system
        together."""
        f = self._f
        d = nxt - prev
        u = (prev[:, 0] - cx) / f
        v = (prev[:, 1] - cy) / f
        n = prev.shape[0]

        a = np.zeros((2 * n, 5), dtype=np.float64)
        b = np.zeros((2 * n,), dtype=np.float64)

        # flow_u = Vx + (u'v')wx - (1+u'^2)*f*wy + v'*f*wz
        # flow_v = Vy + (1+v'^2)*f*wx - (u'v')*f*wy - u'*f*wz
        # col 0: Vx, col 1: Vy, col 2: wx, col 3: wy, col 4: wz
        a[0::2, 0] = 1.0
        a[0::2, 2] = u * v * f
        a[0::2, 3] = -(1.0 + u ** 2) * f
        a[0::2, 4] = v * f
        b[0::2] = d[:, 0]

        a[1::2, 1] = 1.0
        a[1::2, 2] = (1.0 + v ** 2) * f
        a[1::2, 3] = -(u * v) * f
        a[1::2, 4] = -u * f
        b[1::2] = d[:, 1]

        if self._max_condition_number > 0.0:
            try:
                cond = float(np.linalg.cond(a))
            except np.linalg.LinAlgError:
                return None
            if not np.isfinite(cond) or cond > self._max_condition_number:
                return None

        try:
            sol, *_ = np.linalg.lstsq(a, b, rcond=None)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(sol)):
            return None

        vx, vy, wx, wy, wz = (float(s) for s in sol)
        pred_u = vx + (u * v * f) * wx - ((1.0 + u ** 2) * f) * wy + (v * f) * wz
        pred_v = vy + ((1.0 + v ** 2) * f) * wx - (u * v * f) * wy - (u * f) * wz
        residuals = np.sqrt((d[:, 0] - pred_u) ** 2 + (d[:, 1] - pred_v) ** 2)
        return (vx, vy, wx, wy, wz), residuals

    def _solve(
        self,
        prev: np.ndarray,
        nxt: np.ndarray,
        cx: float,
        cy: float,
        n_total: int,
    ) -> Optional[DerotationFit]:
        f = self._f
        if f <= 1e-6:
            return None

        first = self._fit_pass(prev, nxt, cx, cy)
        if first is None:
            return None
        params, residuals = first
        n = prev.shape[0]

        used = np.ones((n,), dtype=np.bool_)
        if self._residual_thresh_px > 0.0:
            used = residuals <= float(self._residual_thresh_px)
            n_used = int(np.count_nonzero(used))
            if n_used < self._min_points:
                # Too few points survived the residual gate to trust any fit. The
                # first-pass solution was computed over points we have just judged
                # inconsistent with the model, so returning it would hand the
                # controller an unvalidated velocity (and, with zero survivors, a
                # NaN RMSE that no downstream check would catch).
                return None
            if n_used < n:
                second = self._fit_pass(prev[used], nxt[used], cx, cy)
                if second is None:
                    # The surviving points alone are ill-conditioned.
                    return None
                params, r2 = second
                vx, vy, wx, wy, wz = params
                return DerotationFit(
                    vx_px=float(vx),
                    vy_px=float(vy),
                    omega_x_rad=float(wx),
                    omega_y_rad=float(wy),
                    omega_z_rad=float(wz),
                    rmse_px=float(np.sqrt(float(np.mean(r2 ** 2)))),
                    n_used=int(n_used),
                    inlier_ratio=float(n_used / max(1, n_total)),
                )

        n_used = int(np.count_nonzero(used))
        vx, vy, wx, wy, wz = params
        rmse = float(np.sqrt(float(np.mean(residuals[used] ** 2))))
        return DerotationFit(
            vx_px=float(vx),
            vy_px=float(vy),
            omega_x_rad=float(wx),
            omega_y_rad=float(wy),
            omega_z_rad=float(wz),
            rmse_px=float(rmse),
            n_used=int(n_used),
            inlier_ratio=float(n_used / max(1, n_total)),
        )


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
        focal_length_px: float = 0.0,
        principal_point_px: Optional[Tuple[float, float]] = None,
        derotation_residual_thresh_px: float = 3.0,
        derotation_min_points: int = 20,
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
        # focal_length_px is supplied in full-resolution pixels, but the derotation
        # solver receives point coordinates and cx/cy from the downscaled tracking
        # image. Normalized coordinates u' = (x - cx) / f are only consistent if f
        # lives in the same frame, so scale it by downscale. Getting this wrong
        # reparameterizes tilt into a large spurious translation *with zero residual*,
        # which means model_rmse_px cannot detect the error.
        self._focal_length_px = float(focal_length_px)
        self._focal_length_px_scaled = float(focal_length_px) * self._downscale
        # Optical centre in full-resolution pixels, scaled into the downscaled
        # tracking frame like f. Assuming the exact image midpoint makes pure
        # rotation look like translation: a yaw of omega_z leaks roughly
        # (cy_err * omega_z, -cx_err * omega_z) straight into the controller.
        # None falls back to the image midpoint.
        self._principal_point_px_scaled: Optional[Tuple[float, float]] = (
            None
            if principal_point_px is None
            else (float(principal_point_px[0]) * self._downscale, float(principal_point_px[1]) * self._downscale)
        )
        self._derot_model: Optional[DerotationMotionModel] = None
        if self._motion_model == "derotation":
            if self._focal_length_px_scaled <= 1e-6:
                # Previously this silently left _derot_model as None and ran the
                # affine-translation model instead, so a run configured for
                # derotation quietly used a different algorithm and reported
                # motion_model="affine_translation".
                raise ValueError(
                    "motion_model='derotation' requires a positive focal_length_px "
                    f"(got {focal_length_px!r}). Run the intrinsic calibration "
                    "(e88_autopilot.run_intrinsics_calibration) or pass --focal-length-px."
                )
            self._derot_model = DerotationMotionModel(
                focal_length_px=float(self._focal_length_px_scaled),
                residual_thresh_px=float(derotation_residual_thresh_px),
                min_points=int(derotation_min_points),
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
            # Phase correlation estimates a pure image shift and cannot separate tilt
            # from translation -- it reports rotational flow as ordinary, normal-quality
            # translation. That is precisely the artifact the derotation model exists to
            # remove, and inter-frame pitch or roll is a common reason the affine fit
            # trips this gate in the first place. So under derotation, fail closed here
            # rather than substituting an estimator that cannot answer the question.
            fallback = (
                None
                if self._motion_model == "derotation"
                else self._fallback_phase_correlation(prev_gray=self._prev_gray, gray=gray, dt=float(dt))
            )
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
        omega_x_rad = 0.0
        omega_y_rad = 0.0
        omega_z_rad = 0.0
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

        derotation_rejected = False
        if self._motion_model == "derotation" and self._derot_model is not None:
            derotation_rejected = n_tracked < 4
        if self._motion_model == "derotation" and self._derot_model is not None and n_tracked >= 4:
            if self._principal_point_px_scaled is None:
                img_cx = float(gray.shape[1]) * 0.5
                img_cy = float(gray.shape[0]) * 0.5
            else:
                img_cx, img_cy = self._principal_point_px_scaled
            derot_fit = self._derot_model.estimate(
                prev_xy=prev_good.reshape(-1, 2).astype(np.float32),
                next_xy=next_good.reshape(-1, 2).astype(np.float32),
                inliers=inliers,
                cx=img_cx,
                cy=img_cy,
            )
            # The max_step_px / min_inlier_ratio gates above were applied to the
            # affine translation, not to this fit. Derotation solves an
            # ill-conditioned system, so it can invent a huge translation on a frame
            # whose affine fit was excellent -- and would otherwise inherit that
            # affine fit's near-1.0 quality. Validate it on its own terms.
            if derot_fit is not None:
                derot_dx = float(derot_fit.vx_px)
                derot_dy = float(derot_fit.vy_px)
                if (
                    not np.isfinite(derot_dx)
                    or not np.isfinite(derot_dy)
                    or abs(derot_dx) > max_step_px
                    or abs(derot_dy) > max_step_px
                    or float(derot_fit.inlier_ratio) < float(self._min_inlier_ratio)
                ):
                    derot_fit = None

            if derot_fit is not None:
                dx_px = float(derot_fit.vx_px)
                dy_px = float(derot_fit.vy_px)
                # Report the support this fit actually had, not the affine fit's.
                inlier_ratio = float(derot_fit.inlier_ratio)
                omega_x_rad = float(derot_fit.omega_x_rad)
                omega_y_rad = float(derot_fit.omega_y_rad)
                omega_z_rad = float(derot_fit.omega_z_rad)
                omega_rad = float(omega_z_rad)
                omega_rad_s = float(omega_rad / float(dt))
                model_rmse_px = float(derot_fit.rmse_px) * scale
                motion_model = str("derotation")
            else:
                # Fail closed. dx_px/dy_px still hold the affine translation, which
                # for a pure camera tilt is exactly the spurious translation
                # derotation exists to remove -- passing it through with a healthy
                # quality score would defeat the point of selecting this model.
                derotation_rejected = True

        if derotation_rejected:
            motion_model = str("derotation_rejected")

        dx_px *= scale
        dy_px *= scale
        prev_xy = prev_good.reshape(-1, 2).astype(np.float32) * scale
        next_xy = next_good.reshape(-1, 2).astype(np.float32) * scale
        self._last_tracks = FlowTracks(prev_xy=prev_xy, next_xy=next_xy, inliers=inliers)

        vx = dx_px / dt
        vy = dy_px / dt

        tracked_ratio = (n_tracked / max(1, n_features))
        quality = float(max(0.0, min(1.0, tracked_ratio * inlier_ratio)))
        if derotation_rejected:
            # Zero quality gates the sample out of the controller (which returns None
            # below min_quality) while still emitting it for diagnostics.
            quality = 0.0

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
            omega_x_rad=float(omega_x_rad),
            omega_y_rad=float(omega_y_rad),
            omega_z_rad=float(omega_z_rad),
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
