import sys
import unittest
from pathlib import Path

import cv2
import numpy as np
from unittest import mock


_EXPERIMENTAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXPERIMENTAL_ROOT))

from e88_autopilot.optical_flow import LucasKanadeDriftEstimator
from e88_autopilot.optical_flow import TranslationRotationMotionModel
from e88_autopilot.optical_flow import DerotationMotionModel


class TestLucasKanadeDriftEstimator(unittest.TestCase):
    def test_detects_known_translation(self):
        h, w = 240, 320
        base = np.zeros((h, w, 3), dtype=np.uint8)

        rng = np.random.default_rng(0)
        for _ in range(200):
            x = int(rng.integers(10, w - 10))
            y = int(rng.integers(10, h - 10))
            cv2.circle(base, (x, y), 2, (255, 255, 255), -1)

        dx, dy = 6.0, -4.0
        m = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        shifted = cv2.warpAffine(base, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        est = LucasKanadeDriftEstimator(downscale=1.0, reinit_every_n_frames=0, min_tracked_features=10)

        self.assertIsNone(est.update(base, timestamp=0.0))
        out = est.update(shifted, timestamp=0.1)
        self.assertIsNotNone(out)
        assert out is not None

        self.assertAlmostEqual(out.dx_px, dx, delta=2.0)
        self.assertAlmostEqual(out.dy_px, dy, delta=2.0)
        self.assertGreater(out.n_tracked, 10)

    def test_downscale_rescales_output_to_full_resolution(self):
        h, w = 240, 320
        base = np.zeros((h, w, 3), dtype=np.uint8)

        rng = np.random.default_rng(0)
        for _ in range(200):
            x = int(rng.integers(10, w - 10))
            y = int(rng.integers(10, h - 10))
            cv2.circle(base, (x, y), 2, (255, 255, 255), -1)

        dx, dy = 10.0, -6.0
        m = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        shifted = cv2.warpAffine(base, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        est = LucasKanadeDriftEstimator(downscale=0.5, reinit_every_n_frames=0, min_tracked_features=10)

        self.assertIsNone(est.update(base, timestamp=0.0))
        out = est.update(shifted, timestamp=0.1)
        self.assertIsNotNone(out)
        assert out is not None

        # dx/dy should be reported in full-resolution pixels even when tracking is downscaled.
        self.assertAlmostEqual(out.dx_px, dx, delta=3.0)
        self.assertAlmostEqual(out.dy_px, dy, delta=3.0)

    def test_phase_correlation_fallback_path(self):
        # Force LK to fail and ensure phase correlation provides the estimate.
        h, w = 120, 160
        base = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.circle(base, (50, 50), 5, (255, 255, 255), -1)

        shifted = base.copy()

        est = LucasKanadeDriftEstimator(downscale=1.0, reinit_every_n_frames=0, min_tracked_features=10)

        self.assertIsNone(est.update(base, timestamp=0.0))

        with mock.patch("cv2.calcOpticalFlowPyrLK", return_value=(None, None, None)):
            with mock.patch("cv2.phaseCorrelate", return_value=((3.0, -2.0), 0.9)):
                out = est.update(shifted, timestamp=0.1)

        self.assertIsNotNone(out)
        assert out is not None
        self.assertAlmostEqual(out.dx_px, 3.0, delta=1e-6)
        self.assertAlmostEqual(out.dy_px, -2.0, delta=1e-6)
        self.assertGreater(out.quality, 0.0)

    def test_translation_rotation_model_removes_center_rotation_translation(self):
        h, w = 240, 320
        frame0 = np.zeros((h, w, 3), dtype=np.uint8)
        frame1 = frame0.copy()

        rng = np.random.default_rng(123)
        n = 120
        prev_xy = rng.uniform([0.0, 0.0], [float(w), float(h)], size=(n, 2)).astype(np.float32)
        prev_pts = prev_xy.reshape(-1, 1, 2).astype(np.float32)

        cx = float(np.median(prev_xy[:, 0]))
        cy = float(np.median(prev_xy[:, 1]))
        theta = 0.05
        c = float(np.cos(theta))
        s = float(np.sin(theta))
        x = prev_xy[:, 0] - cx
        y = prev_xy[:, 1] - cy
        next_xy = np.stack([cx + c * x - s * y, cy + s * x + c * y], axis=1).astype(np.float32)
        next_pts = next_xy.reshape(-1, 1, 2).astype(np.float32)

        tx = cx * (1.0 - c) + s * cy
        ty = cy * (1.0 - c) - s * cx
        m = np.array([[c, -s, tx], [s, c, ty]], dtype=np.float32)

        status = np.ones((n, 1), dtype=np.uint8)
        inliers = np.ones((n, 1), dtype=np.uint8)

        def _fake_good_features_to_track(_gray, mask=None, **kwargs):
            return prev_pts.copy()

        def _fake_lk(_prev_gray, _gray, _prev_pts, _next, **kwargs):
            return next_pts.copy(), status.copy(), None

        def _fake_affine(_prev, _next, method=None, ransacReprojThreshold=None):
            return m.copy(), inliers.copy()

        with mock.patch("cv2.goodFeaturesToTrack", side_effect=_fake_good_features_to_track):
            with mock.patch("cv2.calcOpticalFlowPyrLK", side_effect=_fake_lk):
                with mock.patch("cv2.estimateAffinePartial2D", side_effect=_fake_affine):
                    est_aff = LucasKanadeDriftEstimator(
                        downscale=1.0,
                        reinit_every_n_frames=0,
                        min_tracked_features=10,
                        motion_model="affine_translation",
                    )
                    self.assertIsNone(est_aff.update(frame0, timestamp=0.0))
                    out_aff = est_aff.update(frame1, timestamp=0.1)
                    self.assertIsNotNone(out_aff)
                    assert out_aff is not None
                    self.assertAlmostEqual(out_aff.dx_px, float(tx), delta=1e-6)
                    self.assertAlmostEqual(out_aff.dy_px, float(ty), delta=1e-6)

        with mock.patch("cv2.goodFeaturesToTrack", side_effect=_fake_good_features_to_track):
            with mock.patch("cv2.calcOpticalFlowPyrLK", side_effect=_fake_lk):
                with mock.patch("cv2.estimateAffinePartial2D", side_effect=_fake_affine):
                    est_tr = LucasKanadeDriftEstimator(
                        downscale=1.0,
                        reinit_every_n_frames=0,
                        min_tracked_features=10,
                        motion_model="translation_rotation",
                        tr_residual_thresh_px=0.5,
                        tr_min_points=50,
                    )
                    self.assertIsNone(est_tr.update(frame0, timestamp=0.0))
                    out_tr = est_tr.update(frame1, timestamp=0.1)
                    self.assertIsNotNone(out_tr)
                    assert out_tr is not None

                    self.assertEqual(out_tr.motion_model, "translation_rotation")
                    self.assertAlmostEqual(out_tr.dx_px, 0.0, delta=0.25)
                    self.assertAlmostEqual(out_tr.dy_px, 0.0, delta=0.25)
                    self.assertAlmostEqual(out_tr.raw_dx_px, float(tx), delta=1e-6)
                    self.assertAlmostEqual(out_tr.raw_dy_px, float(ty), delta=1e-6)
                    self.assertAlmostEqual(out_tr.omega_rad, theta, delta=0.01)


class TestTranslationRotationMotionModel(unittest.TestCase):
    def test_estimates_pure_rotation_with_near_zero_translation(self):
        rng = np.random.default_rng(0)
        n = 200
        prev = rng.uniform([0.0, 0.0], [320.0, 240.0], size=(n, 2)).astype(np.float32)

        cx = float(np.median(prev[:, 0]))
        cy = float(np.median(prev[:, 1]))
        theta = 0.04
        c = float(np.cos(theta))
        s = float(np.sin(theta))
        x = prev[:, 0] - cx
        y = prev[:, 1] - cy
        next_xy = np.stack([cx + c * x - s * y, cy + s * x + c * y], axis=1).astype(np.float32)

        model = TranslationRotationMotionModel(residual_thresh_px=0.5, min_points=50)
        fit = model.estimate(prev_xy=prev, next_xy=next_xy, inliers=None)
        self.assertIsNotNone(fit)
        assert fit is not None

        self.assertAlmostEqual(fit.tx_px, 0.0, delta=0.25)
        self.assertAlmostEqual(fit.ty_px, 0.0, delta=0.25)
        self.assertAlmostEqual(fit.omega_rad, theta, delta=0.01)
        self.assertLess(fit.rmse_px, 0.3)

    def test_estimates_translation_and_rotation(self):
        rng = np.random.default_rng(1)
        n = 250
        prev = rng.uniform([0.0, 0.0], [320.0, 240.0], size=(n, 2)).astype(np.float32)

        cx = float(np.median(prev[:, 0]))
        cy = float(np.median(prev[:, 1]))
        theta = -0.03
        dx = 4.0
        dy = -3.0
        c = float(np.cos(theta))
        s = float(np.sin(theta))
        x = prev[:, 0] - cx
        y = prev[:, 1] - cy
        next_xy = np.stack([cx + c * x - s * y + dx, cy + s * x + c * y + dy], axis=1).astype(np.float32)

        model = TranslationRotationMotionModel(residual_thresh_px=0.75, min_points=80)
        fit = model.estimate(prev_xy=prev, next_xy=next_xy, inliers=None)
        self.assertIsNotNone(fit)
        assert fit is not None

        self.assertAlmostEqual(fit.tx_px, dx, delta=0.25)
        self.assertAlmostEqual(fit.ty_px, dy, delta=0.25)
        self.assertAlmostEqual(fit.omega_rad, theta, delta=0.01)

    def test_inlier_mask_ignores_outliers(self):
        rng = np.random.default_rng(2)
        n_in = 180
        n_out = 50
        prev_in = rng.uniform([0.0, 0.0], [320.0, 240.0], size=(n_in, 2)).astype(np.float32)

        cx = float(np.median(prev_in[:, 0]))
        cy = float(np.median(prev_in[:, 1]))
        theta = 0.05
        dx = -2.5
        dy = 1.0
        c = float(np.cos(theta))
        s = float(np.sin(theta))
        x = prev_in[:, 0] - cx
        y = prev_in[:, 1] - cy
        next_in = np.stack([cx + c * x - s * y + dx, cy + s * x + c * y + dy], axis=1).astype(np.float32)

        prev_out = rng.uniform([0.0, 0.0], [320.0, 240.0], size=(n_out, 2)).astype(np.float32)
        next_out = rng.uniform([0.0, 0.0], [320.0, 240.0], size=(n_out, 2)).astype(np.float32)

        prev = np.concatenate([prev_in, prev_out], axis=0)
        nxt = np.concatenate([next_in, next_out], axis=0)
        inliers = np.zeros((n_in + n_out,), dtype=np.bool_)
        inliers[:n_in] = True

        model = TranslationRotationMotionModel(residual_thresh_px=0.75, min_points=80)
        fit = model.estimate(prev_xy=prev, next_xy=nxt, inliers=inliers)
        self.assertIsNotNone(fit)
        assert fit is not None

        self.assertAlmostEqual(fit.tx_px, dx, delta=0.3)
        self.assertAlmostEqual(fit.ty_px, dy, delta=0.3)
        self.assertAlmostEqual(fit.omega_rad, theta, delta=0.015)


class TestDerotationMotionModel(unittest.TestCase):
    """Prove that DerotationMotionModel correctly separates camera
    rotation (tilt) from translational velocity using the pinhole
    camera optical flow equations."""

    def _generate_flow(
        self,
        *,
        n: int = 200,
        w: float = 160.0,
        h: float = 120.0,
        f: float = 56.0,
        vx: float = 0.0,
        vy: float = 0.0,
        wx: float = 0.0,
        wy: float = 0.0,
        wz: float = 0.0,
        seed: int = 42,
        noise_std: float = 0.0,
    ):
        rng = np.random.default_rng(seed)
        cx, cy = w * 0.5, h * 0.5
        prev = rng.uniform([10.0, 10.0], [w - 10.0, h - 10.0], size=(n, 2)).astype(np.float64)

        u_prime = (prev[:, 0] - cx) / f
        v_prime = (prev[:, 1] - cy) / f

        dx = vx + (u_prime * v_prime * f) * wx - ((1.0 + u_prime ** 2) * f) * wy + (v_prime * f) * wz
        dy = vy + ((1.0 + v_prime ** 2) * f) * wx - (u_prime * v_prime * f) * wy - (u_prime * f) * wz

        if noise_std > 0.0:
            dx += rng.normal(0.0, noise_std, size=n)
            dy += rng.normal(0.0, noise_std, size=n)

        nxt = prev + np.stack([dx, dy], axis=1)
        return prev.astype(np.float32), nxt.astype(np.float32), cx, cy

    def test_pure_pitch_gives_zero_translation(self):
        """Pure pitch rotation (wx) should yield Vx~0, Vy~0."""
        f = 56.0
        wx = 0.02  # ~1.15 deg pitch
        prev, nxt, cx, cy = self._generate_flow(f=f, wx=wx)

        model = DerotationMotionModel(focal_length_px=f, residual_thresh_px=3.0, min_points=20)
        fit = model.estimate(prev_xy=prev, next_xy=nxt, inliers=None, cx=cx, cy=cy)
        self.assertIsNotNone(fit)
        assert fit is not None

        self.assertAlmostEqual(fit.vx_px, 0.0, delta=0.1)
        self.assertAlmostEqual(fit.vy_px, 0.0, delta=0.1)
        self.assertAlmostEqual(fit.omega_x_rad, wx, delta=0.001)
        self.assertLess(fit.rmse_px, 0.1)

    def test_pure_roll_gives_zero_translation(self):
        """Pure roll rotation (wy) should yield Vx~0, Vy~0."""
        f = 56.0
        wy = -0.015  # ~0.86 deg roll
        prev, nxt, cx, cy = self._generate_flow(f=f, wy=wy)

        model = DerotationMotionModel(focal_length_px=f, residual_thresh_px=3.0, min_points=20)
        fit = model.estimate(prev_xy=prev, next_xy=nxt, inliers=None, cx=cx, cy=cy)
        self.assertIsNotNone(fit)
        assert fit is not None

        self.assertAlmostEqual(fit.vx_px, 0.0, delta=0.1)
        self.assertAlmostEqual(fit.vy_px, 0.0, delta=0.1)
        self.assertAlmostEqual(fit.omega_y_rad, wy, delta=0.001)
        self.assertLess(fit.rmse_px, 0.1)

    def test_pure_yaw_gives_zero_translation(self):
        """Pure yaw rotation (wz) should yield Vx~0, Vy~0."""
        f = 56.0
        wz = 0.03
        prev, nxt, cx, cy = self._generate_flow(f=f, wz=wz)

        model = DerotationMotionModel(focal_length_px=f, residual_thresh_px=3.0, min_points=20)
        fit = model.estimate(prev_xy=prev, next_xy=nxt, inliers=None, cx=cx, cy=cy)
        self.assertIsNotNone(fit)
        assert fit is not None

        self.assertAlmostEqual(fit.vx_px, 0.0, delta=0.1)
        self.assertAlmostEqual(fit.vy_px, 0.0, delta=0.1)
        self.assertAlmostEqual(fit.omega_z_rad, wz, delta=0.001)

    def test_pure_translation_gives_zero_rotation(self):
        """Pure translation should yield wx~0, wy~0, wz~0."""
        f = 56.0
        vx, vy = 4.0, -3.0
        prev, nxt, cx, cy = self._generate_flow(f=f, vx=vx, vy=vy)

        model = DerotationMotionModel(focal_length_px=f, residual_thresh_px=3.0, min_points=20)
        fit = model.estimate(prev_xy=prev, next_xy=nxt, inliers=None, cx=cx, cy=cy)
        self.assertIsNotNone(fit)
        assert fit is not None

        self.assertAlmostEqual(fit.vx_px, vx, delta=0.1)
        self.assertAlmostEqual(fit.vy_px, vy, delta=0.1)
        self.assertAlmostEqual(fit.omega_x_rad, 0.0, delta=0.001)
        self.assertAlmostEqual(fit.omega_y_rad, 0.0, delta=0.001)
        self.assertAlmostEqual(fit.omega_z_rad, 0.0, delta=0.001)

    def test_mixed_translation_and_tilt(self):
        """Mixed translation + pitch/roll: model correctly recovers both."""
        f = 56.0
        vx, vy = 3.0, -2.0
        wx, wy = 0.01, -0.008
        prev, nxt, cx, cy = self._generate_flow(f=f, vx=vx, vy=vy, wx=wx, wy=wy)

        model = DerotationMotionModel(focal_length_px=f, residual_thresh_px=3.0, min_points=20)
        fit = model.estimate(prev_xy=prev, next_xy=nxt, inliers=None, cx=cx, cy=cy)
        self.assertIsNotNone(fit)
        assert fit is not None

        self.assertAlmostEqual(fit.vx_px, vx, delta=0.15)
        self.assertAlmostEqual(fit.vy_px, vy, delta=0.15)
        self.assertAlmostEqual(fit.omega_x_rad, wx, delta=0.002)
        self.assertAlmostEqual(fit.omega_y_rad, wy, delta=0.002)

    def test_comparison_with_translation_rotation_model(self):
        """Show that TranslationRotationMotionModel gives WRONG translation
        for pure pitch, while DerotationMotionModel gives correct ~zero."""
        f = 56.0
        wx = 0.02  # pure pitch, no translation
        prev, nxt, cx, cy = self._generate_flow(f=f, wx=wx, n=250)

        # TranslationRotationMotionModel: should give WRONG tx/ty
        tr_model = TranslationRotationMotionModel(residual_thresh_px=3.0, min_points=20)
        tr_fit = tr_model.estimate(prev_xy=prev, next_xy=nxt, inliers=None)
        self.assertIsNotNone(tr_fit)
        assert tr_fit is not None
        # The old model will attribute some of the rotation to translation
        tr_translation_magnitude = abs(tr_fit.tx_px) + abs(tr_fit.ty_px)

        # DerotationMotionModel: should give correct ~zero translation
        derot_model = DerotationMotionModel(focal_length_px=f, residual_thresh_px=3.0, min_points=20)
        derot_fit = derot_model.estimate(prev_xy=prev, next_xy=nxt, inliers=None, cx=cx, cy=cy)
        self.assertIsNotNone(derot_fit)
        assert derot_fit is not None
        derot_translation_magnitude = abs(derot_fit.vx_px) + abs(derot_fit.vy_px)

        # Derotation should be much closer to zero than the old model
        self.assertLess(derot_translation_magnitude, 0.2)
        self.assertGreater(tr_translation_magnitude, derot_translation_magnitude)

    def test_robust_to_noise(self):
        """With moderate tracking noise, still separates tilt from translation."""
        f = 56.0
        vx, vy = 2.0, -1.5
        wx, wy = 0.012, -0.01
        prev, nxt, cx, cy = self._generate_flow(
            f=f, vx=vx, vy=vy, wx=wx, wy=wy, noise_std=0.5, n=200
        )

        model = DerotationMotionModel(focal_length_px=f, residual_thresh_px=3.0, min_points=20)
        fit = model.estimate(prev_xy=prev, next_xy=nxt, inliers=None, cx=cx, cy=cy)
        self.assertIsNotNone(fit)
        assert fit is not None

        self.assertAlmostEqual(fit.vx_px, vx, delta=0.5)
        self.assertAlmostEqual(fit.vy_px, vy, delta=0.5)
        self.assertAlmostEqual(fit.omega_x_rad, wx, delta=0.005)
        self.assertAlmostEqual(fit.omega_y_rad, wy, delta=0.005)

    def test_tolerates_focal_length_error(self):
        """Even with 30% error in focal length, still separates tilt."""
        f_true = 56.0
        f_wrong = 56.0 * 1.3  # 30% overestimate
        wx = 0.02  # pure pitch
        prev, nxt, cx, cy = self._generate_flow(f=f_true, wx=wx)

        model = DerotationMotionModel(focal_length_px=f_wrong, residual_thresh_px=3.0, min_points=20)
        fit = model.estimate(prev_xy=prev, next_xy=nxt, inliers=None, cx=cx, cy=cy)
        self.assertIsNotNone(fit)
        assert fit is not None

        # Translation should still be much closer to zero than the total flow magnitude
        self.assertLess(abs(fit.vx_px), 1.0)
        self.assertLess(abs(fit.vy_px), 1.0)


class TestDerotationFocalLengthDownscale(unittest.TestCase):
    """The derotation solver receives point coordinates and cx/cy from the
    downscaled tracking image, but focal_length_px is supplied in
    full-resolution pixels. f must be converted to the downscaled frame or the
    normalized coordinates u' = (x - cx) / f are inconsistent.

    These tests are about the *wiring* between LucasKanadeDriftEstimator and
    DerotationMotionModel, not about the solver algebra (covered above)."""

    @staticmethod
    def _tilt_tracks(*, f: float, w: float, h: float, wy: float, n: int = 200, seed: int = 7):
        """Synthesize a pure-tilt flow field in full-resolution pixels."""
        rng = np.random.default_rng(seed)
        cx, cy = w * 0.5, h * 0.5
        prev = rng.uniform([10.0, 10.0], [w - 10.0, h - 10.0], size=(n, 2))
        u_prime = (prev[:, 0] - cx) / f
        v_prime = (prev[:, 1] - cy) / f
        dx = -((1.0 + u_prime ** 2) * f) * wy
        dy = -(u_prime * v_prime * f) * wy
        nxt = prev + np.stack([dx, dy], axis=1)
        return prev.astype(np.float32), nxt.astype(np.float32)

    def test_estimator_scales_focal_length_by_downscale(self):
        est = LucasKanadeDriftEstimator(motion_model="derotation", focal_length_px=366.0, downscale=0.5)
        self.assertIsNotNone(est._derot_model)
        assert est._derot_model is not None
        self.assertAlmostEqual(est._derot_model.focal_length_px, 183.0, places=6)

        est_full = LucasKanadeDriftEstimator(motion_model="derotation", focal_length_px=366.0, downscale=1.0)
        assert est_full._derot_model is not None
        self.assertAlmostEqual(est_full._derot_model.focal_length_px, 366.0, places=6)

    def test_failed_derotation_fit_is_gated_out_not_silently_affine(self):
        """When the derotation fit is rejected, dx/dy still hold the affine
        translation -- which for pure camera tilt is exactly the spurious translation
        derotation exists to remove. It must not reach the controller with a healthy
        quality score."""
        h, w = 200, 260
        rng = np.random.default_rng(4)
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Texture confined to a small central patch: enough for LK to track, but a
        # near-singular derotation system.
        for _ in range(400):
            x = int(rng.integers(w // 2 - 22, w // 2 + 22))
            y = int(rng.integers(h // 2 - 22, h // 2 + 22))
            frame[y, x] = 255

        est = LucasKanadeDriftEstimator(
            motion_model="derotation",
            focal_length_px=200.0,
            downscale=1.0,
            reinit_every_n_frames=0,
            min_tracked_features=5,
        )
        shifted = np.roll(frame, 2, axis=1)
        self.assertIsNone(est.update(frame, timestamp=0.0))
        out = est.update(shifted, timestamp=0.05)

        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(out.motion_model, "derotation_rejected")
        self.assertEqual(out.quality, 0.0)

    def test_derotation_never_falls_back_to_phase_correlation(self):
        """Phase correlation estimates a pure image shift and reports rotational flow
        as ordinary, normal-quality translation -- exactly the artifact derotation
        exists to remove. Inter-frame tilt is also a common reason the affine fit
        trips the gate that triggers the fallback, so the two must not be paired."""
        h, w = 200, 260
        rng = np.random.default_rng(8)
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        for _ in range(500):
            frame[int(rng.integers(3, h - 3)), int(rng.integers(3, w - 3))] = 255

        # A translation far beyond max_translation_frac forces the fallback branch.
        shifted = np.roll(frame, int(0.4 * min(h, w)), axis=1)

        with mock.patch.object(
            LucasKanadeDriftEstimator, "_fallback_phase_correlation", return_value=(999.0, 0.0, 0.9)
        ) as patched:
            est = LucasKanadeDriftEstimator(
                motion_model="derotation",
                focal_length_px=200.0,
                downscale=1.0,
                reinit_every_n_frames=0,
                min_tracked_features=5,
            )
            self.assertIsNone(est.update(frame, timestamp=0.0))
            out = est.update(shifted, timestamp=0.05)
            patched.assert_not_called()

        if out is not None:
            self.assertNotEqual(out.motion_model, "phase_corr")

        # The same setup under affine_translation *does* use the fallback, confirming
        # the test actually reaches that branch.
        with mock.patch.object(
            LucasKanadeDriftEstimator, "_fallback_phase_correlation", return_value=(999.0, 0.0, 0.9)
        ) as patched:
            est2 = LucasKanadeDriftEstimator(
                motion_model="affine_translation", downscale=1.0, reinit_every_n_frames=0, min_tracked_features=5
            )
            self.assertIsNone(est2.update(frame, timestamp=0.0))
            est2.update(shifted, timestamp=0.05)
            patched.assert_called()

    def test_derotation_without_focal_length_raises_instead_of_downgrading(self):
        """Previously this silently left the derotation model unbuilt and ran the
        affine-translation model, so a run configured for derotation quietly used a
        different algorithm and reported motion_model='affine_translation'."""
        with self.assertRaises(ValueError):
            LucasKanadeDriftEstimator(motion_model="derotation", focal_length_px=0.0)
        # Other models are unaffected.
        LucasKanadeDriftEstimator(motion_model="translation_rotation", focal_length_px=0.0)

    def test_principal_point_is_scaled_by_downscale(self):
        """Assuming the image midpoint makes pure yaw look like translation, leaking
        roughly (cy_err * wz, -cx_err * wz) into the controller."""
        est = LucasKanadeDriftEstimator(
            motion_model="derotation", focal_length_px=366.0, principal_point_px=(330.0, 250.0), downscale=0.5
        )
        self.assertEqual(est._principal_point_px_scaled, (165.0, 125.0))

        default = LucasKanadeDriftEstimator(motion_model="derotation", focal_length_px=366.0, downscale=0.5)
        self.assertIsNone(default._principal_point_px_scaled)

    def test_downscaled_solve_matches_full_resolution_solve(self):
        """Same physical tilt, tracked at full and half resolution, must give
        the same omega and the same translation once rescaled."""
        f_full, w, h, wy = 366.0, 640.0, 480.0, -0.01
        prev, nxt = self._tilt_tracks(f=f_full, w=w, h=h, wy=wy)

        full = DerotationMotionModel(focal_length_px=f_full, min_points=20)
        fit_full = full.estimate(prev_xy=prev, next_xy=nxt, inliers=None, cx=w * 0.5, cy=h * 0.5)

        s = 0.5
        half = DerotationMotionModel(focal_length_px=f_full * s, min_points=20)
        fit_half = half.estimate(
            prev_xy=(prev * s).astype(np.float32),
            next_xy=(nxt * s).astype(np.float32),
            inliers=None,
            cx=w * 0.5 * s,
            cy=h * 0.5 * s,
        )

        assert fit_full is not None and fit_half is not None
        # Rotation is scale-free; translation is in downscaled px and rescales by 1/s.
        self.assertAlmostEqual(fit_half.omega_y_rad, fit_full.omega_y_rad, delta=1e-6)
        self.assertAlmostEqual(fit_half.vx_px / s, fit_full.vx_px, delta=0.05)
        self.assertAlmostEqual(fit_half.vy_px / s, fit_full.vy_px, delta=0.05)
        self.assertAlmostEqual(fit_half.vx_px / s, 0.0, delta=0.05)

    def test_unscaled_focal_length_fabricates_translation_with_zero_residual(self):
        """Regression guard documenting why the bug was invisible: feeding a 2x
        too large f fits pure tilt *exactly* while inventing a large
        translation, so model_rmse_px cannot detect the error."""
        f_true, w, h, wy = 183.0, 320.0, 240.0, -0.01
        prev, nxt = self._tilt_tracks(f=f_true, w=w, h=h, wy=wy)

        # Conditioning gate disabled: this test probes the raw algebra, and a 2x-wrong
        # f is itself badly conditioned (~7700) so the gate would reject it first.
        wrong = DerotationMotionModel(focal_length_px=f_true * 2.0, min_points=20, max_condition_number=0.0)
        fit = wrong.estimate(prev_xy=prev, next_xy=nxt, inliers=None, cx=w * 0.5, cy=h * 0.5)
        assert fit is not None

        # Residual stays tiny -- RMSE is blind to this failure.
        self.assertLess(fit.rmse_px, 0.05)
        # ...while a pure-tilt scene is reported as a large translation.
        self.assertGreater(abs(fit.vx_px), 3.0)

        # The correctly-scaled model reports no translation on the same data.
        right = DerotationMotionModel(focal_length_px=f_true, min_points=20)
        fit_ok = right.estimate(prev_xy=prev, next_xy=nxt, inliers=None, cx=w * 0.5, cy=h * 0.5)
        assert fit_ok is not None
        self.assertAlmostEqual(fit_ok.vx_px, 0.0, delta=0.05)


class TestDerotationGuards(unittest.TestCase):
    """Guards on a solver whose failure mode is invisible to its own residual."""

    @staticmethod
    def _tracks(*, f, w, h, n=200, seed=11, frac=1.0, vx=0.0):
        rng = np.random.default_rng(seed)
        cx, cy = w * 0.5, h * 0.5
        lo = np.array([cx - frac * w / 2 + 5, cy - frac * h / 2 + 5])
        hi = np.array([cx + frac * w / 2 - 5, cy + frac * h / 2 - 5])
        prev = rng.uniform(lo, hi, size=(n, 2))
        nxt = prev + np.array([vx, 0.0])
        return prev.astype(np.float32), nxt.astype(np.float32)

    def test_rejects_features_concentrated_near_the_optical_axis(self):
        """Translation and tilt are separated only by the (1+u'^2) radial term, so
        centrally-clustered features make the system near-singular. lstsq still
        returns an answer with a tiny residual, so RMSE cannot catch it."""
        f, w, h = 183.0, 320.0, 240.0
        model = DerotationMotionModel(focal_length_px=f, min_points=20)

        prev, nxt = self._tracks(f=f, w=w, h=h, frac=1.0, vx=1.0)
        self.assertIsNotNone(model.estimate(prev_xy=prev, next_xy=nxt, inliers=None, cx=w / 2, cy=h / 2))

        prev, nxt = self._tracks(f=f, w=w, h=h, frac=0.2, vx=1.0)
        self.assertIsNone(model.estimate(prev_xy=prev, next_xy=nxt, inliers=None, cx=w / 2, cy=h / 2))

    def test_conditioning_gate_can_be_disabled(self):
        f, w, h = 183.0, 320.0, 240.0
        prev, nxt = self._tracks(f=f, w=w, h=h, frac=0.2, vx=1.0)
        model = DerotationMotionModel(focal_length_px=f, min_points=20, max_condition_number=0.0)
        self.assertIsNotNone(model.estimate(prev_xy=prev, next_xy=nxt, inliers=None, cx=w / 2, cy=h / 2))

    def test_rechecks_conditioning_after_the_residual_refit(self):
        """Residual filtering can strip exactly the peripheral tracks that were
        holding the system together, so the refit must be conditioning-checked too --
        a near-singular refit yields a low-RMSE, plausible-magnitude, possibly
        sign-flipped translation that the outer gates would happily accept."""
        f, w, h = 183.0, 320.0, 240.0
        cx, cy = w / 2, h / 2
        rng = np.random.default_rng(19)

        # A well-spread core that alone is ill-conditioned (tight central cluster),
        # plus peripheral points that will be rejected by the residual gate.
        centre = rng.uniform([cx - 20, cy - 20], [cx + 20, cy + 20], size=(40, 2))
        edge = np.array(
            [[15.0, 15.0], [w - 15, 15.0], [w - 15, h - 15], [15.0, h - 15]] * 10, dtype=np.float64
        )
        prev = np.vstack([centre, edge])
        nxt = prev + np.array([1.0, 0.0])
        # Make the peripheral points inconsistent so the residual gate drops them.
        nxt[len(centre) :] += rng.normal(0.0, 40.0, size=(len(edge), 2))

        model = DerotationMotionModel(
            focal_length_px=f, residual_thresh_px=1.0, min_points=20, max_condition_number=5000.0
        )
        fit = model.estimate(
            prev_xy=prev.astype(np.float32), next_xy=nxt.astype(np.float32), inliers=None, cx=cx, cy=cy
        )
        self.assertIsNone(fit)

    def test_returns_none_when_too_few_points_survive_the_residual_gate(self):
        """The first-pass solution was fitted over points since judged inconsistent
        with the model; returning it would hand the controller an unvalidated
        velocity (and a NaN RMSE when nothing survives)."""
        f, w, h = 183.0, 320.0, 240.0
        rng = np.random.default_rng(5)
        cx, cy = w / 2, h / 2
        prev = rng.uniform([10, 10], [w - 10, h - 10], size=(60, 2))
        # Displacements that no rigid-motion model can explain.
        nxt = prev + rng.normal(0.0, 60.0, size=(60, 2))

        model = DerotationMotionModel(focal_length_px=f, residual_thresh_px=0.05, min_points=20)
        fit = model.estimate(
            prev_xy=prev.astype(np.float32), next_xy=nxt.astype(np.float32), inliers=None, cx=cx, cy=cy
        )
        self.assertIsNone(fit)


if __name__ == "__main__":
    unittest.main()
