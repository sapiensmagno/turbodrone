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


if __name__ == "__main__":
    unittest.main()
