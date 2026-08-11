import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


_EXPERIMENTAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXPERIMENTAL_ROOT))

from e88_autopilot.reference_detection import ReferenceDetector


class TestReferenceDetector(unittest.TestCase):
    def test_orb_homography_detects_reference_and_estimates_size(self) -> None:
        rng = np.random.default_rng(0)

        ref_h, ref_w = 220, 240
        ref = np.zeros((ref_h, ref_w, 3), dtype=np.uint8)
        for _ in range(400):
            x = int(rng.integers(5, ref_w - 5))
            y = int(rng.integers(5, ref_h - 5))
            cv2.circle(ref, (x, y), 2, (255, 255, 255), -1)

        frame_h, frame_w = 360, 420
        frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

        src = np.float32([[0, 0], [ref_w - 1, 0], [ref_w - 1, ref_h - 1], [0, ref_h - 1]])
        dst = np.float32(
            [
                [90, 70],
                [300, 55],
                [330, 250],
                [110, 280],
            ]
        )

        h = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(ref, h, (frame_w, frame_h), flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(
            np.ones((ref_h, ref_w), dtype=np.uint8) * 255,
            h,
            (frame_w, frame_h),
            flags=cv2.INTER_NEAREST,
        )

        frame[mask > 0] = warped[mask > 0]

        det = ReferenceDetector(reference_bgr=ref, use_markers=False, min_matches=20, min_inliers=15, min_inlier_ratio=0.3)
        out = det.detect(frame)

        self.assertTrue(out.detected)
        self.assertIsNotNone(out.quad_xy)
        self.assertGreater(out.n_inliers, 0)
        self.assertGreater(out.inlier_ratio, 0.0)

        q = np.asarray(out.quad_xy, dtype=np.float32).reshape(4, 2)
        w1 = float(np.linalg.norm(q[1] - q[0]))
        w2 = float(np.linalg.norm(q[2] - q[3]))
        h1 = float(np.linalg.norm(q[3] - q[0]))
        h2 = float(np.linalg.norm(q[2] - q[1]))
        expected = 0.5 * (0.5 * (w1 + w2) + 0.5 * (h1 + h2))

        self.assertAlmostEqual(out.ref_size_px, expected, delta=25.0)

    def test_orb_measures_pad_rect_not_whole_reference_image(self) -> None:
        """The reference image is stored uncropped so ORB has surrounding texture to
        match, but pad_width_m describes only the pad. When a pad rectangle is
        supplied the projected quad must track the pad, not the image border --
        otherwise ref_size_px measures the wrong object and m_per_px (and the focal
        length derived from it) are scaled by the photo/pad extent ratio."""
        rng = np.random.default_rng(3)

        ref_h, ref_w = 240, 240
        ref = np.zeros((ref_h, ref_w, 3), dtype=np.uint8)
        for _ in range(600):
            x = int(rng.integers(3, ref_w - 3))
            y = int(rng.integers(3, ref_h - 3))
            cv2.circle(ref, (x, y), 2, (255, 255, 255), -1)

        # The "pad" occupies the middle half of the reference image, so measuring the
        # whole image would overstate its size by 2x in each dimension.
        pad_rect = (60.0, 60.0, 120.0, 120.0)
        pad_corners = np.float32(
            [
                [pad_rect[0], pad_rect[1]],
                [pad_rect[0] + pad_rect[2], pad_rect[1]],
                [pad_rect[0] + pad_rect[2], pad_rect[1] + pad_rect[3]],
                [pad_rect[0], pad_rect[1] + pad_rect[3]],
            ]
        )

        frame_h, frame_w = 420, 480
        src = np.float32([[0, 0], [ref_w - 1, 0], [ref_w - 1, ref_h - 1], [0, ref_h - 1]])
        dst = np.float32([[100, 80], [360, 80], [360, 340], [100, 340]])
        h = cv2.getPerspectiveTransform(src, dst)

        frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        warped = cv2.warpPerspective(ref, h, (frame_w, frame_h), flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(
            np.ones((ref_h, ref_w), dtype=np.uint8) * 255, h, (frame_w, frame_h), flags=cv2.INTER_NEAREST
        )
        frame[mask > 0] = warped[mask > 0]

        kw = dict(use_markers=False, min_matches=20, min_inliers=15, min_inlier_ratio=0.3)

        with_pad = ReferenceDetector(reference_bgr=ref, pad_corners_px=pad_corners, **kw)
        whole = ReferenceDetector(reference_bgr=ref, **kw)

        self.assertFalse(with_pad.pad_corners_are_full_image)
        self.assertTrue(whole.pad_corners_are_full_image)
        self.assertAlmostEqual(with_pad.reference_size_px, 120.0, delta=1.0)
        self.assertAlmostEqual(whole.reference_size_px, 239.0, delta=1.0)

        out_pad = with_pad.detect(frame)
        out_whole = whole.detect(frame)
        self.assertTrue(out_pad.detected)
        self.assertTrue(out_whole.detected)

        # Ground truth: where the pad corners actually land in the frame.
        expected = cv2.perspectiveTransform(pad_corners.reshape(-1, 1, 2), h).reshape(4, 2)
        got = np.asarray(out_pad.quad_xy, dtype=np.float32).reshape(4, 2)
        self.assertLess(float(np.max(np.linalg.norm(got - expected, axis=1))), 8.0)

        # The pad spans half the reference image, so measuring the whole image
        # overstates the size by ~2x -- exactly the scale error being fixed.
        self.assertAlmostEqual(out_whole.ref_size_px / out_pad.ref_size_px, 2.0, delta=0.15)

    def test_contour_fallback_detects_square_like_board(self) -> None:
        # Create a reference with few features so ORB is likely to be ineffective.
        ref_h, ref_w = 200, 200
        ref = np.zeros((ref_h, ref_w, 3), dtype=np.uint8)
        cv2.rectangle(ref, (10, 10), (ref_w - 10, ref_h - 10), (255, 255, 255), 3)

        frame_h, frame_w = 360, 420
        frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

        # Place a rotated square on the frame and add an extra "tablecloth" extension.
        rect = ((220.0, 190.0), (170.0, 170.0), 25.0)
        box = cv2.boxPoints(rect).astype(np.int32)
        cv2.drawContours(frame, [box], 0, (255, 255, 255), 3)
        cv2.rectangle(frame, (180, 40), (260, 190), (255, 255, 255), 3)

        det = ReferenceDetector(reference_bgr=ref, use_markers=False, min_matches=2000, min_inliers=2000)
        out = det.detect(frame)

        self.assertTrue(out.detected)
        self.assertIsNotNone(out.quad_xy)
        self.assertEqual(out.mode, "contours")
        self.assertGreater(out.ref_size_px, 50.0)
        self.assertAlmostEqual(float(out.ref_width_px), float(out.ref_height_px), delta=5.0)


if __name__ == "__main__":
    unittest.main()
