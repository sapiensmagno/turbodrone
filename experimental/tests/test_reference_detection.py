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
