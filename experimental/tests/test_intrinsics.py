import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


_EXPERIMENTAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXPERIMENTAL_ROOT))

from e88_autopilot.intrinsics import (
    CameraIntrinsics,
    IntrinsicsCollector,
    calibrate_from_corners,
    find_chessboard_corners,
    load_intrinsics,
    object_points,
    resolve_camera_geometry,
    save_intrinsics,
)


PATTERN = (9, 6)
SQUARE_M = 0.025


def _render_chessboard_view(
    *, rvec, tvec, camera_matrix, dist, image_size=(640, 480), pattern=PATTERN, square_m=SQUARE_M
):
    """Project a chessboard at a known pose and draw it, so calibration can be
    tested against intrinsics we chose ourselves."""
    w, h = image_size
    img = np.full((h, w, 3), 255, dtype=np.uint8)

    cols, rows = pattern
    # Draw the full square grid (one more than the inner-corner count each way).
    sq = []
    for j in range(rows + 1):
        for i in range(cols + 1):
            if (i + j) % 2 == 0:
                continue
            sq.append(
                np.float32(
                    [
                        [(i - 1) * square_m, (j - 1) * square_m, 0.0],
                        [i * square_m, (j - 1) * square_m, 0.0],
                        [i * square_m, j * square_m, 0.0],
                        [(i - 1) * square_m, j * square_m, 0.0],
                    ]
                )
            )
    for quad in sq:
        proj, _ = cv2.projectPoints(quad, rvec, tvec, camera_matrix, dist)
        cv2.fillConvexPoly(img, proj.reshape(-1, 2).astype(np.int32), (0, 0, 0))
    return img


def _truth_intrinsics():
    mtx = np.array([[520.0, 0.0, 318.0], [0.0, 520.0, 242.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.zeros((5,), dtype=np.float64)
    return mtx, dist


def _poses():
    out = []
    for tx, ty, rx, ry in [
        (-0.10, -0.07, 0.0, 0.0),
        (-0.02, -0.07, 0.20, 0.0),
        (-0.10, 0.00, 0.0, 0.20),
        (-0.02, 0.00, -0.20, 0.10),
        (-0.06, -0.04, 0.15, -0.15),
        (-0.12, -0.02, -0.10, -0.10),
        (-0.04, -0.09, 0.10, 0.25),
        (-0.08, -0.05, -0.25, 0.05),
        (-0.05, -0.02, 0.05, -0.05),
        (-0.11, -0.08, -0.05, 0.15),
    ]:
        out.append((np.array([rx, ry, 0.0]), np.array([tx, ty, 0.42])))
    return out


class TestIntrinsics(unittest.TestCase):
    def test_recovers_known_intrinsics_from_rendered_views(self) -> None:
        """Calibration is graded against intrinsics we chose, not against itself."""
        mtx, dist = _truth_intrinsics()

        corner_sets = []
        for rvec, tvec in _poses():
            img = _render_chessboard_view(rvec=rvec, tvec=tvec, camera_matrix=mtx, dist=dist)
            corners = find_chessboard_corners(img, pattern_size=PATTERN)
            if corners is not None:
                corner_sets.append(corners)

        self.assertGreaterEqual(len(corner_sets), 6, "renderer failed to produce detectable boards")

        intr = calibrate_from_corners(
            corner_sets,
            image_size=(640, 480),
            pattern_size=PATTERN,
            square_size_m=SQUARE_M,
            camera_id="test",
        )

        self.assertAlmostEqual(intr.fx, 520.0, delta=15.0)
        self.assertAlmostEqual(intr.fy, 520.0, delta=15.0)
        self.assertAlmostEqual(intr.cx, 318.0, delta=15.0)
        self.assertAlmostEqual(intr.cy, 242.0, delta=15.0)
        self.assertLess(intr.rms_reproj_error_px, 1.0)
        self.assertAlmostEqual(intr.focal_length_px, 520.0, delta=15.0)

    def test_object_points_grid_is_metric(self) -> None:
        objp = object_points(pattern_size=(3, 2), square_size_m=0.02)
        self.assertEqual(objp.shape, (6, 3))
        np.testing.assert_allclose(objp[0], [0.0, 0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(objp[1], [0.02, 0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(objp[3], [0.0, 0.02, 0.0], atol=1e-9)

    def test_collector_rejects_near_duplicate_views(self) -> None:
        """Pose diversity, not view count, is what constrains the model."""
        mtx, dist = _truth_intrinsics()
        rvec, tvec = np.array([0.0, 0.0, 0.0]), np.array([-0.10, -0.07, 0.42])
        img = _render_chessboard_view(rvec=rvec, tvec=tvec, camera_matrix=mtx, dist=dist)

        c = IntrinsicsCollector(pattern_size=PATTERN, square_size_m=SQUARE_M)
        ok1, _ = c.try_add(img)
        ok2, reason2 = c.try_add(img.copy())

        self.assertTrue(ok1)
        self.assertFalse(ok2)
        self.assertIn("similar", reason2)
        self.assertEqual(c.n_views, 1)

    def test_collector_accepts_a_tilt_at_the_same_position(self) -> None:
        """Tilting the board in place barely moves its centroid, so a centroid-only
        rule would reject it -- yet out-of-plane tilt is precisely what constrains
        focal length and distortion."""
        mtx, dist = _truth_intrinsics()
        tvec = np.array([-0.10, -0.07, 0.42])
        flat = _render_chessboard_view(rvec=np.array([0.0, 0.0, 0.0]), tvec=tvec, camera_matrix=mtx, dist=dist)
        tilted = _render_chessboard_view(rvec=np.array([0.20, 0.0, 0.0]), tvec=tvec, camera_matrix=mtx, dist=dist)

        # The centroids are close enough that the old 40 px rule would have rejected it.
        ca = find_chessboard_corners(flat, pattern_size=PATTERN)
        cb = find_chessboard_corners(tilted, pattern_size=PATTERN)
        assert ca is not None and cb is not None
        centroid_shift = float(
            np.linalg.norm(
                np.asarray(ca).reshape(-1, 2).mean(axis=0) - np.asarray(cb).reshape(-1, 2).mean(axis=0)
            )
        )
        self.assertLess(centroid_shift, 40.0)

        c = IntrinsicsCollector(pattern_size=PATTERN, square_size_m=SQUARE_M)
        self.assertTrue(c.try_add(flat)[0])
        ok, _ = c.try_add(tilted)
        self.assertTrue(ok, "a meaningful tilt in place must be accepted")
        self.assertEqual(c.n_views, 2)

    def test_coverage_report_flags_a_tilt_free_view_set(self) -> None:
        """A set that only slides the board around has low tilt spread; RMS alone
        would not reveal that focal length and distortion are weakly constrained."""
        mtx, dist = _truth_intrinsics()

        flat_only = IntrinsicsCollector(pattern_size=PATTERN, square_size_m=SQUARE_M)
        for tx in (-0.13, -0.10, -0.07, -0.04, -0.01):
            img = _render_chessboard_view(
                rvec=np.array([0.0, 0.0, 0.0]), tvec=np.array([tx, -0.07, 0.42]), camera_matrix=mtx, dist=dist
            )
            flat_only.try_add(img)

        varied = IntrinsicsCollector(pattern_size=PATTERN, square_size_m=SQUARE_M)
        for rvec, tvec in _poses():
            img = _render_chessboard_view(rvec=rvec, tvec=tvec, camera_matrix=mtx, dist=dist)
            varied.try_add(img)

        flat_cov = flat_only.coverage_report()
        varied_cov = varied.coverage_report()

        self.assertGreater(flat_cov["n_views"], 1)
        self.assertGreater(varied_cov["n_views"], 1)
        self.assertLess(flat_cov["tilt_spread"], 0.02)
        self.assertGreater(varied_cov["tilt_spread"], flat_cov["tilt_spread"] * 3)

    def test_diverse_pose_set_is_fully_accepted(self) -> None:
        """The rejection threshold must not throw away a deliberately varied set."""
        mtx, dist = _truth_intrinsics()
        c = IntrinsicsCollector(pattern_size=PATTERN, square_size_m=SQUARE_M)
        detected = 0
        for rvec, tvec in _poses():
            img = _render_chessboard_view(rvec=rvec, tvec=tvec, camera_matrix=mtx, dist=dist)
            if find_chessboard_corners(img, pattern_size=PATTERN) is not None:
                detected += 1
                c.try_add(img)
        self.assertEqual(c.n_views, detected)

    def test_collector_rejects_frames_without_a_board(self) -> None:
        c = IntrinsicsCollector(pattern_size=PATTERN, square_size_m=SQUARE_M)
        ok, reason = c.try_add(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertFalse(ok)
        self.assertIn("not found", reason)

    def test_calibrate_requires_minimum_views(self) -> None:
        c = IntrinsicsCollector(pattern_size=PATTERN, square_size_m=SQUARE_M)
        with self.assertRaises(ValueError):
            c.calibrate(camera_id="test")

    def test_save_and_load_round_trip_keyed_by_camera_and_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "intrinsics.json"
            a = _make_intrinsics(camera_id="cam1", w=640, h=480, fx=500.0)
            b = _make_intrinsics(camera_id="cam2", w=640, h=480, fx=400.0)
            save_intrinsics(a, path=p)
            save_intrinsics(b, path=p)

            got_a = load_intrinsics(camera_id="cam1", image_w=640, image_h=480, path=p)
            got_b = load_intrinsics(camera_id="cam2", image_w=640, image_h=480, path=p)
            assert got_a is not None and got_b is not None
            self.assertAlmostEqual(got_a.fx, 500.0, places=6)
            self.assertAlmostEqual(got_b.fx, 400.0, places=6)

            self.assertIsNone(load_intrinsics(camera_id="cam9", image_w=640, image_h=480, path=p))

    def test_load_rescales_to_a_different_resolution(self) -> None:
        """Focal length and principal point scale with resolution; distortion
        coefficients are on normalized coordinates and must not."""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "intrinsics.json"
            save_intrinsics(_make_intrinsics(camera_id="cam2", w=640, h=480, fx=520.0), path=p)

            got = load_intrinsics(camera_id="cam2", image_w=320, image_h=240, path=p)
            assert got is not None
            self.assertAlmostEqual(got.fx, 260.0, places=6)
            self.assertAlmostEqual(got.cx, 160.0, places=6)
            self.assertAlmostEqual(got.k1, -0.25, places=6)

            strict = load_intrinsics(camera_id="cam2", image_w=320, image_h=240, path=p, allow_rescale=False)
            self.assertIsNone(strict)

    def test_resolve_geometry_prefers_calibration_over_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "intrinsics.json"
            save_intrinsics(_make_intrinsics(camera_id="cam2", w=640, h=480, fx=520.0), path=p)

            f_px, principal, note = resolve_camera_geometry(
                camera_id="cam2", image_w=640, image_h=480, path=p
            )
            self.assertAlmostEqual(f_px, 520.0, places=6)
            self.assertEqual(principal, (320.0, 240.0))
            self.assertIn("calibrated", note)

    def test_resolve_geometry_explicit_override_wins(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "intrinsics.json"
            save_intrinsics(_make_intrinsics(camera_id="cam2", w=640, h=480, fx=520.0), path=p)

            f_px, principal, note = resolve_camera_geometry(
                camera_id="cam2", image_w=640, image_h=480, focal_length_px_override=300.0, path=p
            )
            self.assertAlmostEqual(f_px, 300.0, places=6)
            self.assertIsNone(principal)
            self.assertIn("explicit", note)

    def test_resolve_geometry_reports_zero_when_uncalibrated(self) -> None:
        """f = 0 makes the derotation model raise rather than silently running a
        different algorithm."""
        with tempfile.TemporaryDirectory() as td:
            f_px, principal, note = resolve_camera_geometry(
                camera_id="cam2", image_w=640, image_h=480, path=Path(td) / "none.json"
            )
            self.assertEqual(f_px, 0.0)
            self.assertIsNone(principal)
            self.assertIn("no calibration", note)

    def test_resolve_geometry_notes_a_rescale(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "intrinsics.json"
            save_intrinsics(_make_intrinsics(camera_id="cam2", w=640, h=480, fx=520.0), path=p)

            f_px, principal, note = resolve_camera_geometry(
                camera_id="cam2", image_w=320, image_h=240, path=p
            )
            self.assertAlmostEqual(f_px, 260.0, places=6)
            self.assertEqual(principal, (160.0, 120.0))
            self.assertIn("rescaled", note)

    def test_load_missing_file_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(
                load_intrinsics(camera_id="cam2", image_w=640, image_h=480, path=Path(td) / "nope.json")
            )


def _make_intrinsics(*, camera_id: str, w: int, h: int, fx: float) -> CameraIntrinsics:
    return CameraIntrinsics(
        camera_id=camera_id,
        image_w=w,
        image_h=h,
        fx=fx,
        fy=fx,
        cx=w / 2.0,
        cy=h / 2.0,
        k1=-0.25,
        k2=0.1,
        p1=0.0,
        p2=0.0,
        k3=0.0,
        rms_reproj_error_px=0.3,
        n_views=10,
        pattern_cols=9,
        pattern_rows=6,
        square_size_m=0.025,
        created_at_ts=1.0,
    )


if __name__ == "__main__":
    unittest.main()
