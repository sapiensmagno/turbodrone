import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


_EXPERIMENTAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXPERIMENTAL_ROOT))

from e88_autopilot.reference_store import ReferenceStore


class TestReferenceStore(unittest.TestCase):
    def test_create_load_list_and_update_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base_dir = Path(td)
            store = ReferenceStore(base_dir=base_dir)

            img = np.zeros((32, 48, 3), dtype=np.uint8)
            img[10:20, 12:36] = 255

            rec = store.create(
                pad_type="test_pad",
                pad_width_m=0.40,
                pad_height_m=0.30,
                reference_capture_height_m=0.50,
                markers_present=False,
                image_bgr=img,
            )

            self.assertTrue((base_dir / rec.reference_id).exists())

            loaded = store.load(rec.reference_id)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.reference_id, rec.reference_id)
            self.assertEqual(loaded.pad_type, "test_pad")
            self.assertAlmostEqual(loaded.pad_width_m, 0.40, places=6)
            self.assertAlmostEqual(loaded.pad_height_m, 0.30, places=6)
            self.assertAlmostEqual(loaded.reference_capture_height_m, 0.50, places=6)
            self.assertFalse(loaded.markers_present)
            self.assertIsNone(loaded.calibration_height_m)
            self.assertIsNone(loaded.calibration_ref_size_px)

            img_loaded = store.load_image_bgr(rec.reference_id)
            self.assertIsNotNone(img_loaded)
            assert img_loaded is not None
            self.assertEqual(img_loaded.shape, img.shape)

            lst = store.list()
            self.assertEqual(len(lst), 1)
            self.assertEqual(lst[0].reference_id, rec.reference_id)

            store.update_calibration(rec.reference_id, calibration_height_m=0.6, calibration_ref_size_px=123.0)
            loaded2 = store.load(rec.reference_id)
            self.assertIsNotNone(loaded2)
            assert loaded2 is not None
            self.assertAlmostEqual(float(loaded2.calibration_height_m), 0.6, places=6)
            self.assertAlmostEqual(float(loaded2.calibration_ref_size_px), 123.0, places=6)

    def _make(self, store, **kw):
        img = np.zeros((32, 48, 3), dtype=np.uint8)
        img[10:20, 12:36] = 255
        return store.create(
            pad_type="test_pad",
            pad_width_m=0.30,
            pad_height_m=0.30,
            reference_capture_height_m=0.50,
            markers_present=False,
            image_bgr=img,
            **kw,
        )

    def test_pad_quad_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = ReferenceStore(base_dir=Path(td))
            quad = [[4.0, 6.0], [24.0, 6.0], [24.0, 18.0], [4.0, 18.0]]
            rec = self._make(store, pad_quad_px=quad)

            loaded = store.load(rec.reference_id)
            assert loaded is not None
            self.assertTrue(loaded.has_pad_quad)
            np.testing.assert_allclose(
                loaded.pad_corners_px(image_w=48, image_h=32), np.float32(quad), atol=1e-5
            )

    def test_rotated_pad_keeps_true_corners_not_a_bounding_box(self) -> None:
        """A bounding box of a 45-degree-rotated pad overstates its extent by ~sqrt(2),
        which would flow straight into m_per_px and the derived focal length."""
        with tempfile.TemporaryDirectory() as td:
            store = ReferenceStore(base_dir=Path(td))
            # A 10x10 square rotated 45 degrees: side 10*sqrt(2)/2 ~ 7.07 apart.
            quad = [[20.0, 10.0], [30.0, 20.0], [20.0, 30.0], [10.0, 20.0]]
            rec = self._make(store, pad_quad_px=quad)

            loaded = store.load(rec.reference_id)
            assert loaded is not None
            got = loaded.pad_corners_px(image_w=48, image_h=32)
            side = float(np.linalg.norm(got[1] - got[0]))
            self.assertAlmostEqual(side, 14.142, delta=0.01)
            # The axis-aligned bounding box would have been 20 px wide.
            self.assertLess(side, 20.0)

    def test_missing_pad_quad_falls_back_to_full_image(self) -> None:
        """Records registered before pad_quad_px existed must still load, falling
        back to the full image extent (the legacy, metrically wrong behaviour)."""
        with tempfile.TemporaryDirectory() as td:
            store = ReferenceStore(base_dir=Path(td))
            rec = self._make(store)

            loaded = store.load(rec.reference_id)
            assert loaded is not None
            self.assertFalse(loaded.has_pad_quad)
            self.assertIsNone(loaded.pad_quad_px)

            np.testing.assert_allclose(
                loaded.pad_corners_px(image_w=48, image_h=32),
                np.float32([[0, 0], [47, 0], [47, 31], [0, 31]]),
                atol=1e-5,
            )

    def test_degenerate_pad_quad_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = ReferenceStore(base_dir=Path(td))
            bad_quads = [
                [[0, 0], [10, 0], [20, 0], [30, 0]],       # collinear
                [[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]],  # ~zero area
                [[0, 0], [1, 1]],                          # wrong shape
                "not a quad",
                # Area alone would accept these, but downstream _quad_is_valid()
                # requires a 4-point convex hull and would reject every detection --
                # a reference that looks configured but can never match.
                [[0, 0], [20, 0], [20, 0], [0, 20]],       # duplicated corner
                [[0, 0], [20, 0], [10, 5], [0, 20]],       # concave fourth corner
                # "Bowtie": four corners of a square clicked out of perimeter order.
                # Convex hull is still 4 points, but the edges are the pad's
                # diagonals, which the detector would read as width and height.
                [[0, 0], [20, 20], [20, 0], [0, 20]],
            ]
            for bad in bad_quads:
                rec = self._make(store, pad_quad_px=bad)
                loaded = store.load(rec.reference_id)
                assert loaded is not None
                self.assertIsNone(loaded.pad_quad_px, f"expected {bad!r} to be rejected")

    def test_changing_pad_quad_invalidates_altitude_calibration(self) -> None:
        """calibration_ref_size_px was measured against the previous pad extent, so
        it is meaningless once the extent changes."""
        with tempfile.TemporaryDirectory() as td:
            store = ReferenceStore(base_dir=Path(td))
            quad = [[4.0, 6.0], [24.0, 6.0], [24.0, 18.0], [4.0, 18.0]]
            rec = self._make(store, pad_quad_px=quad)
            store.update_calibration(rec.reference_id, calibration_height_m=0.7, calibration_ref_size_px=156.0)

            store.update_pad_quad(
                rec.reference_id, pad_quad_px=[[2.0, 2.0], [32.0, 2.0], [32.0, 26.0], [2.0, 26.0]]
            )
            after = store.load(rec.reference_id)
            assert after is not None
            self.assertIsNone(after.calibration_height_m)
            self.assertIsNone(after.calibration_ref_size_px)

    def test_setting_identical_pad_quad_preserves_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = ReferenceStore(base_dir=Path(td))
            quad = [[4.0, 6.0], [24.0, 6.0], [24.0, 18.0], [4.0, 18.0]]
            rec = self._make(store, pad_quad_px=quad)
            store.update_calibration(rec.reference_id, calibration_height_m=0.7, calibration_ref_size_px=156.0)

            store.update_pad_quad(rec.reference_id, pad_quad_px=quad)
            after = store.load(rec.reference_id)
            assert after is not None
            self.assertAlmostEqual(float(after.calibration_ref_size_px), 156.0, places=6)


if __name__ == "__main__":
    unittest.main()
