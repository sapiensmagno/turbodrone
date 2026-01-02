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


if __name__ == "__main__":
    unittest.main()
