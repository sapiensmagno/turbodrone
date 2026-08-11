import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np


_EXPERIMENTAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXPERIMENTAL_ROOT))

from e88_autopilot.frame_recorder import FrameRecorder


def _frame(i: int, *, h: int = 48, w: int = 64) -> np.ndarray:
    f = np.zeros((h, w, 3), dtype=np.uint8)
    f[:, :] = (int(i) % 200) + 30
    return f


class TestFrameRecorder(unittest.TestCase):
    def test_writes_video_and_aligned_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rec = FrameRecorder(session_dir=Path(td), fps=20.0)
            rec.start()
            for i in range(10):
                rec.submit(_frame(i), seq=i + 1, ts=1000.0 + 0.05 * i, t_received=500.0 + 0.05 * i)
            stats = rec.stop()

            self.assertEqual(int(stats["frames_written"]), 10)
            self.assertEqual(int(stats["write_errors"]), 0)
            self.assertTrue(rec.path.exists())
            self.assertGreater(rec.path.stat().st_size, 0)

            lines = [json.loads(l) for l in rec.sidecar_path.read_text().splitlines() if l.strip()]
            self.assertEqual(len(lines), 10)
            # frame_index must be dense and ordered so it indexes into the video.
            self.assertEqual([r["frame_index"] for r in lines], list(range(10)))
            self.assertEqual([r["seq"] for r in lines], list(range(1, 11)))
            self.assertAlmostEqual(lines[3]["ts"], 1000.15, places=6)

            cap = cv2.VideoCapture(str(rec.path))
            try:
                self.assertTrue(cap.isOpened())
                n = 0
                while True:
                    ok, _ = cap.read()
                    if not ok:
                        break
                    n += 1
            finally:
                cap.release()
            self.assertEqual(n, 10)

    def test_submit_never_blocks_when_queue_is_full(self) -> None:
        """The control loop must never wait on disk I/O. Overflow is dropped and
        counted, not blocked on."""
        with tempfile.TemporaryDirectory() as td:
            rec = FrameRecorder(session_dir=Path(td), fps=20.0, queue_size=2)
            # Deliberately not started: nothing drains the queue.
            rec._thread = object()  # type: ignore[assignment]

            t0 = time.monotonic()
            for i in range(200):
                rec.submit(_frame(i), seq=i, ts=float(i), t_received=float(i))
            elapsed = time.monotonic() - t0

            self.assertLess(elapsed, 2.0)
            self.assertGreater(int(rec.stats()["frames_dropped"]), 0)

    def test_submit_before_start_is_a_no_op(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rec = FrameRecorder(session_dir=Path(td))
            rec.submit(_frame(0), seq=1, ts=1.0, t_received=1.0)
            self.assertEqual(int(rec.stats()["frames_written"]), 0)
            self.assertFalse(rec.path.exists())

    def test_resolution_change_is_dropped_not_written(self) -> None:
        """A mid-session camera switch changes frame size; writing it would corrupt
        the stream."""
        with tempfile.TemporaryDirectory() as td:
            rec = FrameRecorder(session_dir=Path(td), fps=20.0)
            rec.start()
            rec.submit(_frame(0, h=48, w=64), seq=1, ts=1.0, t_received=1.0)
            time.sleep(0.2)  # let the writer establish the frame size
            rec.submit(_frame(1, h=96, w=128), seq=2, ts=2.0, t_received=2.0)
            stats = rec.stop()

            self.assertEqual(int(stats["frames_written"]), 1)
            self.assertEqual(int(stats["frames_dropped"]), 1)

    def test_stop_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rec = FrameRecorder(session_dir=Path(td), fps=20.0)
            rec.start()
            rec.submit(_frame(0), seq=1, ts=1.0, t_received=1.0)
            first = rec.stop()
            second = rec.stop()
            self.assertEqual(int(first["frames_written"]), int(second["frames_written"]))


if __name__ == "__main__":
    unittest.main()
