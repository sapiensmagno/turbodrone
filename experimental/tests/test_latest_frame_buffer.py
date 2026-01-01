import sys
import time
import unittest
from pathlib import Path
from queue import Empty, Queue

import numpy as np


_EXPERIMENTAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXPERIMENTAL_ROOT))

from e88_autopilot.autostabilizer import LatestFrameBuffer


class _FakeDrone:
    def __init__(self) -> None:
        self._q: "Queue[tuple[np.ndarray, float]]" = Queue()

    def push(self, frame: np.ndarray, ts: float) -> None:
        self._q.put((frame, float(ts)))

    def get_frame_with_timestamp(self, timeout: float):
        try:
            return self._q.get(timeout=float(timeout))
        except Empty:
            return None


class TestLatestFrameBuffer(unittest.TestCase):
    def test_latest_frame_overwrites_and_counts_drops(self) -> None:
        drone = _FakeDrone()
        buf = LatestFrameBuffer(drone, timeout_sec=0.01)
        buf.start()
        try:
            base_ts = 100.0
            frame0 = np.zeros((2, 2, 3), dtype=np.uint8)
            drone.push(frame0, base_ts)

            # Wait for first frame.
            deadline = time.time() + 0.5
            item = None
            while time.time() < deadline and item is None:
                item = buf.get_latest()
                time.sleep(0.001)

            self.assertIsNotNone(item)
            assert item is not None

            frame, ts, _t_received, seq, dropped = item
            self.assertEqual(seq, 1)
            self.assertEqual(dropped, 0)
            self.assertAlmostEqual(ts, base_ts, places=6)
            np.testing.assert_array_equal(frame, frame0)

            # Push a burst of frames without reading. Only the latest should remain.
            for i in range(1, 6):
                frame_i = np.full((2, 2, 3), fill_value=i, dtype=np.uint8)
                drone.push(frame_i, base_ts + i)

            # Wait until producer reaches at least seq=6.
            deadline = time.time() + 0.5
            item2 = None
            while time.time() < deadline:
                item2 = buf.get_latest()
                if item2 is not None and int(item2[3]) >= 6:
                    break
                time.sleep(0.001)

            self.assertIsNotNone(item2)
            assert item2 is not None

            frame2, ts2, _t_received2, seq2, dropped2 = item2
            self.assertEqual(seq2, 6)
            # We consumed seq=1, then jumped to seq=6 => dropped 4 frames (2..5)
            self.assertEqual(dropped2, 4)
            self.assertAlmostEqual(ts2, base_ts + 5, places=6)
            np.testing.assert_array_equal(frame2, np.full((2, 2, 3), 5, dtype=np.uint8))

            # Re-reading without new frames should not increase dropped.
            item3 = buf.get_latest()
            self.assertIsNotNone(item3)
            assert item3 is not None
            self.assertEqual(int(item3[3]), 6)
            self.assertEqual(int(item3[4]), 4)
        finally:
            buf.stop()
