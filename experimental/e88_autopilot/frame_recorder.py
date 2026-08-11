from __future__ import annotations

import json
import threading
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


class FrameRecorder:
    """Records the raw camera frames alongside a session's telemetry.

    Sessions previously stored telemetry only, which makes the vision pipeline
    impossible to replay offline: you can see that the velocity estimate went wrong
    but not what the estimator was looking at. Every motion-model comparison then has
    to be re-flown instead of re-run.

    Frames are handed off to a writer thread through a bounded queue. The control
    loop must never block on disk I/O, so when the queue is full the frame is dropped
    and counted rather than waited on -- a gap in the recording is recoverable, a
    stalled control loop is not.

    Output is MJPEG (one self-contained JPEG per frame) rather than an inter-frame
    codec so that a truncated file from a crash or battery pull still decodes up to
    the last complete frame.

    The `frames.jsonl` sidecar carries the identifiers needed to align frames with
    telemetry samples and to reconstruct frame reuse and drop behaviour:
        {"frame_index": int, "seq": int, "ts": float, "t_received": float}
    `frame_index` is the position within the video file, which diverges from `seq`
    whenever a frame is dropped -- so the sidecar, not the frame count, is the source
    of truth for alignment.
    """

    def __init__(
        self,
        *,
        session_dir: Path,
        fps: float = 20.0,
        queue_size: int = 64,
        filename: str = "frames.avi",
        sidecar_filename: str = "frames.jsonl",
    ) -> None:
        self._session_dir = Path(session_dir)
        self._fps = float(max(1.0, fps))
        self._queue_size = int(max(1, queue_size))
        # No JPEG-quality knob: cv2.VIDEOWRITER_PROP_QUALITY has no effect on the
        # MJPG backend in this OpenCV build (measured: identical file size at quality
        # 20 and 95), so exposing it would be a setting that silently does nothing.
        self._path = self._session_dir / str(filename)
        self._sidecar_path = self._session_dir / str(sidecar_filename)

        self._q: "Queue[Optional[Tuple[np.ndarray, int, float, float]]]" = Queue(maxsize=self._queue_size)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

        self._lock = threading.Lock()
        self._writer: Optional[cv2.VideoWriter] = None
        self._sidecar_fp = None
        self._frame_size: Optional[Tuple[int, int]] = None

        self._frames_written = 0
        self._frames_dropped = 0
        self._write_errors = 0

    @property
    def path(self) -> Path:
        return self._path

    @property
    def sidecar_path(self) -> Path:
        return self._sidecar_path

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "frames_written": int(self._frames_written),
                "frames_dropped": int(self._frames_dropped),
                "write_errors": int(self._write_errors),
                "path": str(self._path),
            }

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self._sidecar_fp = self._sidecar_path.open("a", encoding="utf-8")
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="FrameRecorder", daemon=True)
        self._thread.start()

    def submit(self, frame_bgr: np.ndarray, *, seq: int, ts: float, t_received: float) -> None:
        """Hand a frame to the writer thread. Never blocks."""
        if self._thread is None:
            return
        try:
            # The caller may reuse or mutate its buffer, so copy before queueing.
            self._q.put_nowait((frame_bgr.copy(), int(seq), float(ts), float(t_received)))
        except Full:
            with self._lock:
                self._frames_dropped += 1

    def stop(self, *, timeout_sec: float = 5.0) -> Dict[str, Any]:
        t = self._thread
        self._thread = None
        if t is not None:
            self._stop.set()
            try:
                self._q.put_nowait(None)  # wake the writer if it is idle
            except Full:
                pass
            t.join(timeout=float(timeout_sec))
            if t.is_alive():
                # The writer owns the VideoWriter and the sidecar handle and closes
                # them itself. Releasing them from here while it is still draining
                # would let it write to closed handles or reopen and truncate the
                # AVI, corrupting an otherwise good recording. Leaking briefly is
                # strictly safer; the thread is a daemon and will finish or die with
                # the process.
                with self._lock:
                    self._write_errors += 1
        return self.stats()

    def _run(self) -> None:
        try:
            while True:
                try:
                    item = self._q.get(timeout=0.2)
                except Empty:
                    if self._stop.is_set():
                        break
                    continue
                if item is None:
                    break
                frame, seq, ts, t_received = item
                self._write_one(frame, seq, ts, t_received)

            # Drain whatever is still queued so a clean stop does not truncate the tail.
            while True:
                try:
                    item = self._q.get_nowait()
                except Empty:
                    break
                if item is None:
                    continue
                frame, seq, ts, t_received = item
                self._write_one(frame, seq, ts, t_received)
        finally:
            self._release()

    def _release(self) -> None:
        """Close the output files. Called only on the writer thread, which owns them."""
        with self._lock:
            w = self._writer
            self._writer = None
            fp = self._sidecar_fp
            self._sidecar_fp = None
        if w is not None:
            try:
                w.release()
            except Exception:
                pass
        if fp is not None:
            try:
                fp.close()
            except Exception:
                pass

    def _write_one(self, frame: np.ndarray, seq: int, ts: float, t_received: float) -> None:
        """Only ever called on the writer thread.

        `_lock` guards counters and handles, never blocking I/O: `submit()` takes the
        same lock just to count a dropped frame, so holding it across a stalled codec
        or disk write would stall frame acquisition itself."""
        h, w = frame.shape[:2]

        if self._writer is None:
            writer = cv2.VideoWriter(
                str(self._path), cv2.VideoWriter_fourcc(*"MJPG"), float(self._fps), (int(w), int(h))
            )
            if not writer.isOpened():
                with self._lock:
                    self._write_errors += 1
                return
            with self._lock:
                self._frame_size = (int(w), int(h))
                self._writer = writer

        # A mid-session resolution change (e.g. camera switch) would corrupt the
        # stream; drop those frames rather than write garbage.
        if self._frame_size is not None and (int(w), int(h)) != self._frame_size:
            with self._lock:
                self._frames_dropped += 1
            return

        writer = self._writer
        fp = self._sidecar_fp
        if writer is None:
            return

        frame_index = int(self._frames_written)
        try:
            writer.write(frame)
        except Exception:
            with self._lock:
                self._write_errors += 1
            return
        with self._lock:
            self._frames_written += 1

        if fp is not None:
            try:
                fp.write(
                    json.dumps(
                        {
                            "frame_index": int(frame_index),
                            "seq": int(seq),
                            "ts": float(ts),
                            "t_received": float(t_received),
                        }
                    )
                    + "\n"
                )
                fp.flush()
            except Exception:
                with self._lock:
                    self._write_errors += 1
