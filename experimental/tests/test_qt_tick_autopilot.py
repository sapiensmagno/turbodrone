import os
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


_EXPERIMENTAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXPERIMENTAL_ROOT))


class TestQtTickAutopilot(unittest.TestCase):
    def test_tick_autopilot_does_not_crash_when_vel_ms_unavailable(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

        try:
            from PyQt5.QtCore import Qt
            from PyQt5.QtWidgets import QApplication
        except Exception as e:
            self.skipTest(f"PyQt5 not available: {e}")

        # Must be set before QApplication is created.
        QApplication.setAttribute(Qt.AA_UseSoftwareOpenGL, True)

        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        from e88_qt import app as qt_app

        class _FakeDrone:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def connect(self) -> None:
                return

        qt_app.Drone = _FakeDrone

        win = qt_app.E88QtControllerWindow()

        # Avoid GUI/display work in unit test.
        win._show_frame = lambda *args, **kwargs: None  # type: ignore[assignment]
        win.traj_widget.add_sample = lambda *args, **kwargs: None  # type: ignore[assignment]

        frame = np.zeros((240, 320, 3), dtype=np.uint8)

        t = SimpleNamespace(
            dt_total_ms=1.0,
            dt_flow_ms=1.0,
            frame_age_ms=1.0,
            frame_stale_ms=1.0,
            estimated_latency_ms=1.0,
            loop_rate_hz=60.0,
            frame_rate_hz=30.0,
            frame_seq=1,
            frames_dropped=0,
            t_loop_start=float(time.monotonic()),
            t_frame_received=float(time.monotonic()),
            phase="hold",
            flow=None,
            frame_bgr=frame,
            tracks=None,
            ref_detected=False,
            ref_quad_xy=None,
            altitude_est_m=None,
            altitude_source=None,
            scale_stable=False,
            visual_scale_enabled=True,
            visual_scale_error="",
            kf_input_vx_px_s=0.0,
            kf_input_vy_px_s=0.0,
            kf_gated=False,
            cmd_roll=0.0,
            cmd_pitch=0.0,
            cmd_throttle=128.0,
            pos_x_px=0.0,
            pos_y_px=0.0,
        )

        # Force the diagnostics update block to be skipped, which previously caused
        # vx_ms/vy_ms to remain uninitialized while still being used by the overlay.
        win._last_diag_update_t = float(time.monotonic())

        class _FakeWorker:
            session_dir = None
            net_rtt_ms = None

            def __init__(self, telemetry) -> None:
                self._telemetry = telemetry

            def pop_latest(self):
                if self._telemetry is None:
                    return None
                t0 = self._telemetry
                self._telemetry = None
                return t0

        win._autopilot_worker = _FakeWorker(t)

        # Regression: should not raise UnboundLocalError.
        win._tick_autopilot()
