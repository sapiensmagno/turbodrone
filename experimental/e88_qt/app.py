import os
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional


def _load_dotenv() -> None:
    root = Path(__file__).resolve().parents[2]
    candidates = [root / ".env", root / "experimental" / ".env"]
    for p in candidates:
        if not p.exists():
            continue
        try:
            for raw in p.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip()
                if len(v) >= 2 and ((v[0] == '"' and v[-1] == '"') or (v[0] == "'" and v[-1] == "'")):
                    v = v[1:-1]
                if k and (k not in os.environ):
                    os.environ[k] = v
        except Exception:
            pass


_load_dotenv()

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QSettings, QThread, QTimer
from PyQt5.QtGui import QColor, QImage, QKeySequence, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QShortcut,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from e88_autopilot.calibration import load_calibration, run_stationary_calibration, save_calibration
from e88_autopilot.autostabilizer import AutoStabilizer, StabilizerConfig, StabilizerTelemetry
from e88_autopilot.reference_store import ReferenceStore
from e88_autopilot.visual_scale import VisualScaleEstimator
from e88_autopilot.session_recorder import SessionRecorder, build_default_meta, measure_icmp_ping_rtt_ms
from e88.config import E88Config
from turbodrone import Drone


class _AutostabilizerWorker(QThread):
    def __init__(
        self,
        drone: Drone,
        *,
        cfg: StabilizerConfig,
        duration_sec: Optional[float],
        recordings_dir: Path,
        initial_flow_sign_state: int,
        initial_control_sign_state: int,
        initial_sign_notes: str,
    ) -> None:
        super().__init__()
        self._drone = drone
        self._cfg = cfg
        self._duration_sec = duration_sec

        self._recordings_dir = Path(recordings_dir)
        self._session_recorder: Optional[SessionRecorder] = None
        self._session_dir: Optional[Path] = None

        self._net_rtt_ms: Optional[float] = None

        self._flow_sign_state = int(initial_flow_sign_state)
        self._control_sign_state = int(initial_control_sign_state)
        self._sign_notes = str(initial_sign_notes)

        self._latest_lock = threading.Lock()
        self._latest: Optional[StabilizerTelemetry] = None

        self._stabilizer: Optional[AutoStabilizer] = None
        self._error: Optional[BaseException] = None

    @staticmethod
    def _tri_to_optional_bool(state: int) -> Optional[bool]:
        if int(state) == 1:
            return None
        if int(state) == 2:
            return True
        return False

    def update_sign_verification(self, *, flow_state: int, control_state: int, notes: str) -> None:
        self._flow_sign_state = int(flow_state)
        self._control_sign_state = int(control_state)
        self._sign_notes = str(notes)

        r = self._session_recorder
        if r is None:
            return

        r.update_meta({"sign_verification": self._build_sign_verification(updated_at=time.time())})

    def _build_sign_verification(self, *, updated_at: Optional[float]) -> dict:
        return {
            "flow_sign_ok": self._tri_to_optional_bool(self._flow_sign_state),
            "control_sign_ok": self._tri_to_optional_bool(self._control_sign_state),
            "notes": str(self._sign_notes),
            "updated_at": updated_at,
        }

    @staticmethod
    def _get_ping_host() -> str:
        return str(E88Config().drone_ip)

    def _start_session(self) -> None:
        host = self._get_ping_host()
        self._net_rtt_ms = measure_icmp_ping_rtt_ms(host)

        self._recordings_dir.mkdir(parents=True, exist_ok=True)
        self._session_recorder = SessionRecorder(base_dir=self._recordings_dir)

        meta = build_default_meta(cfg=self._cfg, net_rtt_ms=self._net_rtt_ms, mode="qt_autostabilizer")
        meta["sign_verification"] = self._build_sign_verification(updated_at=None)
        self._session_dir = self._session_recorder.start(meta=meta)

    def _close_session(self) -> None:
        r = self._session_recorder
        self._session_recorder = None
        if r is not None:
            r.close()

    def run(self) -> None:
        try:
            self._start_session()

            self._stabilizer = AutoStabilizer(self._drone, cfg=self._cfg, telemetry_sink=self._on_telemetry)
            self._stabilizer.activate()
            self._stabilizer.run(duration_sec=self._duration_sec)
        except BaseException as e:
            self._error = e
        finally:
            self._close_session()

    def _on_telemetry(self, t: StabilizerTelemetry) -> None:
        with self._latest_lock:
            self._latest = t

        r = self._session_recorder
        if r is None:
            return

        r.write_sample(self._telemetry_to_sample(t))

    @staticmethod
    def _telemetry_to_sample(t: StabilizerTelemetry) -> dict:
        return {
            "phase": str(t.phase),
            "timestamp": float(t.timestamp),
            "pos_x_px": float(t.pos_x_px),
            "pos_y_px": float(t.pos_y_px),
            "flow": t.flow,
            "kalman": t.kalman,
            "cond_vx_px_s": float(t.kf_input_vx_px_s),
            "cond_vy_px_s": float(t.kf_input_vy_px_s),
            "cond_gated": bool(t.kf_gated),
            "used_vx_px_s": float(t.used_vx_px_s),
            "used_vy_px_s": float(t.used_vy_px_s),
            "cmd_roll": float(t.cmd_roll),
            "cmd_pitch": float(t.cmd_pitch),
            "cmd_throttle": float(t.cmd_throttle),
            "t_loop_start": float(t.t_loop_start),
            "t_frame_received": float(t.t_frame_received),
            "t_flow_start": float(t.t_flow_start),
            "t_flow_end": float(t.t_flow_end),
            "t_kf_end": float(t.t_kf_end),
            "t_ctrl_end": float(t.t_ctrl_end),
            "t_cmd_sent": float(t.t_cmd_sent),
            "dt_flow_ms": float(t.dt_flow_ms),
            "dt_total_ms": float(t.dt_total_ms),
            "frame_age_ms": float(t.frame_age_ms),
            "frame_stale_ms": float(t.frame_stale_ms),
            "frame_seq": int(t.frame_seq),
            "frame_is_new": bool(t.frame_is_new),
            "frames_dropped": int(t.frames_dropped),
            "estimated_latency_ms": float(t.estimated_latency_ms),
            "loop_rate_hz": float(t.loop_rate_hz),
            "frame_rate_hz": float(t.frame_rate_hz),
            "visual_scale_enabled": bool(getattr(t, "visual_scale_enabled", False)),
            "visual_scale_error": str(getattr(t, "visual_scale_error", "")),
            "altitude_est_m": None if t.altitude_est_m is None else float(t.altitude_est_m),
            "altitude_source": str(t.altitude_source),
            "ref_detected": bool(t.ref_detected),
            "ref_width_px": float(t.ref_width_px),
            "ref_height_px": float(t.ref_height_px),
            "ref_size_px": float(t.ref_size_px),
            "vx_m_s": None if t.vx_m_s is None else float(t.vx_m_s),
            "vy_m_s": None if t.vy_m_s is None else float(t.vy_m_s),
            "used_vx_m_s": None if t.used_vx_m_s is None else float(t.used_vx_m_s),
            "used_vy_m_s": None if t.used_vy_m_s is None else float(t.used_vy_m_s),
            "scale_stable": bool(t.scale_stable),
            "ref_mode": str(t.ref_mode),
            "ref_n_matches": int(t.ref_n_matches),
            "ref_n_inliers": int(t.ref_n_inliers),
            "ref_inlier_ratio": float(t.ref_inlier_ratio),
            "ref_reproj_error_px": float(t.ref_reproj_error_px),
        }

    def pop_latest(self) -> Optional[StabilizerTelemetry]:
        with self._latest_lock:
            t = self._latest
            self._latest = None
        return t

    def request_stop(self) -> None:
        s = self._stabilizer
        if s is None:
            return
        s.request_stop()

    @property
    def error(self) -> Optional[BaseException]:
        return self._error

    @property
    def session_dir(self) -> Optional[Path]:
        return self._session_dir

    @property
    def net_rtt_ms(self) -> Optional[float]:
        return self._net_rtt_ms


class _CalibrationWorker(QThread):
    def __init__(self, drone: Drone, *, duration_sec: float, min_quality: float) -> None:
        super().__init__()
        self._drone = drone
        self._duration_sec = float(duration_sec)
        self._min_quality = float(min_quality)

        self._result = None
        self._error: Optional[BaseException] = None

    def run(self) -> None:
        try:
            r = run_stationary_calibration(
                self._drone,
                duration_sec=float(self._duration_sec),
                min_quality=float(self._min_quality),
            )
            save_calibration(r)
            self._result = r
        except BaseException as e:
            self._error = e

    @property
    def result(self):
        return self._result

    @property
    def error(self) -> Optional[BaseException]:
        return self._error


class _TrajectoryWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(360, 360)
        self._pos = []
        self._cmd_roll = 0.0
        self._cmd_pitch = 0.0
        self._max_cmd = 0.35

    def reset(self) -> None:
        self._pos = []
        self._cmd_roll = 0.0
        self._cmd_pitch = 0.0
        self.update()

    def set_max_cmd(self, v: float) -> None:
        self._max_cmd = float(max(1e-6, v))

    def add_sample(self, *, x_px: float, y_px: float, cmd_roll: float, cmd_pitch: float) -> None:
        self._pos.append((float(x_px), float(y_px)))
        if len(self._pos) > 800:
            self._pos = self._pos[-800:]
        self._cmd_roll = float(cmd_roll)
        self._cmd_pitch = float(cmd_pitch)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(10, 10, 10))

        w = float(self.width())
        h = float(self.height())
        if w < 2 or h < 2:
            return

        if not self._pos:
            p.setPen(QPen(QColor(180, 180, 180), 1))
            p.drawText(self.rect(), Qt.AlignCenter, "No motion yet")
            return

        xs = [v[0] for v in self._pos]
        ys = [v[1] for v in self._pos]
        min_x, max_x = float(min(xs)), float(max(xs))
        min_y, max_y = float(min(ys)), float(max(ys))
        span_x = max(1.0, max_x - min_x)
        span_y = max(1.0, max_y - min_y)
        center_x = 0.5 * (min_x + max_x)
        center_y = 0.5 * (min_y + max_y)
        scale = 0.45 * min(w / span_x, h / span_y)

        def map_pt(x: float, y: float):
            sx = (x - center_x) * scale + (w * 0.5)
            sy = (h * 0.5) - ((y - center_y) * scale)
            return sx, sy

        grid_pen = QPen(QColor(35, 35, 35), 1)
        p.setPen(grid_pen)
        for i in range(1, 5):
            x = (w * i) / 5.0
            y = (h * i) / 5.0
            p.drawLine(int(x), 0, int(x), int(h))
            p.drawLine(0, int(y), int(w), int(y))

        axis_pen = QPen(QColor(80, 80, 80), 1)
        p.setPen(axis_pen)
        p.drawLine(int(w * 0.5), 0, int(w * 0.5), int(h))
        p.drawLine(0, int(h * 0.5), int(w), int(h * 0.5))

        traj_pen = QPen(QColor(0, 180, 255), 2)
        p.setPen(traj_pen)
        last = None
        for (x, y) in self._pos:
            sx, sy = map_pt(x, y)
            if last is not None:
                p.drawLine(int(last[0]), int(last[1]), int(sx), int(sy))
            last = (sx, sy)

        cur_x, cur_y = self._pos[-1]
        cx, cy = map_pt(cur_x, cur_y)
        p.setPen(QPen(QColor(255, 255, 255), 1))
        p.setBrush(QColor(255, 255, 255))
        p.drawEllipse(int(cx) - 3, int(cy) - 3, 6, 6)

        base = max(50.0, max(span_x, span_y))
        arrow_len = 0.20 * base
        dx = (self._cmd_roll / self._max_cmd) * arrow_len
        dy = (self._cmd_pitch / self._max_cmd) * arrow_len
        ax, ay = map_pt(cur_x + dx, cur_y + dy)
        p.setPen(QPen(QColor(255, 220, 0), 3))
        p.drawLine(int(cx), int(cy), int(ax), int(ay))


class E88QtControllerWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("E88 Pro Controller")
        self.setGeometry(100, 100, 1100, 850)

        self._drone = Drone(protocol="E88")
        self._drone.connect()

        self._accel = 50
        self._decel = 5

        self._roll = 128
        self._pitch = 128
        self._throttle = 128
        self._yaw = 128

        self._autopilot_worker: Optional[_AutostabilizerWorker] = None
        self._last_autopilot: Optional[StabilizerTelemetry] = None
        self._autopilot_running = False

        self._diag_window_sec = 1.0
        self._diag_series = {
            "dt_total_ms": deque(),
            "dt_flow_ms": deque(),
            "frame_age_ms": deque(),
            "frame_stale_ms": deque(),
            "estimated_latency_ms": deque(),
            "loop_rate_hz": deque(),
            "frame_rate_hz": deque(),
            "frame_seq": deque(),
            "frames_dropped": deque(),
            "t_loop_start": deque(),
            "t_frame_received": deque(),
        }
        self._last_diag_update_t: Optional[float] = None

        self._calibration_worker: Optional[_CalibrationWorker] = None
        self._calibration_running = False
        self._calibration_started_at = 0.0

        self._settings = QSettings("turbodrone", "e88_qt")

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        top_row = QHBoxLayout()

        self.cam1_button = QPushButton("Cam 1")
        self.cam1_button.setFixedSize(100, 40)
        self.cam1_button.setCheckable(True)
        top_row.addWidget(self.cam1_button)

        self.cam2_button = QPushButton("Cam 2")
        self.cam2_button.setFixedSize(100, 40)
        self.cam2_button.setCheckable(True)
        top_row.addWidget(self.cam2_button)

        self._cam_button_group = QButtonGroup(self)
        self._cam_button_group.setExclusive(True)
        self._cam_button_group.addButton(self.cam1_button)
        self._cam_button_group.addButton(self.cam2_button)
        self._selected_cam = 1
        self.cam1_button.setChecked(True)
        self.cam1_button.clicked.connect(lambda: self._select_camera(1))
        self.cam2_button.clicked.connect(lambda: self._select_camera(2))

        self.autopilot_start_button = QPushButton("Start autostabilizer")
        self.autopilot_start_button.setFixedSize(160, 40)
        self.autopilot_start_button.clicked.connect(self._start_autopilot)
        top_row.addWidget(self.autopilot_start_button)

        self.emergency_land_button = QPushButton("EMERGENCY LAND")
        self.emergency_land_button.setFixedSize(170, 40)
        self.emergency_land_button.setStyleSheet("background-color: #b00020; color: white; font-weight: bold;")
        self.emergency_land_button.clicked.connect(self._emergency_land)
        top_row.addWidget(self.emergency_land_button)

        self.calibrate_button = QPushButton("Calibrate (still)")
        self.calibrate_button.setFixedSize(140, 40)
        self.calibrate_button.clicked.connect(self._start_calibration)
        top_row.addWidget(self.calibrate_button)

        self.gyro_calib_button = QPushButton("Calibrate gyro")
        self.gyro_calib_button.setFixedSize(140, 40)
        self.gyro_calib_button.clicked.connect(self._calibrate_gyro)
        top_row.addWidget(self.gyro_calib_button)

        self.autopilot_stop_button = QPushButton("Stop autostabilizer")
        self.autopilot_stop_button.setFixedSize(160, 40)
        self.autopilot_stop_button.clicked.connect(self._stop_autopilot)
        self.autopilot_stop_button.setEnabled(False)
        top_row.addWidget(self.autopilot_stop_button)

        top_row.addStretch(1)
        self.layout.addLayout(top_row)

        self.status_label = QLabel("")
        self.status_label.setTextFormat(Qt.PlainText)
        self.layout.addWidget(self.status_label)

        video_row = QHBoxLayout()

        self.traj_widget = _TrajectoryWidget(self)
        video_row.addWidget(self.traj_widget)

        self.image_label = QLabel(self)
        self.image_label.setFixedSize(640, 480)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black; border: 1px solid gray;")
        self.image_label.setText("Loading RTSP stream...")
        video_row.addWidget(self.image_label, alignment=Qt.AlignCenter)

        self.layout.addLayout(video_row)

        bottom_grid = QGridLayout()

        self.autopilot_cfg_group = QGroupBox("Autostabilizer")
        self.autopilot_cfg_form_left = QFormLayout()
        self.autopilot_cfg_form_right = QFormLayout()
        autopilot_cfg_cols = QHBoxLayout(self.autopilot_cfg_group)
        autopilot_cfg_cols.addLayout(self.autopilot_cfg_form_left)
        autopilot_cfg_cols.addLayout(self.autopilot_cfg_form_right)

        cfg_defaults = StabilizerConfig()

        self.cfg_enable_takeoff = QCheckBox()
        self.cfg_enable_takeoff.setChecked(bool(cfg_defaults.enable_takeoff))
        self.autopilot_cfg_form_left.addRow("Enable takeoff", self.cfg_enable_takeoff)

        self.cfg_takeoff_throttle = QDoubleSpinBox()
        self.cfg_takeoff_throttle.setRange(0.0, 100.0)
        self.cfg_takeoff_throttle.setValue(float(cfg_defaults.takeoff_throttle))
        self.autopilot_cfg_form_left.addRow("Takeoff throttle", self.cfg_takeoff_throttle)

        self.cfg_takeoff_duration = QDoubleSpinBox()
        self.cfg_takeoff_duration.setRange(0.0, 10.0)
        self.cfg_takeoff_duration.setSingleStep(0.1)
        self.cfg_takeoff_duration.setValue(float(cfg_defaults.takeoff_duration_sec))
        self.autopilot_cfg_form_left.addRow("Takeoff duration (s)", self.cfg_takeoff_duration)

        self.cfg_climb_throttle = QDoubleSpinBox()
        self.cfg_climb_throttle.setRange(0.0, 100.0)
        self.cfg_climb_throttle.setValue(float(cfg_defaults.climb_throttle))
        self.autopilot_cfg_form_left.addRow("Climb throttle", self.cfg_climb_throttle)

        self.cfg_climb_duration = QDoubleSpinBox()
        self.cfg_climb_duration.setRange(0.0, 10.0)
        self.cfg_climb_duration.setSingleStep(0.1)
        self.cfg_climb_duration.setValue(float(cfg_defaults.climb_duration_sec))
        self.autopilot_cfg_form_left.addRow("Climb duration (s)", self.cfg_climb_duration)

        self.cfg_settle_good_frames = QSpinBox()
        self.cfg_settle_good_frames.setRange(0, 60)
        self.cfg_settle_good_frames.setValue(int(cfg_defaults.settle_good_frames))
        self.autopilot_cfg_form_left.addRow("Settle good frames", self.cfg_settle_good_frames)

        self.cfg_base_throttle = QDoubleSpinBox()
        self.cfg_base_throttle.setRange(0.0, 100.0)
        self.cfg_base_throttle.setValue(float(cfg_defaults.base_throttle))
        self.autopilot_cfg_form_left.addRow("Base throttle", self.cfg_base_throttle)

        self.cfg_rate_hz = QDoubleSpinBox()
        self.cfg_rate_hz.setRange(1.0, 60.0)
        self.cfg_rate_hz.setValue(float(cfg_defaults.cmd_rate_hz))
        self.autopilot_cfg_form_left.addRow("Cmd rate (Hz)", self.cfg_rate_hz)

        self.cfg_use_kalman = QCheckBox()
        self.cfg_use_kalman.setChecked(bool(cfg_defaults.use_kalman))
        self.autopilot_cfg_form_left.addRow("Use Kalman", self.cfg_use_kalman)

        self.cfg_sigma_a = QDoubleSpinBox()
        self.cfg_sigma_a.setRange(0.1, 500.0)
        self.cfg_sigma_a.setValue(float(cfg_defaults.kalman_sigma_a))
        self.autopilot_cfg_form_left.addRow("Kalman sigma_a", self.cfg_sigma_a)

        self.cfg_sigma_v = QDoubleSpinBox()
        self.cfg_sigma_v.setRange(0.1, 500.0)
        self.cfg_sigma_v.setValue(float(cfg_defaults.kalman_sigma_v))
        self.autopilot_cfg_form_left.addRow("Kalman sigma_v", self.cfg_sigma_v)

        self.cfg_min_quality = QDoubleSpinBox()
        self.cfg_min_quality.setRange(0.0, 1.0)
        self.cfg_min_quality.setSingleStep(0.01)
        self.cfg_min_quality.setValue(float(cfg_defaults.min_quality))
        self.autopilot_cfg_form_left.addRow("Min quality", self.cfg_min_quality)

        self.cfg_max_cmd = QDoubleSpinBox()
        self.cfg_max_cmd.setRange(0.0, 1.0)
        self.cfg_max_cmd.setSingleStep(0.01)
        self.cfg_max_cmd.setValue(float(cfg_defaults.max_cmd))
        self.cfg_max_cmd.valueChanged.connect(lambda v: self.traj_widget.set_max_cmd(float(v)))
        self.autopilot_cfg_form_left.addRow("Max cmd", self.cfg_max_cmd)

        self.cfg_kp_vx = QDoubleSpinBox()
        self.cfg_kp_vx.setDecimals(6)
        self.cfg_kp_vx.setRange(0.0, 1.0)
        self.cfg_kp_vx.setValue(float(cfg_defaults.kp_vx))
        self.autopilot_cfg_form_right.addRow("Kp vx", self.cfg_kp_vx)

        self.cfg_kp_vy = QDoubleSpinBox()
        self.cfg_kp_vy.setDecimals(6)
        self.cfg_kp_vy.setRange(0.0, 1.0)
        self.cfg_kp_vy.setValue(float(cfg_defaults.kp_vy))
        self.autopilot_cfg_form_right.addRow("Kp vy", self.cfg_kp_vy)

        self.cfg_ki_vx = QDoubleSpinBox()
        self.cfg_ki_vx.setDecimals(6)
        self.cfg_ki_vx.setRange(0.0, 1.0)
        self.cfg_ki_vx.setValue(float(cfg_defaults.ki_vx))
        self.autopilot_cfg_form_right.addRow("Ki vx", self.cfg_ki_vx)

        self.cfg_ki_vy = QDoubleSpinBox()
        self.cfg_ki_vy.setDecimals(6)
        self.cfg_ki_vy.setRange(0.0, 1.0)
        self.cfg_ki_vy.setValue(float(cfg_defaults.ki_vy))
        self.autopilot_cfg_form_right.addRow("Ki vy", self.cfg_ki_vy)

        self.cfg_deadband = QDoubleSpinBox()
        self.cfg_deadband.setRange(0.0, 100.0)
        self.cfg_deadband.setValue(float(cfg_defaults.deadband_px_s))
        self.autopilot_cfg_form_right.addRow("Deadband (px/s)", self.cfg_deadband)

        self.cfg_est_deadband = QDoubleSpinBox()
        self.cfg_est_deadband.setRange(0.0, 100.0)
        self.cfg_est_deadband.setValue(float(cfg_defaults.estimator_deadband_px_s))
        self.autopilot_cfg_form_right.addRow("Estimator deadband (px/s)", self.cfg_est_deadband)

        self.cfg_roll_sign = QDoubleSpinBox()
        self.cfg_roll_sign.setRange(-1.0, 1.0)
        self.cfg_roll_sign.setSingleStep(2.0)
        self.cfg_roll_sign.setValue(float(cfg_defaults.roll_sign))
        self.autopilot_cfg_form_right.addRow("Roll sign", self.cfg_roll_sign)

        self.cfg_pitch_sign = QDoubleSpinBox()
        self.cfg_pitch_sign.setRange(-1.0, 1.0)
        self.cfg_pitch_sign.setSingleStep(2.0)
        self.cfg_pitch_sign.setValue(float(cfg_defaults.pitch_sign))
        self.autopilot_cfg_form_right.addRow("Pitch sign", self.cfg_pitch_sign)

        self.cfg_duration = QDoubleSpinBox()
        self.cfg_duration.setRange(0.0, 600.0)
        self.cfg_duration.setValue(0.0)
        self.autopilot_cfg_form_right.addRow("Duration (s, 0=inf)", self.cfg_duration)

        self.cfg_calib_duration = QDoubleSpinBox()
        self.cfg_calib_duration.setRange(1.0, 60.0)
        self.cfg_calib_duration.setSingleStep(1.0)
        self.cfg_calib_duration.setValue(10.0)
        self.autopilot_cfg_form_right.addRow("Calib duration (s)", self.cfg_calib_duration)

        self.visual_scale_group = QGroupBox("Visual Scale")
        self.visual_scale_form = QFormLayout(self.visual_scale_group)

        self.cfg_reference_id = QComboBox()
        self.visual_scale_form.addRow("Reference", self.cfg_reference_id)

        ref_buttons_row = QHBoxLayout()
        self.cfg_reference_reload = QPushButton("Reload")
        self.cfg_reference_reload.clicked.connect(self._reload_references)
        self.cfg_reference_delete = QPushButton("Delete")
        self.cfg_reference_delete.clicked.connect(self._delete_selected_reference)
        ref_buttons_row.addWidget(self.cfg_reference_reload)
        ref_buttons_row.addWidget(self.cfg_reference_delete)
        ref_buttons_row.addStretch(1)
        self.visual_scale_form.addRow("", ref_buttons_row)

        self.cfg_use_m_s_control = QCheckBox()
        self.cfg_use_m_s_control.setChecked(bool(cfg_defaults.use_m_s_control))
        self.visual_scale_form.addRow("Use m/s control", self.cfg_use_m_s_control)

        self.ref_pad_type = QLineEdit()
        self.ref_pad_type.setText("pad")
        self.visual_scale_form.addRow("Pad type", self.ref_pad_type)

        self.ref_pad_width_m = QDoubleSpinBox()
        self.ref_pad_width_m.setRange(0.01, 10.0)
        self.ref_pad_width_m.setDecimals(3)
        self.ref_pad_width_m.setValue(0.30)
        self.visual_scale_form.addRow("Pad width (m)", self.ref_pad_width_m)

        self.ref_pad_height_m = QDoubleSpinBox()
        self.ref_pad_height_m.setRange(0.01, 10.0)
        self.ref_pad_height_m.setDecimals(3)
        self.ref_pad_height_m.setValue(0.30)
        self.visual_scale_form.addRow("Pad height (m)", self.ref_pad_height_m)

        self.ref_capture_height_m = QDoubleSpinBox()
        self.ref_capture_height_m.setRange(0.01, 50.0)
        self.ref_capture_height_m.setDecimals(2)
        self.ref_capture_height_m.setValue(0.50)
        self.visual_scale_form.addRow("Ref capture height (m)", self.ref_capture_height_m)

        self.ref_markers_present = QCheckBox()
        self.ref_markers_present.setChecked(False)
        self.visual_scale_form.addRow("Markers present", self.ref_markers_present)

        reg_row = QHBoxLayout()
        self.ref_capture_btn = QPushButton("Capture")
        self.ref_capture_btn.clicked.connect(self._capture_reference_from_camera)
        self.ref_load_btn = QPushButton("From file")
        self.ref_load_btn.clicked.connect(self._register_reference_from_file)
        reg_row.addWidget(self.ref_capture_btn)
        reg_row.addWidget(self.ref_load_btn)
        self.visual_scale_form.addRow("Register", reg_row)

        self.ref_calib_height_m = QDoubleSpinBox()
        self.ref_calib_height_m.setRange(0.01, 50.0)
        self.ref_calib_height_m.setDecimals(2)
        self.ref_calib_height_m.setValue(0.50)
        self.visual_scale_form.addRow("Calib height (m)", self.ref_calib_height_m)

        self.ref_calib_btn = QPushButton("Calibrate altitude")
        self.ref_calib_btn.clicked.connect(self._calibrate_reference_altitude)
        self.visual_scale_form.addRow("", self.ref_calib_btn)

        self.ref_status_label = QLabel("-")
        self.ref_status_label.setTextFormat(Qt.PlainText)
        self.visual_scale_form.addRow("Status", self.ref_status_label)

        self.keyboard_help_group = QGroupBox("Keyboard")
        self.keyboard_help_label = QLabel()
        self.keyboard_help_label.setTextFormat(Qt.PlainText)
        self.keyboard_help_label.setStyleSheet("font-family: monospace;")
        help_layout = QVBoxLayout(self.keyboard_help_group)
        help_layout.addWidget(self.keyboard_help_label)
        self._update_keyboard_help()

        bottom_grid.addWidget(self.autopilot_cfg_group, 0, 0)
        bottom_grid.addWidget(self.visual_scale_group, 0, 1)

        self.diagnostics_group = QGroupBox("Diagnostics")
        self.diagnostics_form = QFormLayout(self.diagnostics_group)

        self.diag_session_label = QLabel("-")
        self.diag_session_label.setTextFormat(Qt.PlainText)
        self.diagnostics_form.addRow("Session", self.diag_session_label)

        self.diag_rtt_label = QLabel("-")
        self.diag_rtt_label.setTextFormat(Qt.PlainText)
        self.diagnostics_form.addRow("RTT", self.diag_rtt_label)

        self.diag_loop_rate_label = QLabel("-")
        self.diag_loop_rate_label.setTextFormat(Qt.PlainText)
        self.diagnostics_form.addRow("Loop rate", self.diag_loop_rate_label)

        self.diag_frame_rate_label = QLabel("-")
        self.diag_frame_rate_label.setTextFormat(Qt.PlainText)
        self.diagnostics_form.addRow("Frame rate", self.diag_frame_rate_label)

        self.diag_loop_jitter_label = QLabel("-")
        self.diag_loop_jitter_label.setTextFormat(Qt.PlainText)
        self.diagnostics_form.addRow("loop dt std", self.diag_loop_jitter_label)

        self.diag_frame_jitter_label = QLabel("-")
        self.diag_frame_jitter_label.setTextFormat(Qt.PlainText)
        self.diagnostics_form.addRow("frame dt std", self.diag_frame_jitter_label)

        self.diag_dt_total_label = QLabel("-")
        self.diag_dt_total_label.setTextFormat(Qt.PlainText)
        self.diagnostics_form.addRow("dt total", self.diag_dt_total_label)

        self.diag_dt_flow_label = QLabel("-")
        self.diag_dt_flow_label.setTextFormat(Qt.PlainText)
        self.diagnostics_form.addRow("dt flow", self.diag_dt_flow_label)

        self.diag_frame_age_label = QLabel("-")
        self.diag_frame_age_label.setTextFormat(Qt.PlainText)
        self.diagnostics_form.addRow("frame age", self.diag_frame_age_label)

        self.diag_frame_stale_label = QLabel("-")
        self.diag_frame_stale_label.setTextFormat(Qt.PlainText)
        self.diagnostics_form.addRow("frame stale", self.diag_frame_stale_label)

        self.diag_drop_pct_label = QLabel("-")
        self.diag_drop_pct_label.setTextFormat(Qt.PlainText)
        self.diagnostics_form.addRow("drop %", self.diag_drop_pct_label)

        self.diag_latency_label = QLabel("-")
        self.diag_latency_label.setTextFormat(Qt.PlainText)
        self.diagnostics_form.addRow("est latency", self.diag_latency_label)

        self.diag_quality_label = QLabel("-")
        self.diag_quality_label.setTextFormat(Qt.PlainText)
        self.diagnostics_form.addRow("quality", self.diag_quality_label)

        self.diag_features_label = QLabel("-")
        self.diag_features_label.setTextFormat(Qt.PlainText)
        self.diagnostics_form.addRow("features", self.diag_features_label)

        self.diag_inlier_label = QLabel("-")
        self.diag_inlier_label.setTextFormat(Qt.PlainText)
        self.diagnostics_form.addRow("inlier ratio", self.diag_inlier_label)

        self.diag_fallback_label = QLabel("-")
        self.diag_fallback_label.setTextFormat(Qt.PlainText)
        self.diagnostics_form.addRow("fallback", self.diag_fallback_label)

        self.diag_altitude_label = QLabel("-")
        self.diag_altitude_label.setTextFormat(Qt.PlainText)
        self.diagnostics_form.addRow("altitude", self.diag_altitude_label)

        self.diag_scale_stable_label = QLabel("-")
        self.diag_scale_stable_label.setTextFormat(Qt.PlainText)
        self.diagnostics_form.addRow("scale stable", self.diag_scale_stable_label)

        self.diag_vel_ms_label = QLabel("-")
        self.diag_vel_ms_label.setTextFormat(Qt.PlainText)
        self.diagnostics_form.addRow("vel (m/s)", self.diag_vel_ms_label)

        self.diag_ref_stats_label = QLabel("-")
        self.diag_ref_stats_label.setTextFormat(Qt.PlainText)
        self.diagnostics_form.addRow("ref stats", self.diag_ref_stats_label)

        bottom_grid.addWidget(self.diagnostics_group, 0, 2, 2, 1)
        bottom_grid.addWidget(self.keyboard_help_group, 1, 0, 1, 2)

        self.sign_flow_cb = QCheckBox("Flow sign OK")
        self.sign_flow_cb.setTristate(True)
        self.sign_flow_cb.setCheckState(Qt.PartiallyChecked)
        self.diagnostics_form.addRow("", self.sign_flow_cb)

        self.sign_control_cb = QCheckBox("Control sign OK")
        self.sign_control_cb.setTristate(True)
        self.sign_control_cb.setCheckState(Qt.PartiallyChecked)

        self.diagnostics_form.addRow("", self.sign_control_cb)

        self.sign_notes_edit = QLineEdit()
        self.sign_notes_edit.setText("")
        self.diagnostics_form.addRow("Notes", self.sign_notes_edit)

        self.sign_save_button = QPushButton("Save")
        self.sign_save_button.clicked.connect(self._save_sign_verification)
        self.diagnostics_form.addRow("", self.sign_save_button)

        self._sc_toggle_autopilot = QShortcut(QKeySequence("P"), self)
        self._sc_toggle_autopilot.activated.connect(self._toggle_autopilot)

        self._sc_emergency_land = QShortcut(QKeySequence(Qt.Key_Escape), self)
        self._sc_emergency_land.activated.connect(self._emergency_land)

        self._video_timer = QTimer(self)
        self._video_timer.timeout.connect(self._tick_video)
        self._video_timer.start(33)

        self._autopilot_timer = QTimer(self)
        self._autopilot_timer.timeout.connect(self._tick_autopilot)
        self._autopilot_timer.start(30)

        self.layout.addLayout(bottom_grid)

        self._control_timer = QTimer(self)
        self._control_timer.timeout.connect(self._tick_controls)
        self._control_timer.start(30)

        saved = load_calibration()
        if saved is not None:
            self.cfg_sigma_v.setValue(float(saved.kalman_sigma_v))
            self.cfg_est_deadband.setValue(float(saved.estimator_deadband_px_s))
            self.cfg_deadband.setValue(float(saved.estimator_deadband_px_s))
            self.status_label.setText(
                f"Loaded calibration: est_deadband {saved.estimator_deadband_px_s:.2f} px/s, sigma_v {saved.kalman_sigma_v:.2f}"
            )

        last_ref = str(self._settings.value("visual_scale/last_reference_id", "") or "").strip()
        self._reload_references(select_id=(None if not last_ref else last_ref))

        self.cfg_reference_id.currentIndexChanged.connect(self._on_reference_selected)
        self._on_reference_selected()

    @staticmethod
    def _format_reference_label(r) -> str:
        ts = float(getattr(r, "created_at_ts", 0.0))
        try:
            prefix = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
        except Exception:
            prefix = "unknown_time"
        rid = str(getattr(r, "reference_id", ""))
        suffix = rid[-6:] if len(rid) >= 6 else rid
        calib = "calib" if (r.calibration_height_m is not None and r.calibration_ref_size_px is not None) else "no_calib"
        return f"{prefix} {r.pad_type} {r.pad_width_m:.2f}x{r.pad_height_m:.2f}m {calib} ({suffix})"

    def _selected_reference_id(self) -> Optional[str]:
        try:
            v = self.cfg_reference_id.currentData()
            if v is None:
                return None
            s = str(v).strip()
            return None if not s else s
        except Exception:
            return None

    def _reload_references(self, _checked: bool = False, *, select_id: Optional[str] = None) -> None:
        prev = self._selected_reference_id()
        self.cfg_reference_id.blockSignals(True)
        try:
            self.cfg_reference_id.clear()
            self.cfg_reference_id.addItem("(none)", None)
            store = ReferenceStore()
            refs = list(store.list())
            refs.sort(key=lambda x: float(x.created_at_ts), reverse=True)
            for r in refs:
                self.cfg_reference_id.addItem(str(self._format_reference_label(r)), str(r.reference_id))

            pick = select_id
            if pick is None:
                pick = prev
            if pick is not None:
                for i in range(self.cfg_reference_id.count()):
                    if str(self.cfg_reference_id.itemData(i)) == str(pick):
                        self.cfg_reference_id.setCurrentIndex(i)
                        break
        finally:
            self.cfg_reference_id.blockSignals(False)

        self._on_reference_selected()

    def _on_reference_selected(self, *_args) -> None:
        ref_id = self._selected_reference_id()
        self._settings.setValue("visual_scale/last_reference_id", "" if ref_id is None else str(ref_id))

        if ref_id is None:
            self.ref_status_label.setText("-")
            return

        store = ReferenceStore()
        record = store.load(ref_id)
        if record is None:
            self.ref_status_label.setText("-")
            return

        if record.calibration_height_m is not None:
            try:
                self.ref_calib_height_m.setValue(float(record.calibration_height_m))
            except Exception:
                pass

        if record.calibration_height_m is not None and record.calibration_ref_size_px is not None:
            self.ref_status_label.setText(
                f"Selected {ref_id}: \ncalibrated height={float(record.calibration_height_m):.2f}m ref_size_px={float(record.calibration_ref_size_px):.1f}"
            )
        else:
            self.ref_status_label.setText(f"Selected {ref_id}: not calibrated")

    def _delete_selected_reference(self, _checked: bool = False) -> None:
        if self._autopilot_running or self._calibration_running:
            return
        ref_id = self._selected_reference_id()
        if ref_id is None:
            return
        r = QMessageBox.question(
            self,
            "Delete reference",
            f"Delete reference {ref_id}? This will remove it from disk.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if r != QMessageBox.Yes:
            return
        try:
            ReferenceStore().delete(ref_id)
        except Exception as e:
            QMessageBox.warning(self, "Delete reference", f"Failed to delete: {type(e).__name__}: {e}")
            return

        self._settings.setValue("visual_scale/last_reference_id", "")
        self._reload_references(select_id=None)

    def _capture_reference_from_camera(self) -> None:
        if self._autopilot_running or self._calibration_running:
            return
        frame = self._drone.get_frame(timeout=1.0)
        if frame is None:
            QMessageBox.warning(self, "Reference", "No frame available from drone")
            return
        self._create_reference_from_image(frame_bgr=frame)

    def _register_reference_from_file(self) -> None:
        if self._autopilot_running or self._calibration_running:
            return
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "Select reference image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp);;All files (*)",
        )
        if not path:
            return
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            QMessageBox.warning(self, "Reference", f"Failed to load image: {path}")
            return
        self._create_reference_from_image(frame_bgr=img)

    def _create_reference_from_image(self, *, frame_bgr: np.ndarray) -> None:
        store = ReferenceStore()
        try:
            r = store.create(
                pad_type=str(self.ref_pad_type.text()).strip() or "pad",
                pad_width_m=float(self.ref_pad_width_m.value()),
                pad_height_m=float(self.ref_pad_height_m.value()),
                reference_capture_height_m=float(self.ref_capture_height_m.value()),
                markers_present=bool(self.ref_markers_present.isChecked()),
                image_bgr=frame_bgr,
            )
        except Exception as e:
            QMessageBox.warning(self, "Reference", f"Failed to create reference: {type(e).__name__}: {e}")
            return

        self.ref_status_label.setText(f"Created reference {r.reference_id}")
        self._reload_references(select_id=str(r.reference_id))

    def _calibrate_reference_altitude(self) -> None:
        if self._autopilot_running or self._calibration_running:
            return
        ref_id = self._selected_reference_id()
        if ref_id is None:
            QMessageBox.warning(self, "Calibration", "Select a reference first")
            return

        store = ReferenceStore()
        record = store.load(ref_id)
        img = store.load_image_bgr(ref_id)
        if record is None or img is None:
            QMessageBox.warning(self, "Calibration", "Failed to load reference record/image")
            return

        frame = self._drone.get_frame(timeout=1.0)
        if frame is None:
            QMessageBox.warning(self, "Calibration", "No frame available from drone")
            return

        try:
            est = VisualScaleEstimator(record=record, reference_image_bgr=img, stable_required_frames=1)
            scale = est.update(frame_bgr=frame, timestamp=float(time.monotonic()), vx_px_s=0.0, vy_px_s=0.0)
        except Exception as e:
            QMessageBox.warning(self, "Calibration", f"Detection failed: {type(e).__name__}: {e}")
            return

        if not bool(scale.ref_detected) or float(scale.ref_size_px) <= 1e-6:
            QMessageBox.warning(self, "Calibration", "Reference not detected in current frame")
            return

        try:
            store.update_calibration(
                ref_id,
                calibration_height_m=float(self.ref_calib_height_m.value()),
                calibration_ref_size_px=float(scale.ref_size_px),
            )
        except Exception as e:
            QMessageBox.warning(self, "Calibration", f"Failed to save calibration: {type(e).__name__}: {e}")
            return

        self.ref_status_label.setText(
            f"Calibrated {ref_id}: height={float(self.ref_calib_height_m.value()):.2f}m ref_size_px={float(scale.ref_size_px):.1f}"
        )
        self._reload_references(select_id=str(ref_id))

    def closeEvent(self, event):
        try:
            self._stop_autopilot()

            w = self._autopilot_worker
            if w is not None and w.isRunning():
                w.wait(2000)

            cw = self._calibration_worker
            if cw is not None and cw.isRunning():
                cw.wait(2000)

            self._drone.close()
        except Exception:
            pass
        event.accept()

    def keyPressEvent(self, event):
        key = event.key()

        if self._autopilot_running and key in {
            Qt.Key_Up,
            Qt.Key_Down,
            Qt.Key_Left,
            Qt.Key_Right,
            Qt.Key_W,
            Qt.Key_S,
            Qt.Key_A,
            Qt.Key_D,
        }:
            event.accept()
            return

        if key == Qt.Key_Up:
            self._pitch = min(200, self._pitch + self._accel)
            event.accept()
            return
        if key == Qt.Key_Down:
            self._pitch = max(50, self._pitch - self._accel)
            event.accept()
            return
        if key == Qt.Key_Left:
            self._roll = max(50, self._roll - self._accel)
            event.accept()
            return
        if key == Qt.Key_Right:
            self._roll = min(200, self._roll + self._accel)
            event.accept()
            return

        if key == Qt.Key_W:
            self._throttle = min(200, self._throttle + self._accel)
            event.accept()
            return
        if key == Qt.Key_S:
            self._throttle = max(50, self._throttle - self._accel)
            event.accept()
            return
        if key == Qt.Key_D:
            self._yaw = min(200, self._yaw + self._accel)
            event.accept()
            return
        if key == Qt.Key_A:
            self._yaw = max(50, self._yaw - self._accel)
            event.accept()
            return

        if key == Qt.Key_Z:
            self._drone.takeoff()
            event.accept()
            return
        if key == Qt.Key_X:
            self._drone.land()
            event.accept()
            return
        if key == Qt.Key_C:
            self._calibrate_gyro()
            event.accept()
            return
        if key == Qt.Key_F:
            self._drone.flip()
            event.accept()
            return
        if key == Qt.Key_H:
            self._drone.toggle_headless()
            event.accept()
            return

        if key == Qt.Key_1:
            self._drone.switch_camera(1)
            event.accept()
            return
        if key == Qt.Key_2:
            self._drone.switch_camera(2)
            event.accept()
            return

        super().keyPressEvent(event)

    def _tick_controls(self) -> None:
        if self._autopilot_running or self._calibration_running:
            return
        self._drone.set_sticks_raw(
            roll=self._roll,
            pitch=self._pitch,
            throttle=self._throttle,
            yaw=self._yaw,
        )

        self._roll = self._decay_to_center(self._roll)
        self._pitch = self._decay_to_center(self._pitch)
        self._throttle = self._decay_to_center(self._throttle)
        self._yaw = self._decay_to_center(self._yaw)

    def _tick_video(self) -> None:
        if self._autopilot_running or self._calibration_running:
            return
        frame = self._drone.get_frame(timeout=0)
        if frame is None:
            return

        self._show_frame(frame, rotate_90_cw=True)

    def _show_frame(self, frame_bgr: np.ndarray, *, rotate_90_cw: bool) -> None:
        rgb_image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if rotate_90_cw:
            rgb_image = cv2.rotate(rgb_image, cv2.ROTATE_90_CLOCKWISE)

        h, w, _ch = rgb_image.shape
        bytes_per_line = rgb_image.strides[0]
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

        p = q_img.scaled(640, 480, Qt.KeepAspectRatio)
        self.image_label.setPixmap(QPixmap.fromImage(p))
        self.image_label.setText("")
        self.image_label.setStyleSheet("background-color: black; border: 1px solid gray;")

    def _build_cfg(self) -> StabilizerConfig:
        return StabilizerConfig(
            enable_takeoff=bool(self.cfg_enable_takeoff.isChecked()),
            takeoff_throttle=float(self.cfg_takeoff_throttle.value()),
            takeoff_duration_sec=float(self.cfg_takeoff_duration.value()),
            climb_throttle=float(self.cfg_climb_throttle.value()),
            climb_duration_sec=float(self.cfg_climb_duration.value()),
            settle_good_frames=int(self.cfg_settle_good_frames.value()),
            base_throttle=float(self.cfg_base_throttle.value()),
            cmd_rate_hz=float(self.cfg_rate_hz.value()),
            use_kalman=bool(self.cfg_use_kalman.isChecked()),
            kalman_sigma_a=float(self.cfg_sigma_a.value()),
            kalman_sigma_v=float(self.cfg_sigma_v.value()),
            min_quality=float(self.cfg_min_quality.value()),
            max_cmd=float(self.cfg_max_cmd.value()),
            kp_vx=float(self.cfg_kp_vx.value()),
            kp_vy=float(self.cfg_kp_vy.value()),
            ki_vx=float(self.cfg_ki_vx.value()),
            ki_vy=float(self.cfg_ki_vy.value()),
            deadband_px_s=float(self.cfg_deadband.value()),
            estimator_deadband_px_s=float(self.cfg_est_deadband.value()),
            roll_sign=float(self.cfg_roll_sign.value()),
            pitch_sign=float(self.cfg_pitch_sign.value()),
            enable_visual_scale=True,
            reference_id=self._selected_reference_id(),
            use_m_s_control=bool(self.cfg_use_m_s_control.isChecked()),
        )

    def _start_autopilot(self) -> None:
        if self._autopilot_worker is not None and self._autopilot_worker.isRunning():
            return
        if self._calibration_running:
            return

        ref_id = self._selected_reference_id()
        if ref_id is None:
            QMessageBox.warning(
                self,
                "Autostabilizer",
                "Select a Reference and calibrate altitude before starting autostabilizer.",
            )
            return

        store = ReferenceStore()
        record = store.load(ref_id)
        if record is None:
            QMessageBox.warning(self, "Autostabilizer", "Failed to load selected reference.")
            return

        if record.calibration_height_m is None or record.calibration_ref_size_px is None:
            QMessageBox.warning(
                self,
                "Autostabilizer",
                "Selected Reference is not calibrated. Click 'Calibrate altitude' first.",
            )
            return

        cfg = self._build_cfg()
        duration = float(self.cfg_duration.value())
        duration_sec = None if duration <= 0.0 else duration

        recordings_dir = Path(__file__).resolve().parents[1] / "e88_autopilot" / "sessions"
        self._autopilot_worker = _AutostabilizerWorker(
            self._drone,
            cfg=cfg,
            duration_sec=duration_sec,
            recordings_dir=recordings_dir,
            initial_flow_sign_state=int(self.sign_flow_cb.checkState()),
            initial_control_sign_state=int(self.sign_control_cb.checkState()),
            initial_sign_notes=str(self.sign_notes_edit.text()),
        )
        self._autopilot_worker.finished.connect(self._on_autopilot_finished)
        self._autopilot_worker.start()
        self._set_autopilot_running(True)
        self.traj_widget.reset()
        self.status_label.setText("Autostabilizer running")

    def _stop_autopilot(self) -> None:
        w = self._autopilot_worker
        if w is None:
            return
        self.status_label.setText("Autostabilizer stopping...")
        w.request_stop()

    def _on_autopilot_finished(self) -> None:
        w = self._autopilot_worker
        self._set_autopilot_running(False)

        try:
            self._drone.send_cmd(roll=0.0, pitch=0.0, yaw=0.0, throttle=float(self.cfg_base_throttle.value()))
        except Exception:
            pass

        self._roll = 128
        self._pitch = 128
        self._yaw = 128

        if w is None:
            self.status_label.setText("")
            return
        if w.error is not None:
            self.status_label.setText(f"Autostabilizer error: {type(w.error).__name__}: {w.error}")
        else:
            self.status_label.setText("Autostabilizer stopped")
        self._autopilot_worker = None

    def _set_autopilot_running(self, running: bool) -> None:
        self._autopilot_running = bool(running)
        self.autopilot_start_button.setEnabled(not self._autopilot_running)
        self.autopilot_stop_button.setEnabled(self._autopilot_running)
        self.autopilot_cfg_group.setEnabled(not self._autopilot_running)
        self.visual_scale_group.setEnabled(not self._autopilot_running)
        self.calibrate_button.setEnabled((not self._autopilot_running) and (not self._calibration_running))
        self.gyro_calib_button.setEnabled((not self._autopilot_running) and (not self._calibration_running))
        self.cam1_button.setEnabled((not self._autopilot_running) and (not self._calibration_running))
        self.cam2_button.setEnabled((not self._autopilot_running) and (not self._calibration_running))

    def _set_calibration_running(self, running: bool) -> None:
        self._calibration_running = bool(running)
        self.autopilot_start_button.setEnabled((not self._calibration_running) and (not self._autopilot_running))
        self.autopilot_stop_button.setEnabled(self._autopilot_running and (not self._calibration_running))
        self.autopilot_cfg_group.setEnabled((not self._autopilot_running) and (not self._calibration_running))
        self.visual_scale_group.setEnabled((not self._autopilot_running) and (not self._calibration_running))
        self.calibrate_button.setEnabled((not self._autopilot_running) and (not self._calibration_running))
        self.gyro_calib_button.setEnabled((not self._autopilot_running) and (not self._calibration_running))
        self.cam1_button.setEnabled((not self._autopilot_running) and (not self._calibration_running))
        self.cam2_button.setEnabled((not self._autopilot_running) and (not self._calibration_running))

    def _calibrate_gyro(self) -> None:
        if self._autopilot_running or self._calibration_running:
            return
        try:
            self._drone.calibrate()
            self.status_label.setText("Gyro calibration requested")
        except Exception as e:
            self.status_label.setText(f"Gyro calibration error: {type(e).__name__}: {e}")

    def _select_camera(self, cam: int) -> None:
        if self._autopilot_running or self._calibration_running:
            return
        cam = int(cam)
        if cam not in (1, 2):
            return
        try:
            self._drone.switch_camera(cam)
            self._selected_cam = int(cam)
            self.status_label.setText(f"Switched to Cam {cam}")
        except Exception as e:
            self.status_label.setText(f"Camera switch error: {type(e).__name__}: {e}")

    def _start_calibration(self) -> None:
        if self._autopilot_running:
            return
        if self._calibration_worker is not None and self._calibration_worker.isRunning():
            return

        duration_sec = float(self.cfg_calib_duration.value())
        min_quality = float(self.cfg_min_quality.value())
        self.status_label.setText("Calibrating... keep drone still")
        self._calibration_started_at = float(time.monotonic())
        self._calibration_worker = _CalibrationWorker(self._drone, duration_sec=duration_sec, min_quality=min_quality)
        self._calibration_worker.finished.connect(self._on_calibration_finished)
        self._set_calibration_running(True)
        self._calibration_worker.start()

    def _on_calibration_finished(self) -> None:
        w = self._calibration_worker
        self._set_calibration_running(False)

        if w is None:
            self.status_label.setText("")
            return
        if w.error is not None:
            self.status_label.setText(f"Calibration error: {type(w.error).__name__}: {w.error}")
            self._calibration_worker = None
            return

        r = w.result
        if r is None:
            self.status_label.setText("Calibration error: no result")
            self._calibration_worker = None
            return

        prev_sigma_v = float(self.cfg_sigma_v.value())
        prev_est_db = float(self.cfg_est_deadband.value())
        prev_db = float(self.cfg_deadband.value())

        self.cfg_sigma_v.setValue(float(r.kalman_sigma_v))
        self.cfg_est_deadband.setValue(float(r.estimator_deadband_px_s))
        self.cfg_deadband.setValue(float(r.estimator_deadband_px_s))

        changed = []
        if abs(prev_sigma_v - float(r.kalman_sigma_v)) > 1e-6:
            changed.append("Kalman sigma_v")
            self._flash_widget(self.cfg_sigma_v)
        if abs(prev_est_db - float(r.estimator_deadband_px_s)) > 1e-6:
            changed.append("Estimator deadband")
            self._flash_widget(self.cfg_est_deadband)
        if abs(prev_db - float(r.estimator_deadband_px_s)) > 1e-6:
            changed.append("Deadband")
            self._flash_widget(self.cfg_deadband)

        changed_str = "" if not changed else (" (updated: " + ", ".join(changed) + ")")
        self.status_label.setText(
            f"Calibration saved. est_deadband {r.estimator_deadband_px_s:.2f} px/s, sigma_v {r.kalman_sigma_v:.2f}{changed_str}"
        )
        self._calibration_worker = None

    def _update_keyboard_help(self) -> None:
        manual = [
            "Arrow keys: roll/pitch",
            "W/S: throttle up/down",
            "A/D: yaw left/right",
        ]
        actions = [
            "Z: takeoff",
            "X: land",
            "C: calibrate gyro",
            "1/2: switch camera",
            "H: toggle headless",
            "F: flip",
            "P: start/stop autostabilizer",
        ]
        emergency = [
            "Esc: emergency land (also stops autostabilizer)",
        ]

        w0 = 34
        w1 = 30

        def _col(s: str, w: int) -> str:
            s2 = str(s)
            if len(s2) > w:
                s2 = s2[: max(0, w - 1)] + "…"
            return s2.ljust(w)

        rows = max(len(manual), len(actions), len(emergency))
        lines = []
        lines.append(_col("Manual control", w0) + "  " + _col("Actions", w1) + "  " + "Emergency")
        lines.append(_col("(disabled during autopilot)", w0) + "  " + _col("", w1) + "  " + "")
        lines.append("")
        for i in range(rows):
            m = manual[i] if i < len(manual) else ""
            a = actions[i] if i < len(actions) else ""
            e = emergency[i] if i < len(emergency) else ""
            lines.append(_col(m, w0) + "  " + _col(a, w1) + "  " + e)

        self.keyboard_help_label.setText("\n".join(lines))

    def _toggle_autopilot(self) -> None:
        if self._calibration_running:
            return
        if self._autopilot_running:
            self._stop_autopilot()
            return
        self._start_autopilot()

    def _emergency_land(self) -> None:
        try:
            self._stop_autopilot()
        except Exception:
            pass

        try:
            self._drone.land()
        except Exception:
            pass

        self.status_label.setText("Emergency land triggered")

    def _flash_widget(self, w: QWidget) -> None:
        try:
            prev = w.styleSheet()
        except Exception:
            prev = ""

        w.setStyleSheet(prev + "\nbackground-color: #fff2a8;")

        def _restore() -> None:
            w.setStyleSheet(prev)

        QTimer.singleShot(2500, _restore)

    def _tick_autopilot(self) -> None:
        w = self._autopilot_worker
        if w is None:
            return

        if w.session_dir is not None:
            self.diag_session_label.setText(str(Path(w.session_dir).name))
        if w.net_rtt_ms is not None:
            self.diag_rtt_label.setText(f"{float(w.net_rtt_ms):.1f} ms")

        t = w.pop_latest()
        if t is None:
            return

        vx_ms = getattr(t, "used_vx_m_s", None)
        vy_ms = getattr(t, "used_vy_m_s", None)
        if vx_ms is None:
            vx_ms = getattr(t, "vx_m_s", None)
        if vy_ms is None:
            vy_ms = getattr(t, "vy_m_s", None)

        now_m = float(time.monotonic())
        cutoff = now_m - float(self._diag_window_sec)

        def _push(k: str, v: float) -> None:
            self._diag_series[k].append((now_m, float(v)))
            while self._diag_series[k] and float(self._diag_series[k][0][0]) < cutoff:
                self._diag_series[k].popleft()

        def _avg(k: str, fallback: float) -> float:
            items = self._diag_series[k]
            if not items:
                return float(fallback)
            return float(sum(float(x[1]) for x in items) / float(len(items)))

        _push("dt_total_ms", float(t.dt_total_ms))
        _push("dt_flow_ms", float(t.dt_flow_ms))
        _push("frame_age_ms", float(t.frame_age_ms))
        _push("frame_stale_ms", float(t.frame_stale_ms))
        _push("estimated_latency_ms", float(t.estimated_latency_ms))
        _push("loop_rate_hz", float(t.loop_rate_hz))
        _push("frame_rate_hz", float(t.frame_rate_hz))
        _push("frame_seq", float(t.frame_seq))
        _push("frames_dropped", float(t.frames_dropped))
        _push("t_loop_start", float(t.t_loop_start))
        _push("t_frame_received", float(t.t_frame_received))

        drop_pct_cum = (100.0 * float(t.frames_dropped) / float(t.frame_seq)) if int(t.frame_seq) > 0 else 0.0
        drop_pct_win: Optional[float] = None
        items_seq = self._diag_series["frame_seq"]
        items_drop = self._diag_series["frames_dropped"]
        if len(items_seq) >= 2 and len(items_drop) >= 2:
            delta_seq = float(items_seq[-1][1]) - float(items_seq[0][1])
            delta_drop = float(items_drop[-1][1]) - float(items_drop[0][1])
            if delta_seq > 0.0:
                drop_pct_win = 100.0 * max(0.0, float(delta_drop)) / float(delta_seq)
        drop_pct_for_overlay = float(drop_pct_win) if drop_pct_win is not None else float(drop_pct_cum)

        def _std_ms_from_monotonic_times(times_s: list[float]) -> Optional[float]:
            if len(times_s) < 3:
                return None
            dts = []
            prev = float(times_s[0])
            for cur in times_s[1:]:
                cur_f = float(cur)
                dt = cur_f - prev
                if dt > 1e-6:
                    dts.append(dt)
                prev = cur_f
            if len(dts) < 2:
                return None
            return float(np.std(np.asarray(dts, dtype=float)) * 1000.0)

        loop_dt_std_ms = _std_ms_from_monotonic_times([float(v) for (_, v) in self._diag_series["t_loop_start"]])

        new_frame_times = []
        last_seq: Optional[int] = None
        for ((_, seq_v), (_, tr_v)) in zip(self._diag_series["frame_seq"], self._diag_series["t_frame_received"]):
            seq_i = int(round(float(seq_v)))
            if last_seq is None or seq_i != int(last_seq):
                new_frame_times.append(float(tr_v))
                last_seq = int(seq_i)
        frame_dt_std_ms = _std_ms_from_monotonic_times(new_frame_times)

        diag_update_period = 0.2
        if self._last_diag_update_t is None or (now_m - float(self._last_diag_update_t)) >= diag_update_period:
            self._last_diag_update_t = float(now_m)

            self.diag_loop_rate_label.setText(f"{_avg('loop_rate_hz', t.loop_rate_hz):.1f} Hz")
            self.diag_frame_rate_label.setText(f"{_avg('frame_rate_hz', t.frame_rate_hz):.1f} Hz")
            self.diag_loop_jitter_label.setText("-" if loop_dt_std_ms is None else f"{loop_dt_std_ms:.1f} ms")
            self.diag_frame_jitter_label.setText("-" if frame_dt_std_ms is None else f"{frame_dt_std_ms:.1f} ms")
            self.diag_dt_total_label.setText(f"{_avg('dt_total_ms', t.dt_total_ms):.1f} ms")
            self.diag_dt_flow_label.setText(f"{_avg('dt_flow_ms', t.dt_flow_ms):.1f} ms")
            self.diag_frame_age_label.setText(f"{_avg('frame_age_ms', t.frame_age_ms):.1f} ms")
            self.diag_frame_stale_label.setText(f"{_avg('frame_stale_ms', t.frame_stale_ms):.1f} ms")
            if drop_pct_win is None:
                self.diag_drop_pct_label.setText(f"{drop_pct_cum:.1f} %")
            else:
                self.diag_drop_pct_label.setText(f"{drop_pct_win:.1f} % (cum {drop_pct_cum:.1f} %)")
            self.diag_latency_label.setText(f"{_avg('estimated_latency_ms', t.estimated_latency_ms):.1f} ms")

            if t.flow is None:
                self.diag_quality_label.setText("-")
                self.diag_features_label.setText("-")
                self.diag_inlier_label.setText("-")
                self.diag_fallback_label.setText("-")
            else:
                self.diag_quality_label.setText(f"{float(t.flow.quality):.2f}")
                self.diag_features_label.setText(f"{int(t.flow.n_tracked)}/{int(t.flow.n_features)}")
                self.diag_inlier_label.setText(f"{float(t.flow.inlier_ratio):.2f}")
                self.diag_fallback_label.setText("1" if bool(t.flow.fallback_used) else "0")

            if t.altitude_est_m is None:
                self.diag_altitude_label.setText("-")
            else:
                self.diag_altitude_label.setText(f"{float(t.altitude_est_m):.2f} m ({str(t.altitude_source)})")

            self.diag_scale_stable_label.setText("1" if bool(t.scale_stable) else "0")
            if vx_ms is None or vy_ms is None:
                self.diag_vel_ms_label.setText("-")
            else:
                self.diag_vel_ms_label.setText(f"vx {float(vx_ms):+.3f}  vy {float(vy_ms):+.3f}")

            if not bool(t.ref_detected):
                vs_err = str(getattr(t, "visual_scale_error", ""))
                vs_on = bool(getattr(t, "visual_scale_enabled", False))
                if vs_on and vs_err:
                    self.diag_ref_stats_label.setText(f"err {vs_err}")
                else:
                    self.diag_ref_stats_label.setText("-")
            else:
                self.diag_ref_stats_label.setText(
                    f"{str(t.ref_mode)} inl {float(t.ref_inlier_ratio):.2f} err {float(t.ref_reproj_error_px):.1f} px"
                )

        frame = t.frame_bgr
        if frame is not None:
            h0, w0 = frame.shape[:2]
            view = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            tracks = t.tracks
            if tracks is not None:
                n = int(min(tracks.prev_xy.shape[0], tracks.next_xy.shape[0], tracks.inliers.shape[0]))
                for i in range(n):
                    x0, y0 = tracks.prev_xy[i]
                    x1, y1 = tracks.next_xy[i]
                    ok = bool(tracks.inliers[i])
                    c = (0, 255, 0) if ok else (0, 0, 255)

                    rx0, ry0 = int(h0 - 1 - float(y0)), int(float(x0))
                    rx1, ry1 = int(h0 - 1 - float(y1)), int(float(x1))
                    cv2.line(view, (rx0, ry0), (rx1, ry1), c, 1)
                    cv2.circle(view, (rx1, ry1), 2, c, -1)

            quad = getattr(t, "ref_quad_xy", None)
            if bool(t.ref_detected) and quad is not None:
                try:
                    q = np.asarray(quad, dtype=np.float32).reshape(4, 2)
                    pts = []
                    for (x, y) in q:
                        pts.append([int(h0 - 1 - float(y)), int(float(x))])
                    poly = np.asarray(pts, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(view, [poly], isClosed=True, color=(0, 255, 0), thickness=2)
                except Exception:
                    pass
            dt_total_ms = float(_avg("dt_total_ms", t.dt_total_ms))
            dt_flow_ms = float(_avg("dt_flow_ms", t.dt_flow_ms))
            frame_age_ms = float(_avg("frame_age_ms", t.frame_age_ms))
            frame_stale_ms = float(_avg("frame_stale_ms", t.frame_stale_ms))

            cv2.putText(
                view,
                f"phase {t.phase} q {(t.flow.quality if t.flow is not None else 0.0):.2f}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                1,
            )
            cv2.putText(
                view,
                "-" if (vx_ms is None or vy_ms is None) else f"vx {float(vx_ms):+.3f} vy {float(vy_ms):+.3f} m/s",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                1,
            )
            cv2.putText(
                view,
                "alt -" if t.altitude_est_m is None else f"alt {float(t.altitude_est_m):.2f} m",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
            cv2.putText(
                view,
                f"kf_in vx {t.kf_input_vx_px_s:+.1f} vy {t.kf_input_vy_px_s:+.1f} px/s gated {int(bool(t.kf_gated))}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
            )
            cv2.putText(
                view,
                f"cmd roll {t.cmd_roll:+.2f} pitch {t.cmd_pitch:+.2f} thr {t.cmd_throttle:.1f}",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )
            cv2.putText(
                view,
                f"dt_total {dt_total_ms:.1f}ms dt_flow {dt_flow_ms:.1f}ms age {frame_age_ms:.1f}ms stale {frame_stale_ms:.1f}ms drop {drop_pct_for_overlay:.1f}%",
                (10, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (180, 180, 180),
                1,
            )
            self._show_frame(view, rotate_90_cw=False)

        self.traj_widget.add_sample(x_px=t.pos_x_px, y_px=t.pos_y_px, cmd_roll=t.cmd_roll, cmd_pitch=t.cmd_pitch)

    def _save_sign_verification(self) -> None:
        w = self._autopilot_worker
        if w is None:
            return
        w.update_sign_verification(
            flow_state=int(self.sign_flow_cb.checkState()),
            control_state=int(self.sign_control_cb.checkState()),
            notes=str(self.sign_notes_edit.text()),
        )

    def _decay_to_center(self, v: int) -> int:
        if v > 128:
            return max(128, v - self._decel)
        if v < 128:
            return min(128, v + self._decel)
        return v


def main() -> None:
    app = QApplication(sys.argv)
    window = E88QtControllerWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
