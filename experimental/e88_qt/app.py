import sys
import threading
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, QTimer
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from e88_autopilot.autostabilizer import AutoStabilizer, StabilizerConfig, StabilizerTelemetry
from turbodrone import Drone


class _AutostabilizerWorker(QThread):
    def __init__(self, drone: Drone, *, cfg: StabilizerConfig, duration_sec: Optional[float]) -> None:
        super().__init__()
        self._drone = drone
        self._cfg = cfg
        self._duration_sec = duration_sec

        self._latest_lock = threading.Lock()
        self._latest: Optional[StabilizerTelemetry] = None

        self._stabilizer: Optional[AutoStabilizer] = None
        self._error: Optional[BaseException] = None

    def run(self) -> None:
        try:
            self._stabilizer = AutoStabilizer(self._drone, cfg=self._cfg, telemetry_sink=self._on_telemetry)
            self._stabilizer.activate()
            self._stabilizer.run(duration_sec=self._duration_sec)
        except BaseException as e:
            self._error = e

    def _on_telemetry(self, t: StabilizerTelemetry) -> None:
        with self._latest_lock:
            self._latest = t

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
        self._autopilot_running = False

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.image_label = QLabel(self)
        self.image_label.setFixedSize(640, 480)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black; border: 1px solid gray;")
        self.image_label.setText("Loading RTSP stream...")
        self.layout.addWidget(self.image_label, alignment=Qt.AlignCenter)

        top_row = QHBoxLayout()

        self.cam1_button = QPushButton("Cam 1")
        self.cam1_button.setFixedSize(100, 40)
        self.cam1_button.clicked.connect(lambda: self._drone.switch_camera(1))
        top_row.addWidget(self.cam1_button)

        self.cam2_button = QPushButton("Cam 2")
        self.cam2_button.setFixedSize(100, 40)
        self.cam2_button.clicked.connect(lambda: self._drone.switch_camera(2))
        top_row.addWidget(self.cam2_button)

        self.autopilot_start_button = QPushButton("Start autostabilizer")
        self.autopilot_start_button.setFixedSize(160, 40)
        self.autopilot_start_button.clicked.connect(self._start_autopilot)
        top_row.addWidget(self.autopilot_start_button)

        self.autopilot_stop_button = QPushButton("Stop autostabilizer")
        self.autopilot_stop_button.setFixedSize(160, 40)
        self.autopilot_stop_button.clicked.connect(self._stop_autopilot)
        self.autopilot_stop_button.setEnabled(False)
        top_row.addWidget(self.autopilot_stop_button)

        top_row.addStretch(1)
        self.layout.addLayout(top_row)

        bottom_row = QHBoxLayout()

        self.autopilot_cfg_group = QGroupBox("Autostabilizer")
        self.autopilot_cfg_form = QFormLayout(self.autopilot_cfg_group)

        self.cfg_enable_takeoff = QCheckBox()
        self.autopilot_cfg_form.addRow("Enable takeoff", self.cfg_enable_takeoff)

        self.cfg_takeoff_throttle = QDoubleSpinBox()
        self.cfg_takeoff_throttle.setRange(0.0, 100.0)
        self.cfg_takeoff_throttle.setValue(100.0)
        self.autopilot_cfg_form.addRow("Takeoff throttle", self.cfg_takeoff_throttle)

        self.cfg_takeoff_duration = QDoubleSpinBox()
        self.cfg_takeoff_duration.setRange(0.0, 10.0)
        self.cfg_takeoff_duration.setSingleStep(0.1)
        self.cfg_takeoff_duration.setValue(0.5)
        self.autopilot_cfg_form.addRow("Takeoff duration (s)", self.cfg_takeoff_duration)

        self.cfg_climb_throttle = QDoubleSpinBox()
        self.cfg_climb_throttle.setRange(0.0, 100.0)
        self.cfg_climb_throttle.setValue(70.0)
        self.autopilot_cfg_form.addRow("Climb throttle", self.cfg_climb_throttle)

        self.cfg_climb_duration = QDoubleSpinBox()
        self.cfg_climb_duration.setRange(0.0, 10.0)
        self.cfg_climb_duration.setSingleStep(0.1)
        self.cfg_climb_duration.setValue(1.5)
        self.autopilot_cfg_form.addRow("Climb duration (s)", self.cfg_climb_duration)

        self.cfg_settle_good_frames = QSpinBox()
        self.cfg_settle_good_frames.setRange(0, 60)
        self.cfg_settle_good_frames.setValue(5)
        self.autopilot_cfg_form.addRow("Settle good frames", self.cfg_settle_good_frames)

        self.cfg_base_throttle = QDoubleSpinBox()
        self.cfg_base_throttle.setRange(0.0, 100.0)
        self.cfg_base_throttle.setValue(50.0)
        self.autopilot_cfg_form.addRow("Base throttle", self.cfg_base_throttle)

        self.cfg_rate_hz = QDoubleSpinBox()
        self.cfg_rate_hz.setRange(1.0, 60.0)
        self.cfg_rate_hz.setValue(20.0)
        self.autopilot_cfg_form.addRow("Cmd rate (Hz)", self.cfg_rate_hz)

        self.cfg_use_kalman = QCheckBox()
        self.cfg_use_kalman.setChecked(True)
        self.autopilot_cfg_form.addRow("Use Kalman", self.cfg_use_kalman)

        self.cfg_sigma_a = QDoubleSpinBox()
        self.cfg_sigma_a.setRange(0.1, 500.0)
        self.cfg_sigma_a.setValue(25.0)
        self.autopilot_cfg_form.addRow("Kalman sigma_a", self.cfg_sigma_a)

        self.cfg_sigma_v = QDoubleSpinBox()
        self.cfg_sigma_v.setRange(0.1, 500.0)
        self.cfg_sigma_v.setValue(60.0)
        self.autopilot_cfg_form.addRow("Kalman sigma_v", self.cfg_sigma_v)

        self.cfg_min_quality = QDoubleSpinBox()
        self.cfg_min_quality.setRange(0.0, 1.0)
        self.cfg_min_quality.setSingleStep(0.01)
        self.cfg_min_quality.setValue(0.15)
        self.autopilot_cfg_form.addRow("Min quality", self.cfg_min_quality)

        self.cfg_max_cmd = QDoubleSpinBox()
        self.cfg_max_cmd.setRange(0.0, 1.0)
        self.cfg_max_cmd.setSingleStep(0.01)
        self.cfg_max_cmd.setValue(0.35)
        self.cfg_max_cmd.valueChanged.connect(lambda v: self.traj_widget.set_max_cmd(float(v)))
        self.autopilot_cfg_form.addRow("Max cmd", self.cfg_max_cmd)

        self.cfg_kp_vx = QDoubleSpinBox()
        self.cfg_kp_vx.setDecimals(6)
        self.cfg_kp_vx.setRange(0.0, 1.0)
        self.cfg_kp_vx.setValue(0.003)
        self.autopilot_cfg_form.addRow("Kp vx", self.cfg_kp_vx)

        self.cfg_kp_vy = QDoubleSpinBox()
        self.cfg_kp_vy.setDecimals(6)
        self.cfg_kp_vy.setRange(0.0, 1.0)
        self.cfg_kp_vy.setValue(0.003)
        self.autopilot_cfg_form.addRow("Kp vy", self.cfg_kp_vy)

        self.cfg_ki_vx = QDoubleSpinBox()
        self.cfg_ki_vx.setDecimals(6)
        self.cfg_ki_vx.setRange(0.0, 1.0)
        self.cfg_ki_vx.setValue(0.0005)
        self.autopilot_cfg_form.addRow("Ki vx", self.cfg_ki_vx)

        self.cfg_ki_vy = QDoubleSpinBox()
        self.cfg_ki_vy.setDecimals(6)
        self.cfg_ki_vy.setRange(0.0, 1.0)
        self.cfg_ki_vy.setValue(0.0005)
        self.autopilot_cfg_form.addRow("Ki vy", self.cfg_ki_vy)

        self.cfg_deadband = QDoubleSpinBox()
        self.cfg_deadband.setRange(0.0, 100.0)
        self.cfg_deadband.setValue(3.0)
        self.autopilot_cfg_form.addRow("Deadband (px/s)", self.cfg_deadband)

        self.cfg_roll_sign = QDoubleSpinBox()
        self.cfg_roll_sign.setRange(-1.0, 1.0)
        self.cfg_roll_sign.setSingleStep(2.0)
        self.cfg_roll_sign.setValue(-1.0)
        self.autopilot_cfg_form.addRow("Roll sign", self.cfg_roll_sign)

        self.cfg_pitch_sign = QDoubleSpinBox()
        self.cfg_pitch_sign.setRange(-1.0, 1.0)
        self.cfg_pitch_sign.setSingleStep(2.0)
        self.cfg_pitch_sign.setValue(-1.0)
        self.autopilot_cfg_form.addRow("Pitch sign", self.cfg_pitch_sign)

        self.cfg_duration = QDoubleSpinBox()
        self.cfg_duration.setRange(0.0, 600.0)
        self.cfg_duration.setValue(0.0)
        self.autopilot_cfg_form.addRow("Duration (s, 0=inf)", self.cfg_duration)

        bottom_row.addWidget(self.autopilot_cfg_group)

        self.traj_widget = _TrajectoryWidget(self)
        bottom_row.addWidget(self.traj_widget)

        self.layout.addLayout(bottom_row)

        self.status_label = QLabel("")
        self.layout.addWidget(self.status_label)

        self.setFocusPolicy(Qt.StrongFocus)

        self._video_timer = QTimer(self)
        self._video_timer.timeout.connect(self._tick_video)
        self._video_timer.start(33)

        self._autopilot_timer = QTimer(self)
        self._autopilot_timer.timeout.connect(self._tick_autopilot)
        self._autopilot_timer.start(30)

        self._control_timer = QTimer(self)
        self._control_timer.timeout.connect(self._tick_controls)
        self._control_timer.start(30)

    def closeEvent(self, event):
        try:
            self._stop_autopilot()

            w = self._autopilot_worker
            if w is not None and w.isRunning():
                w.wait(2000)

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
            self._drone.calibrate()
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
        if self._autopilot_running:
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
        if self._autopilot_running:
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
            roll_sign=float(self.cfg_roll_sign.value()),
            pitch_sign=float(self.cfg_pitch_sign.value()),
        )

    def _start_autopilot(self) -> None:
        if self._autopilot_worker is not None and self._autopilot_worker.isRunning():
            return

        cfg = self._build_cfg()
        duration = float(self.cfg_duration.value())
        duration_sec = None if duration <= 0.0 else duration

        self._autopilot_worker = _AutostabilizerWorker(self._drone, cfg=cfg, duration_sec=duration_sec)
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

    def _tick_autopilot(self) -> None:
        w = self._autopilot_worker
        if w is None:
            return
        t = w.pop_latest()
        if t is None:
            return

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

            cv2.putText(
                view,
                f"phase {t.phase} q {(t.flow.quality if t.flow is not None else 0.0):.2f}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                view,
                f"vx {t.used_vx_px_s:+.1f} vy {t.used_vy_px_s:+.1f} px/s",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                view,
                f"cmd roll {t.cmd_roll:+.2f} pitch {t.cmd_pitch:+.2f} thr {t.cmd_throttle:.1f}",
                (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )
            self._show_frame(view, rotate_90_cw=False)

        self.traj_widget.add_sample(x_px=t.pos_x_px, y_px=t.pos_y_px, cmd_roll=t.cmd_roll, cmd_pitch=t.cmd_pitch)

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
