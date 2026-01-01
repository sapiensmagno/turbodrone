from __future__ import annotations

from dataclasses import dataclass


DEFAULT_DRONE_IP: str = "192.168.1.1"
DEFAULT_RTSP_URL: str = f"rtsp://{DEFAULT_DRONE_IP}:7070/webcam"


@dataclass(frozen=True)
class E88Config:
    rtsp_url: str = DEFAULT_RTSP_URL
    drone_ip: str = DEFAULT_DRONE_IP
    drone_port: int = 7099
    source_port: int = 7099
    control_interval_sec: float = 0.03
    video_reopen_delay_sec: float = 2.0
