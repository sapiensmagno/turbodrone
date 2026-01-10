from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
import threading
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_jsonable(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool)):
        return v
    if is_dataclass(v):
        return {k: _to_jsonable(val) for k, val in asdict(v).items()}
    if isinstance(v, dict):
        return {str(k): _to_jsonable(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_to_jsonable(x) for x in v]
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            pass

    return str(v)


def telemetry_to_flat_sample(t: Any) -> Dict[str, Any]:
    flow = t.flow
    kal = t.kalman

    return {
        "phase": str(t.phase),
        "timestamp": float(t.timestamp),
        "pos_x_px": float(t.pos_x_px),
        "pos_y_px": float(t.pos_y_px),
        "leash_enabled": bool(getattr(t, "leash_enabled", False)),
        "leash_units": str(getattr(t, "leash_units", "")),
        "leash_state": getattr(t, "leash_state", None),
        "ctl_in_vx": float(getattr(t, "ctl_in_vx", 0.0)),
        "ctl_in_vy": float(getattr(t, "ctl_in_vy", 0.0)),
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
        "flow_valid": bool(flow is not None),
        "flow_dt_sec": None if flow is None else float(flow.dt_sec),
        "flow_dx_px": None if flow is None else float(flow.dx_px),
        "flow_dy_px": None if flow is None else float(flow.dy_px),
        "flow_vx_px_s": None if flow is None else float(flow.vx_px_s),
        "flow_vy_px_s": None if flow is None else float(flow.vy_px_s),
        "flow_quality": None if flow is None else float(flow.quality),
        "flow_n_features": None if flow is None else int(flow.n_features),
        "flow_n_tracked": None if flow is None else int(flow.n_tracked),
        "flow_inlier_ratio": None if flow is None else float(flow.inlier_ratio),
        "flow_fallback_used": False if flow is None else bool(flow.fallback_used),
        "flow_motion_model": "" if flow is None else str(getattr(flow, "motion_model", "")),
        "flow_raw_dx_px": None if flow is None else float(getattr(flow, "raw_dx_px", 0.0)),
        "flow_raw_dy_px": None if flow is None else float(getattr(flow, "raw_dy_px", 0.0)),
        "flow_omega_rad": None if flow is None else float(getattr(flow, "omega_rad", 0.0)),
        "flow_omega_rad_s": None if flow is None else float(getattr(flow, "omega_rad_s", 0.0)),
        "flow_model_rmse_px": None if flow is None else float(getattr(flow, "model_rmse_px", 0.0)),
        "kalman_valid": bool(kal is not None),
        "kalman_t": None if kal is None else float(kal.t),
        "kalman_x_px": None if kal is None else float(kal.x_px),
        "kalman_y_px": None if kal is None else float(kal.y_px),
        "kalman_vx_px_s": None if kal is None else float(kal.vx_px_s),
        "kalman_vy_px_s": None if kal is None else float(kal.vy_px_s),
        "kalman_p_vx": None if kal is None else float(kal.p_vx),
        "kalman_p_vy": None if kal is None else float(kal.p_vy),
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


def _parse_ping_rtt_ms(output: str) -> Optional[float]:
    m = re.search(r"min/avg/max/(?:mdev|stddev) = [0-9.]+/([0-9.]+)/", output)
    if m is not None:
        return float(m.group(1))

    m = re.search(r"time=([0-9.]+) ms", output)
    if m is not None:
        return float(m.group(1))

    return None


def measure_icmp_ping_rtt_ms(host: str, *, count: int = 3, timeout_sec: float = 1.0) -> Optional[float]:
    try:
        cmd = [
            "ping",
            "-c",
            str(int(max(1, count))),
            "-W",
            str(int(max(1, round(float(timeout_sec))))),
            str(host),
        ]
        p = subprocess.run(cmd, capture_output=True, text=True)
        out = (p.stdout or "") + "\n" + (p.stderr or "")

        return _parse_ping_rtt_ms(out)
    except Exception:
        return None


class SessionRecorder:
    def __init__(self, *, base_dir: Path) -> None:
        self._base_dir = Path(base_dir)
        self._session_dir: Optional[Path] = None
        self._meta: Dict[str, Any] = {}
        self._samples_fp = None
        self._lock = threading.Lock()

    @property
    def session_dir(self) -> Optional[Path]:
        return self._session_dir

    def start(self, *, meta: Dict[str, Any]) -> Path:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        session_dir = self._base_dir / f"session_{ts}_{int(time.time() * 1000) % 1000000:06d}"
        session_dir.mkdir(parents=True, exist_ok=False)

        self._session_dir = session_dir
        self._meta = dict(meta)
        self._write_meta_locked()

        self._samples_fp = (session_dir / "samples.jsonl").open("a", encoding="utf-8")
        return session_dir

    def close(self) -> None:
        with self._lock:
            fp = self._samples_fp
            self._samples_fp = None
        if fp is not None:
            try:
                fp.close()
            except Exception:
                pass

    def update_meta(self, updates: Dict[str, Any]) -> None:
        with self._lock:
            self._meta.update(updates)
            self._write_meta_locked()

    def write_sample(self, sample: Dict[str, Any]) -> None:
        with self._lock:
            fp = self._samples_fp
            if fp is None:
                return
            try:
                fp.write(json.dumps(_to_jsonable(sample), ensure_ascii=False) + "\n")
                fp.flush()
            except Exception:
                pass

    def _write_meta_locked(self) -> None:
        sd = self._session_dir
        if sd is None:
            return
        meta_path = sd / "meta.json"
        payload = _to_jsonable(self._meta)
        try:
            meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass


def build_default_meta(*, cfg: Any, net_rtt_ms: Optional[float], mode: str) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "mode": str(mode),
        "created_at": _now_iso_utc(),
        "python": sys.version,
        "platform": platform.platform(),
        "cfg": _to_jsonable(cfg),
        "net_rtt_ms": None if net_rtt_ms is None else float(net_rtt_ms),
        "sign_verification": {
            "flow_sign_ok": None,
            "control_sign_ok": None,
            "notes": "",
            "updated_at": None,
        },
    }
