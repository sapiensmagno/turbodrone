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
