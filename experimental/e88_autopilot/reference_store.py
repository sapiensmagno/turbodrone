from __future__ import annotations

import json
import shutil
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class ReferenceRecord:
    reference_id: str
    created_at_ts: float
    pad_type: str
    pad_width_m: float
    pad_height_m: float
    reference_capture_height_m: float
    markers_present: bool
    image_filename: str
    calibration_height_m: Optional[float] = None
    calibration_ref_size_px: Optional[float] = None

    @property
    def pad_dimensions_m(self) -> Tuple[float, float]:
        return float(self.pad_width_m), float(self.pad_height_m)


def default_reference_store_dir() -> Path:
    return Path(__file__).resolve().parent / "references"


class ReferenceStore:
    def __init__(self, *, base_dir: Optional[Path] = None) -> None:
        self._base_dir = Path(default_reference_store_dir() if base_dir is None else base_dir)

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def list(self) -> List[ReferenceRecord]:
        out: List[ReferenceRecord] = []
        if not self._base_dir.exists():
            return out
        for p in sorted(self._base_dir.iterdir()):
            if not p.is_dir():
                continue
            r = self.load(p.name)
            if r is not None:
                out.append(r)
        return out

    def load(self, reference_id: str) -> Optional[ReferenceRecord]:
        d = self._record_dir(reference_id)
        meta_path = d / "meta.json"
        if not meta_path.exists():
            return None
        try:
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return None
            if int(raw.get("version", 0)) != 1:
                return None
            meta = raw.get("record")
            if not isinstance(meta, dict):
                return None
            return ReferenceRecord(
                reference_id=str(meta["reference_id"]),
                created_at_ts=float(meta["created_at_ts"]),
                pad_type=str(meta["pad_type"]),
                pad_width_m=float(meta["pad_width_m"]),
                pad_height_m=float(meta["pad_height_m"]),
                reference_capture_height_m=float(meta["reference_capture_height_m"]),
                markers_present=bool(meta["markers_present"]),
                image_filename=str(meta["image_filename"]),
                calibration_height_m=None if meta.get("calibration_height_m") is None else float(meta["calibration_height_m"]),
                calibration_ref_size_px=None
                if meta.get("calibration_ref_size_px") is None
                else float(meta["calibration_ref_size_px"]),
            )
        except Exception:
            return None

    def load_image_bgr(self, reference_id: str) -> Optional[np.ndarray]:
        r = self.load(reference_id)
        if r is None:
            return None
        p = self._record_dir(reference_id) / r.image_filename
        if not p.exists():
            return None
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            return None
        return img

    def create(
        self,
        *,
        pad_type: str,
        pad_width_m: float,
        pad_height_m: float,
        reference_capture_height_m: float,
        markers_present: bool,
        image_bgr: np.ndarray,
        calibration_height_m: Optional[float] = None,
        calibration_ref_size_px: Optional[float] = None,
    ) -> ReferenceRecord:
        ref_id = uuid.uuid4().hex
        created_at_ts = float(time.time())
        img_filename = "reference.png"

        record = ReferenceRecord(
            reference_id=str(ref_id),
            created_at_ts=float(created_at_ts),
            pad_type=str(pad_type),
            pad_width_m=float(pad_width_m),
            pad_height_m=float(pad_height_m),
            reference_capture_height_m=float(reference_capture_height_m),
            markers_present=bool(markers_present),
            image_filename=str(img_filename),
            calibration_height_m=None if calibration_height_m is None else float(calibration_height_m),
            calibration_ref_size_px=None if calibration_ref_size_px is None else float(calibration_ref_size_px),
        )

        d = self._record_dir(record.reference_id)
        d.mkdir(parents=True, exist_ok=False)

        ok = cv2.imwrite(str(d / img_filename), image_bgr)
        if not ok:
            raise RuntimeError("failed to write reference image")

        self.save(record)
        return record

    def save(self, record: ReferenceRecord) -> None:
        d = self._record_dir(record.reference_id)
        d.mkdir(parents=True, exist_ok=True)

        payload: Dict[str, Any] = {
            "version": 1,
            "record": asdict(record),
        }
        (d / "meta.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def update_calibration(self, reference_id: str, *, calibration_height_m: float, calibration_ref_size_px: float) -> None:
        r = self.load(reference_id)
        if r is None:
            raise ValueError(f"unknown reference_id: {reference_id}")

        updated = ReferenceRecord(
            reference_id=r.reference_id,
            created_at_ts=r.created_at_ts,
            pad_type=r.pad_type,
            pad_width_m=r.pad_width_m,
            pad_height_m=r.pad_height_m,
            reference_capture_height_m=r.reference_capture_height_m,
            markers_present=r.markers_present,
            image_filename=r.image_filename,
            calibration_height_m=float(calibration_height_m),
            calibration_ref_size_px=float(calibration_ref_size_px),
        )
        self.save(updated)

    def delete(self, reference_id: str) -> None:
        d = self._record_dir(reference_id)
        try:
            d_rel = d.resolve().relative_to(self._base_dir.resolve())
        except Exception as e:
            raise ValueError(f"invalid reference_id path: {reference_id}") from e
        if not str(d_rel):
            raise ValueError(f"refusing to delete base_dir: {reference_id}")
        if not d.exists():
            return
        if not d.is_dir():
            raise ValueError(f"reference is not a directory: {reference_id}")
        shutil.rmtree(d)

    def _record_dir(self, reference_id: str) -> Path:
        return self._base_dir / str(reference_id)
