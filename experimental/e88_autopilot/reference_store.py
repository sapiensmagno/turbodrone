from __future__ import annotations

import json
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, replace
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
    # The four corners of the physical pad *within* the reference image, in
    # reference-image pixels, walking around the pad's edge. The ordering contract is
    # about the pad, not the image: edge 0->1 is the pad's physical WIDTH
    # (pad_width_m) and edge 0->3 its physical HEIGHT. Downstream `_quad_sizes`
    # divides pad_width_m by the length of the first edge, so an image-relative
    # ordering would swap the metric axes for a pad rotated 90 degrees in the photo.
    #
    # The reference image is stored uncropped so ORB has surrounding texture to match
    # against, but pad_width_m/pad_height_m describe only the pad. Without this quad
    # the detector measures the projected footprint of the whole reference image and
    # divides it by the pad's physical size, which corrupts m_per_px and the derived
    # focal length by the photo/pad extent ratio.
    #
    # Four corners rather than an axis-aligned rectangle because the pad is rarely
    # photographed square-on: a bounding box of a rotated pad overstates its extent
    # by up to sqrt(2). Ordered corners are also what a PnP pose measurement needs.
    pad_quad_px: Optional[Tuple[Tuple[float, float], ...]] = None

    @property
    def pad_dimensions_m(self) -> Tuple[float, float]:
        return float(self.pad_width_m), float(self.pad_height_m)

    @property
    def has_pad_quad(self) -> bool:
        return self.pad_quad_px is not None

    def pad_corners_px(self, *, image_w: int, image_h: int) -> "np.ndarray":
        """Corners of the pad within the reference image, ordered
        top-left, top-right, bottom-right, bottom-left.

        Falls back to the full image extent when pad_quad_px is unset, which
        reproduces the legacy (metrically wrong) behaviour for un-migrated records."""
        if self.pad_quad_px is None:
            w, h = float(image_w - 1), float(image_h - 1)
            return np.float32([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]])
        return np.asarray(self.pad_quad_px, dtype=np.float32).reshape(4, 2)


def parse_pad_quad(raw: Any) -> Optional[Tuple[Tuple[float, float], ...]]:
    """Validate a 4x2 corner list, rejecting degenerate (collinear//zero-area) quads."""
    if raw is None:
        return None
    try:
        pts = np.asarray(raw, dtype=np.float64).reshape(4, 2)
    except Exception:
        return None
    if not np.all(np.isfinite(pts)):
        return None

    # Shoelace area; a collinear or near-zero-area quad cannot define a scale.
    area = 0.5 * abs(float(np.dot(pts[:, 0], np.roll(pts[:, 1], -1)) - np.dot(pts[:, 1], np.roll(pts[:, 0], -1))))
    if area < 1.0:
        return None

    # Area alone accepts a quad with a duplicated point or a concave fourth corner,
    # as long as the remaining triangle is big enough. Such a record would be stored
    # with has_pad_quad=True but rejected by every downstream detection, because
    # `_quad_is_valid()` requires a 4-point convex hull -- an unusable reference that
    # looks correctly configured.
    if len({(round(float(x), 6), round(float(y), 6)) for x, y in pts}) != 4:
        return None
    hull = cv2.convexHull(pts.astype(np.float32).reshape(-1, 1, 2))
    if hull is None or int(hull.reshape(-1, 2).shape[0]) != 4:
        return None

    # The four points forming a convex hull is not enough: clicked out of perimeter
    # order they describe a self-intersecting ("bowtie") quad whose edges are
    # diagonals of the pad, which the detector would then read as physical width and
    # height. Require the ordering itself to traverse the perimeter, i.e. every
    # consecutive turn has the same sign.
    edges = np.roll(pts, -1, axis=0) - pts
    cross = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
    if not (np.all(cross > 0.0) or np.all(cross < 0.0)):
        return None

    return tuple((float(x), float(y)) for x, y in pts)


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
                pad_quad_px=parse_pad_quad(meta.get("pad_quad_px")),
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
        pad_quad_px: Optional[Any] = None,
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
            pad_quad_px=parse_pad_quad(pad_quad_px),
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

        updated = replace(
            r,
            calibration_height_m=float(calibration_height_m),
            calibration_ref_size_px=float(calibration_ref_size_px),
        )
        self.save(updated)

    def update_pad_quad(self, reference_id: str, *, pad_quad_px: Optional[Any]) -> None:
        """Set the pad's corners within the reference image.

        Changing them invalidates any existing altitude calibration, because
        calibration_ref_size_px was measured against the previous extent."""
        r = self.load(reference_id)
        if r is None:
            raise ValueError(f"unknown reference_id: {reference_id}")

        parsed = parse_pad_quad(pad_quad_px)
        if parsed == r.pad_quad_px:
            return

        self.save(
            replace(
                r,
                pad_quad_px=parsed,
                calibration_height_m=None,
                calibration_ref_size_px=None,
            )
        )

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
