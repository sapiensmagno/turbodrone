from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """Measured camera intrinsics for one camera at one resolution.

    Everything geometric in the autopilot -- metric scale, altitude, and above all
    any attempt to separate camera tilt from translation -- depends on knowing the
    focal length and the lens distortion. Until this is measured, the only available
    focal length comes from the reference-pad calibration, which is an indirect
    estimate that inherits every error in the pad geometry.

    Distortion matters more than it looks. The tilt/translation discriminator in the
    derotation model is the quadratic radial term (1 + u'^2); barrel distortion is
    *also* a radial-quadratic perturbation, so an uncalibrated lens corrupts exactly
    the signal the model relies on.
    """

    camera_id: str
    image_w: int
    image_h: int
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float
    k2: float
    p1: float
    p2: float
    k3: float
    rms_reproj_error_px: float
    n_views: int
    pattern_cols: int
    pattern_rows: int
    square_size_m: float
    created_at_ts: float

    @property
    def focal_length_px(self) -> float:
        """Single focal length for the pinhole model used by the flow estimators."""
        return float(0.5 * (float(self.fx) + float(self.fy)))

    @property
    def camera_matrix(self) -> np.ndarray:
        return np.array(
            [[float(self.fx), 0.0, float(self.cx)], [0.0, float(self.fy), float(self.cy)], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def dist_coeffs(self) -> np.ndarray:
        return np.array([float(self.k1), float(self.k2), float(self.p1), float(self.p2), float(self.k3)], dtype=np.float64)

    def scaled_to(self, *, image_w: int, image_h: int) -> "CameraIntrinsics":
        """Rescale to a different resolution of the same camera.

        Focal length and principal point scale with resolution; the distortion
        coefficients are defined on normalized coordinates and do not."""
        sx = float(image_w) / float(self.image_w)
        sy = float(image_h) / float(self.image_h)
        return CameraIntrinsics(
            camera_id=self.camera_id,
            image_w=int(image_w),
            image_h=int(image_h),
            fx=float(self.fx) * sx,
            fy=float(self.fy) * sy,
            cx=float(self.cx) * sx,
            cy=float(self.cy) * sy,
            k1=self.k1,
            k2=self.k2,
            p1=self.p1,
            p2=self.p2,
            k3=self.k3,
            rms_reproj_error_px=self.rms_reproj_error_px,
            n_views=self.n_views,
            pattern_cols=self.pattern_cols,
            pattern_rows=self.pattern_rows,
            square_size_m=self.square_size_m,
            created_at_ts=self.created_at_ts,
        )


def default_intrinsics_path() -> Path:
    return Path(__file__).resolve().parent / "intrinsics.json"


def save_intrinsics(intr: CameraIntrinsics, *, path: Optional[Path] = None) -> Path:
    """Persist intrinsics, keyed by camera id and resolution.

    Multiple entries coexist because the E88 has two cameras and the RTSP stream
    resolution is whatever the firmware negotiates."""
    p = Path(default_intrinsics_path() if path is None else path)
    payload: Dict[str, Any] = {"version": 1, "cameras": {}}
    if p.exists():
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and int(raw.get("version", 0)) == 1:
                cams = raw.get("cameras")
                if isinstance(cams, dict):
                    payload["cameras"] = cams
        except Exception:
            pass

    payload["cameras"][_key(intr.camera_id, intr.image_w, intr.image_h)] = asdict(intr)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return p


def load_intrinsics(
    *, camera_id: str, image_w: int, image_h: int, path: Optional[Path] = None, allow_rescale: bool = True
) -> Optional[CameraIntrinsics]:
    """Load intrinsics for a camera/resolution.

    With allow_rescale, an entry measured at a different resolution of the same
    camera is rescaled rather than rejected."""
    p = Path(default_intrinsics_path() if path is None else path)
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or int(raw.get("version", 0)) != 1:
            return None
        cams = raw.get("cameras")
        if not isinstance(cams, dict):
            return None

        exact = cams.get(_key(camera_id, image_w, image_h))
        if isinstance(exact, dict):
            return _from_dict(exact)

        if not allow_rescale:
            return None

        # Rescaling assumes the sensor is the same and only the sampling changed.
        # That holds for a plain resize but not across a different aspect ratio,
        # which implies a different crop or a different optical path -- rescaling
        # those would silently produce a wrong focal length and principal point.
        # Pick the closest *compatible* entry rather than whichever came first.
        want_aspect = float(image_w) / float(image_h)
        best: Optional[CameraIntrinsics] = None
        best_delta: Optional[float] = None
        for v in cams.values():
            if not isinstance(v, dict) or str(v.get("camera_id")) != str(camera_id):
                continue
            try:
                cand = _from_dict(v)
            except Exception:
                continue
            aspect = float(cand.image_w) / float(cand.image_h)
            if abs(aspect - want_aspect) > 0.02:
                continue
            delta = abs(float(cand.image_w) - float(image_w))
            if best_delta is None or delta < best_delta:
                best, best_delta = cand, delta

        if best is None:
            return None
        return best.scaled_to(image_w=int(image_w), image_h=int(image_h))
    except Exception:
        return None


def _key(camera_id: str, w: int, h: int) -> str:
    return f"{camera_id}@{int(w)}x{int(h)}"


def _from_dict(d: Dict[str, Any]) -> CameraIntrinsics:
    return CameraIntrinsics(
        camera_id=str(d["camera_id"]),
        image_w=int(d["image_w"]),
        image_h=int(d["image_h"]),
        fx=float(d["fx"]),
        fy=float(d["fy"]),
        cx=float(d["cx"]),
        cy=float(d["cy"]),
        k1=float(d["k1"]),
        k2=float(d["k2"]),
        p1=float(d["p1"]),
        p2=float(d["p2"]),
        k3=float(d["k3"]),
        rms_reproj_error_px=float(d["rms_reproj_error_px"]),
        n_views=int(d["n_views"]),
        pattern_cols=int(d["pattern_cols"]),
        pattern_rows=int(d["pattern_rows"]),
        square_size_m=float(d["square_size_m"]),
        created_at_ts=float(d["created_at_ts"]),
    )


def resolve_camera_geometry(
    *,
    camera_id: str,
    image_w: int,
    image_h: int,
    focal_length_px_override: Optional[float] = None,
    path: Optional[Path] = None,
) -> Tuple[float, Optional[Tuple[float, float]], str]:
    """Focal length and principal point for the flow estimators, in full-resolution
    pixels, plus a human-readable note about where they came from.

    An explicit override wins (so a CLI flag still works), then a saved chessboard
    calibration for this camera and resolution. Returns f = 0.0 when neither exists,
    which the estimators reject for the derotation model rather than silently
    substituting a different one."""
    if focal_length_px_override is not None and float(focal_length_px_override) > 0.0:
        return float(focal_length_px_override), None, "explicit --focal-length-px (no principal point)"

    # load_intrinsics returns an already-rescaled record, so ask for an exact match
    # first to know whether a rescale happened and can be reported.
    exact = load_intrinsics(
        camera_id=str(camera_id), image_w=int(image_w), image_h=int(image_h), path=path, allow_rescale=False
    )
    intr = exact or load_intrinsics(
        camera_id=str(camera_id), image_w=int(image_w), image_h=int(image_h), path=path
    )
    if intr is None:
        return 0.0, None, f"no calibration for {camera_id}@{int(image_w)}x{int(image_h)}"

    note = (
        f"calibrated {intr.camera_id}@{intr.image_w}x{intr.image_h} "
        f"(f={intr.focal_length_px:.1f}, c=({intr.cx:.1f},{intr.cy:.1f}), "
        f"k1={intr.k1:.4f}, rms={intr.rms_reproj_error_px:.3f}px)"
    )
    if exact is None:
        note += " [rescaled from another resolution]"
    return float(intr.focal_length_px), (float(intr.cx), float(intr.cy)), note


def find_chessboard_corners(
    frame_bgr: np.ndarray, *, pattern_size: Tuple[int, int]
) -> Optional[np.ndarray]:
    """Locate inner chessboard corners, refined to sub-pixel accuracy.

    pattern_size is (cols, rows) of *inner* corners -- a standard 10x7-square board
    has 9x6 inner corners. Returns (N, 1, 2) float32 or None."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    ok, corners = cv2.findChessboardCorners(gray, tuple(pattern_size), flags)
    if not ok or corners is None:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)


def object_points(*, pattern_size: Tuple[int, int], square_size_m: float) -> np.ndarray:
    cols, rows = int(pattern_size[0]), int(pattern_size[1])
    objp = np.zeros((rows * cols, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return objp * float(square_size_m)


def calibrate_from_corners(
    corner_sets: Sequence[np.ndarray],
    *,
    image_size: Tuple[int, int],
    pattern_size: Tuple[int, int],
    square_size_m: float,
    camera_id: str,
    min_views: int = 6,
) -> CameraIntrinsics:
    """Run cv2.calibrateCamera over accumulated chessboard views.

    image_size is (width, height)."""
    if len(corner_sets) < int(min_views):
        raise ValueError(f"need at least {int(min_views)} views, got {len(corner_sets)}")

    objp = object_points(pattern_size=pattern_size, square_size_m=square_size_m)
    obj_pts = [objp for _ in corner_sets]
    img_pts = [np.asarray(c, dtype=np.float32).reshape(-1, 1, 2) for c in corner_sets]

    rms, mtx, dist, _rvecs, _tvecs = cv2.calibrateCamera(
        obj_pts, img_pts, (int(image_size[0]), int(image_size[1])), None, None
    )

    d = np.asarray(dist, dtype=np.float64).reshape(-1)
    d = np.pad(d, (0, max(0, 5 - d.size)))[:5]

    return CameraIntrinsics(
        camera_id=str(camera_id),
        image_w=int(image_size[0]),
        image_h=int(image_size[1]),
        fx=float(mtx[0, 0]),
        fy=float(mtx[1, 1]),
        cx=float(mtx[0, 2]),
        cy=float(mtx[1, 2]),
        k1=float(d[0]),
        k2=float(d[1]),
        p1=float(d[2]),
        p2=float(d[3]),
        k3=float(d[4]),
        rms_reproj_error_px=float(rms),
        n_views=int(len(corner_sets)),
        pattern_cols=int(pattern_size[0]),
        pattern_rows=int(pattern_size[1]),
        square_size_m=float(square_size_m),
        created_at_ts=float(time.time()),
    )


def outer_corners(corners: np.ndarray, *, pattern_size: Tuple[int, int]) -> np.ndarray:
    """The four grid corners of a detected chessboard, in traversal order.

    findChessboardCorners returns inner corners row-major over (cols, rows)."""
    cols, rows = int(pattern_size[0]), int(pattern_size[1])
    pts = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    return np.stack([pts[0], pts[cols - 1], pts[cols * rows - 1], pts[cols * (rows - 1)]])


def pose_descriptor(
    corners: np.ndarray, *, pattern_size: Tuple[int, int], image_size: Tuple[int, int]
) -> np.ndarray:
    """A scale-free description of where and how the board sits in the frame.

    Planar calibration is constrained by pose *variety*, not view count. The
    informative axes are:
      - position     (translation across the frame)
      - scale        (distance from the camera)
      - rotation     (in-plane roll of the board)
      - foreshortening (out-of-plane tilt -- the only thing that separates focal
                        length from distance, and what pins down the distortion
                        coefficients)
    Foreshortening is weighted highest because it is both the most valuable and the
    axis an operator is least likely to vary on their own.
    """
    quad = outer_corners(corners, pattern_size=pattern_size)
    diag = float(np.hypot(float(image_size[0]), float(image_size[1])))
    if diag <= 1e-9:
        diag = 1.0

    centroid = quad.mean(axis=0)
    area = 0.5 * abs(
        float(np.dot(quad[:, 0], np.roll(quad[:, 1], -1)) - np.dot(quad[:, 1], np.roll(quad[:, 0], -1)))
    )
    scale = float(np.sqrt(max(area, 1e-9)))

    top = quad[1] - quad[0]
    bottom = quad[2] - quad[3]
    left = quad[3] - quad[0]
    right = quad[2] - quad[1]

    theta = float(np.arctan2(top[1], top[0]))

    def _ratio(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na <= 1e-9 or nb <= 1e-9:
            return 0.0
        return float(np.log(na / nb))

    w_pos, w_scale, w_rot, w_fore = 1.0, 1.0, 0.5, 2.0
    return np.array(
        [
            centroid[0] / diag * w_pos,
            centroid[1] / diag * w_pos,
            float(np.log(scale / diag)) * w_scale,
            float(np.cos(theta)) * w_rot,
            float(np.sin(theta)) * w_rot,
            _ratio(top, bottom) * w_fore,
            _ratio(left, right) * w_fore,
        ],
        dtype=np.float64,
    )


class IntrinsicsCollector:
    """Accumulates chessboard views, rejecting ones that add no new information.

    Calibration quality depends on pose diversity, not view count: twenty views of
    the board in the same place constrain the model no better than one. Rejection is
    scored on a pose descriptor (position, scale, in-plane rotation, out-of-plane
    foreshortening) rather than centroid distance alone -- centroid distance both
    discards genuinely useful views (tilting or resizing the board in place) and
    accepts degenerate ones (the same flat view slid 40 px sideways)."""

    def __init__(
        self,
        *,
        pattern_size: Tuple[int, int] = (9, 6),
        square_size_m: float = 0.025,
        # Calibrated against rendered views at 640x480: a deliberately diverse set of
        # 10 poses has a nearest-neighbour distance of ~0.118, while an 11-degree tilt
        # in place scores ~0.122 and an 8 cm slide of a flat board ~0.124. 0.10 admits
        # all of those and rejects the marginal ones (a 4 cm slide, a 6-degree tilt).
        # <= 0 disables the check.
        min_pose_distance: float = 0.10,
    ) -> None:
        self._pattern_size = (int(pattern_size[0]), int(pattern_size[1]))
        self._square_size_m = float(square_size_m)
        self._min_pose_distance = float(min_pose_distance)

        self._corner_sets: List[np.ndarray] = []
        self._descriptors: List[np.ndarray] = []
        self._image_size: Optional[Tuple[int, int]] = None

    @property
    def pattern_size(self) -> Tuple[int, int]:
        return self._pattern_size

    @property
    def n_views(self) -> int:
        return len(self._corner_sets)

    @property
    def image_size(self) -> Optional[Tuple[int, int]]:
        return self._image_size

    @property
    def corner_sets(self) -> List[np.ndarray]:
        return list(self._corner_sets)

    def try_add(self, frame_bgr: np.ndarray) -> Tuple[bool, str]:
        """Returns (accepted, reason)."""
        h, w = frame_bgr.shape[:2]
        size = (int(w), int(h))
        if self._image_size is not None and size != self._image_size:
            return False, f"resolution changed {self._image_size} -> {size}"

        corners = find_chessboard_corners(frame_bgr, pattern_size=self._pattern_size)
        if corners is None:
            return False, "chessboard not found"

        desc = pose_descriptor(corners, pattern_size=self._pattern_size, image_size=size)
        if self._min_pose_distance > 0.0 and self._descriptors:
            nearest = min(float(np.linalg.norm(desc - prev)) for prev in self._descriptors)
            if nearest < self._min_pose_distance:
                return False, (
                    "too similar to an accepted view - move the board, or change its "
                    "tilt or distance"
                )

        self._image_size = size
        self._corner_sets.append(np.asarray(corners, dtype=np.float32))
        self._descriptors.append(desc)
        return True, f"accepted view {len(self._corner_sets)}"

    def coverage_report(self) -> Dict[str, Any]:
        """Spread of the accepted views along each descriptor axis.

        A set that varies only in position will show near-zero tilt spread, which is
        the case where calibration silently produces a poorly constrained focal
        length and distortion."""
        if not self._descriptors:
            return {"n_views": 0, "position_spread": 0.0, "scale_spread": 0.0, "tilt_spread": 0.0}
        d = np.stack(self._descriptors)
        return {
            "n_views": int(d.shape[0]),
            "position_spread": float(np.linalg.norm(np.std(d[:, 0:2], axis=0))),
            "scale_spread": float(np.std(d[:, 2])),
            "tilt_spread": float(np.linalg.norm(np.std(d[:, 5:7], axis=0))),
        }

    def calibrate(self, *, camera_id: str, min_views: int = 6) -> CameraIntrinsics:
        if self._image_size is None:
            raise ValueError("no views collected")
        return calibrate_from_corners(
            self._corner_sets,
            image_size=self._image_size,
            pattern_size=self._pattern_size,
            square_size_m=self._square_size_m,
            camera_id=str(camera_id),
            min_views=int(min_views),
        )
