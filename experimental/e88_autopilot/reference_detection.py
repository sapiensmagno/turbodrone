from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class ReferenceDetectionResult:
    detected: bool
    mode: str
    quad_xy: Optional[np.ndarray]  # shape (4, 2) float32
    ref_width_px: float
    ref_height_px: float
    ref_size_px: float
    n_kp_frame: int
    n_matches: int
    n_inliers: int
    inlier_ratio: float
    reproj_error_px: float


class ReferenceDetector:
    def __init__(
        self,
        *,
        reference_bgr: np.ndarray,
        use_markers: bool,
        min_matches: int = 30,
        min_inliers: int = 20,
        min_inlier_ratio: float = 0.4,
        contour_min_arc_len_px: float = 50.0,
        contour_canny1: int = 100,
        contour_canny2: int = 400,
        contour_dilate_ksize: int = 5,
        contour_dilate_iter: int = 1,
    ) -> None:
        self._use_markers = bool(use_markers)
        self._min_matches = int(min_matches)
        self._min_inliers = int(min_inliers)
        self._min_inlier_ratio = float(min_inlier_ratio)

        self._contour_min_arc_len_px = float(contour_min_arc_len_px)
        self._contour_canny1 = int(contour_canny1)
        self._contour_canny2 = int(contour_canny2)
        self._contour_dilate_ksize = int(contour_dilate_ksize)
        self._contour_dilate_iter = int(contour_dilate_iter)

        self._ref_gray = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
        self._ref_h, self._ref_w = self._ref_gray.shape[:2]

        self._orb = cv2.ORB_create(nfeatures=1200)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self._ref_kp, self._ref_desc = self._orb.detectAndCompute(self._ref_gray, None)

    @property
    def reference_size_px(self) -> float:
        return float(0.5 * (float(self._ref_w) + float(self._ref_h)))

    def detect(self, frame_bgr: np.ndarray) -> ReferenceDetectionResult:
        if self._use_markers:
            out = self._detect_markers(frame_bgr)
            if out is not None:
                return out
            return ReferenceDetectionResult(
                detected=False,
                mode="markers",
                quad_xy=None,
                ref_width_px=0.0,
                ref_height_px=0.0,
                ref_size_px=0.0,
                n_kp_frame=0,
                n_matches=0,
                n_inliers=0,
                inlier_ratio=0.0,
                reproj_error_px=0.0,
            )

        out = self._detect_orb_homography(frame_bgr)
        if bool(out.detected):
            return out

        fb = self._detect_contour_min_area_rect(frame_bgr)
        if fb is not None:
            return fb
        return out

    def _detect_contour_min_area_rect(self, frame_bgr: np.ndarray) -> Optional[ReferenceDetectionResult]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        edged = cv2.Canny(gray, self._contour_canny1, self._contour_canny2)
        if self._contour_dilate_ksize > 0 and self._contour_dilate_iter > 0:
            k = int(self._contour_dilate_ksize)
            kernel = np.ones((k, k), np.uint8)
            edged = cv2.dilate(edged, kernel, iterations=int(self._contour_dilate_iter))

        contours, _hier = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        min_len = float(self._contour_min_arc_len_px)
        significant = [c for c in contours if float(cv2.arcLength(c, True)) >= min_len]
        if not significant:
            return None

        try:
            all_points = np.vstack(significant)
        except Exception:
            return None

        rect = cv2.minAreaRect(all_points)
        (x, y), (w, h), angle = rect
        w = float(w)
        h = float(h)
        if w <= 1e-6 or h <= 1e-6:
            return None

        side = float(min(w, h))
        sq_rect = ((float(x), float(y)), (float(side), float(side)), float(angle))
        box = cv2.boxPoints(sq_rect).astype(np.float32)

        if not _quad_is_valid(box, frame_shape=gray.shape[:2]):
            return None

        ref_w_px, ref_h_px, ref_size_px = _quad_sizes(box)
        return ReferenceDetectionResult(
            detected=True,
            mode="contours",
            quad_xy=box,
            ref_width_px=float(ref_w_px),
            ref_height_px=float(ref_h_px),
            ref_size_px=float(ref_size_px),
            n_kp_frame=0,
            n_matches=0,
            n_inliers=0,
            inlier_ratio=0.0,
            reproj_error_px=0.0,
        )

    def _detect_orb_homography(self, frame_bgr: np.ndarray) -> ReferenceDetectionResult:
        if self._ref_desc is None or len(self._ref_desc) == 0:
            return ReferenceDetectionResult(
                detected=False,
                mode="orb",
                quad_xy=None,
                ref_width_px=0.0,
                ref_height_px=0.0,
                ref_size_px=0.0,
                n_kp_frame=0,
                n_matches=0,
                n_inliers=0,
                inlier_ratio=0.0,
                reproj_error_px=0.0,
            )

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kp, desc = self._orb.detectAndCompute(gray, None)
        if desc is None or len(desc) == 0:
            return ReferenceDetectionResult(
                detected=False,
                mode="orb",
                quad_xy=None,
                ref_width_px=0.0,
                ref_height_px=0.0,
                ref_size_px=0.0,
                n_kp_frame=int(len(kp)),
                n_matches=0,
                n_inliers=0,
                inlier_ratio=0.0,
                reproj_error_px=0.0,
            )

        matches_knn = self._bf.knnMatch(self._ref_desc, desc, k=2)
        good = []
        for m_n in matches_knn:
            if len(m_n) != 2:
                continue
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) < max(4, self._min_matches):
            return ReferenceDetectionResult(
                detected=False,
                mode="orb",
                quad_xy=None,
                ref_width_px=0.0,
                ref_height_px=0.0,
                ref_size_px=0.0,
                n_kp_frame=int(len(kp)),
                n_matches=int(len(good)),
                n_inliers=0,
                inlier_ratio=0.0,
                reproj_error_px=0.0,
            )

        src = np.float32([self._ref_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        h, inliers = cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=5.0)
        if h is None or inliers is None:
            return ReferenceDetectionResult(
                detected=False,
                mode="orb",
                quad_xy=None,
                ref_width_px=0.0,
                ref_height_px=0.0,
                ref_size_px=0.0,
                n_kp_frame=int(len(kp)),
                n_matches=int(len(good)),
                n_inliers=0,
                inlier_ratio=0.0,
                reproj_error_px=0.0,
            )

        inliers = inliers.reshape(-1).astype(bool)
        n_inl = int(np.count_nonzero(inliers))
        inl_ratio = float(n_inl) / float(max(1, len(good)))

        if n_inl < self._min_inliers or inl_ratio < self._min_inlier_ratio:
            return ReferenceDetectionResult(
                detected=False,
                mode="orb",
                quad_xy=None,
                ref_width_px=0.0,
                ref_height_px=0.0,
                ref_size_px=0.0,
                n_kp_frame=int(len(kp)),
                n_matches=int(len(good)),
                n_inliers=int(n_inl),
                inlier_ratio=float(inl_ratio),
                reproj_error_px=0.0,
            )

        corners = np.float32([[0, 0], [self._ref_w - 1, 0], [self._ref_w - 1, self._ref_h - 1], [0, self._ref_h - 1]]).reshape(
            -1, 1, 2
        )
        proj = cv2.perspectiveTransform(corners, h).reshape(4, 2).astype(np.float32)

        if not _quad_is_valid(proj, frame_shape=gray.shape[:2]):
            return ReferenceDetectionResult(
                detected=False,
                mode="orb",
                quad_xy=None,
                ref_width_px=0.0,
                ref_height_px=0.0,
                ref_size_px=0.0,
                n_kp_frame=int(len(kp)),
                n_matches=int(len(good)),
                n_inliers=int(n_inl),
                inlier_ratio=float(inl_ratio),
                reproj_error_px=0.0,
            )

        ref_w_px, ref_h_px, ref_size_px = _quad_sizes(proj)

        reproj_err = _mean_reprojection_error_px(h, src[inliers], dst[inliers])

        return ReferenceDetectionResult(
            detected=True,
            mode="orb",
            quad_xy=proj,
            ref_width_px=float(ref_w_px),
            ref_height_px=float(ref_h_px),
            ref_size_px=float(ref_size_px),
            n_kp_frame=int(len(kp)),
            n_matches=int(len(good)),
            n_inliers=int(n_inl),
            inlier_ratio=float(inl_ratio),
            reproj_error_px=float(reproj_err),
        )

    def _detect_markers(self, frame_bgr: np.ndarray) -> Optional[ReferenceDetectionResult]:
        aruco = getattr(cv2, "aruco", None)
        if aruco is None:
            return None

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        d = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        corners, ids, _rej = aruco.detectMarkers(gray, d)
        if ids is None or len(ids) < 4:
            return ReferenceDetectionResult(
                detected=False,
                mode="markers",
                quad_xy=None,
                ref_width_px=0.0,
                ref_height_px=0.0,
                ref_size_px=0.0,
                n_kp_frame=0,
                n_matches=0,
                n_inliers=int(0 if ids is None else len(ids)),
                inlier_ratio=0.0,
                reproj_error_px=0.0,
            )

        pts = np.concatenate([c.reshape(-1, 2) for c in corners], axis=0)
        hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2)
        if hull.shape[0] < 4:
            return None

        rect = cv2.minAreaRect(hull.astype(np.float32))
        box = cv2.boxPoints(rect).astype(np.float32)

        if not _quad_is_valid(box, frame_shape=gray.shape[:2]):
            return ReferenceDetectionResult(
                detected=False,
                mode="markers",
                quad_xy=None,
                ref_width_px=0.0,
                ref_height_px=0.0,
                ref_size_px=0.0,
                n_kp_frame=0,
                n_matches=0,
                n_inliers=int(len(ids)),
                inlier_ratio=0.0,
                reproj_error_px=0.0,
            )

        ref_w_px, ref_h_px, ref_size_px = _quad_sizes(box)

        return ReferenceDetectionResult(
            detected=True,
            mode="markers",
            quad_xy=box,
            ref_width_px=float(ref_w_px),
            ref_height_px=float(ref_h_px),
            ref_size_px=float(ref_size_px),
            n_kp_frame=0,
            n_matches=0,
            n_inliers=int(len(ids)),
            inlier_ratio=1.0,
            reproj_error_px=0.0,
        )


def _quad_sizes(quad_xy: np.ndarray) -> Tuple[float, float, float]:
    q = np.asarray(quad_xy, dtype=np.float32).reshape(4, 2)
    w1 = float(np.linalg.norm(q[1] - q[0]))
    w2 = float(np.linalg.norm(q[2] - q[3]))
    h1 = float(np.linalg.norm(q[3] - q[0]))
    h2 = float(np.linalg.norm(q[2] - q[1]))
    w = 0.5 * (w1 + w2)
    h = 0.5 * (h1 + h2)
    return float(w), float(h), float(0.5 * (w + h))


def _quad_is_valid(quad_xy: np.ndarray, *, frame_shape: Tuple[int, int]) -> bool:
    q = np.asarray(quad_xy, dtype=np.float32).reshape(4, 2)

    area = float(abs(cv2.contourArea(q.reshape(-1, 1, 2))))
    if area < 100.0:
        return False

    hull = cv2.convexHull(q.reshape(-1, 1, 2)).reshape(-1, 2)
    if hull.shape[0] != 4:
        return False

    h, w = int(frame_shape[0]), int(frame_shape[1])
    if np.any(~np.isfinite(q)):
        return False

    if np.any(q[:, 0] < -10) or np.any(q[:, 1] < -10) or np.any(q[:, 0] > (w + 10)) or np.any(q[:, 1] > (h + 10)):
        return False

    return True


def _mean_reprojection_error_px(h: np.ndarray, src_xy: np.ndarray, dst_xy: np.ndarray) -> float:
    try:
        if src_xy is None or dst_xy is None:
            return 0.0
        if src_xy.shape[0] < 4:
            return 0.0
        proj = cv2.perspectiveTransform(np.asarray(src_xy, dtype=np.float32), np.asarray(h, dtype=np.float64))
        e = np.linalg.norm(proj.reshape(-1, 2) - np.asarray(dst_xy, dtype=np.float32).reshape(-1, 2), axis=1)
        return float(np.mean(e))
    except Exception:
        return 0.0
