"""Downward camera model: renders the ground plane from a simulated pose.

Because the scene is a plane, the projection is a homography and the render is
exact -- no ray tracing, no approximation of perspective. That matters: the
whole point of feeding synthetic frames to the *real* optical-flow stack is that
the geometry it recovers should be the geometry we put in. A test in
``tests/test_sim_camera.py`` closes that loop by moving the camera a known
distance and checking the real ``LucasKanadeDriftEstimator`` reads back
``f * v / z`` pixels per second.

Two orientation details are easy to get wrong and both are modelled explicitly:

- The camera looks *down*, so its image y axis points backward in the world.
  Positive forward motion moves ground features *down* the image.
- The camera is mounted rotated on the airframe. The autopilot compensates with
  a fixed 90 deg remap (roll driven from vy, pitch from vx -- README 4.4.2), so
  the sim carries the same 90 deg in ``CameraParams.mount_yaw_deg``. Rendering
  it here rather than assuming it away is what lets the simulator catch an
  axis-swap mistake instead of hiding one.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from e88_sim.config import CameraParams
from e88_sim.ground import GroundTexture
from e88_sim.physics import PhysicsState, body_to_world_rotation


# Camera axes expressed in body coordinates, as columns, before the mount
# rotation: image-right = body-right, image-down = body-backward, optical
# axis = body-down.
_CAM_AXES_IN_BODY = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)


@dataclass(frozen=True)
class RenderResult:
    frame_bgr: np.ndarray
    # True if part of the footprint fell outside the ground texture, i.e. the
    # drone drifted beyond the simulated world and the edges are fabricated.
    out_of_bounds: bool
    # Ground sampling distance at the principal point, metres per image pixel.
    gsd_m_per_px: float


@dataclass(frozen=True)
class _Pose:
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float


def pose_from_state(state: PhysicsState) -> _Pose:
    return _Pose(
        x=float(state.x_m),
        y=float(state.y_m),
        z=float(state.z_m),
        roll=float(state.roll_rad),
        pitch=float(state.pitch_rad),
        yaw=float(state.yaw_rad),
    )


def pose_at_offset(state: PhysicsState, dt: float) -> _Pose:
    """Linear extrapolation of the pose by ``dt``, used for exposure sampling."""
    d = float(dt)
    return _Pose(
        x=float(state.x_m) + float(state.vx_m_s) * d,
        y=float(state.y_m) + float(state.vy_m_s) * d,
        z=max(0.0, float(state.z_m) + float(state.vz_m_s) * d),
        roll=float(state.roll_rad) + float(state.roll_rate_rad_s) * d,
        pitch=float(state.pitch_rad) + float(state.pitch_rate_rad_s) * d,
        yaw=float(state.yaw_rad) + float(state.yaw_rate_rad_s) * d,
    )


class DownwardCamera:
    def __init__(self, params: Optional[CameraParams] = None, *, ground: Optional[GroundTexture] = None) -> None:
        from e88_sim.ground import make_ground

        self._p = params or CameraParams()
        self._ground = ground if ground is not None else make_ground()

        self._cx = float(self._p.cx) if self._p.cx is not None else (float(self._p.width) - 1.0) * 0.5
        self._cy = float(self._p.cy) if self._p.cy is not None else (float(self._p.height) - 1.0) * 0.5
        self._k = np.array(
            [[float(self._p.fx), 0.0, self._cx], [0.0, float(self._p.fy), self._cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

        psi_c = math.radians(float(self._p.mount_yaw_deg))
        c, s = math.cos(psi_c), math.sin(psi_c)
        rot_about_optical_axis = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        self._body_to_cam_axes = _CAM_AXES_IN_BODY @ rot_about_optical_axis

        # Texture pixel -> world plane coordinates, as a 3x3 acting on (u, v, 1).
        mpp = float(self._ground.m_per_px)
        u0, v0 = self._ground.center_px
        self._tex_to_world = np.array(
            [[mpp, 0.0, -u0 * mpp], [0.0, -mpp, v0 * mpp], [0.0, 0.0, 1.0]], dtype=np.float64
        )

        self._vignette_mask = self._build_vignette()
        self._distortion_maps = self._build_distortion_maps()
        self._rng = np.random.default_rng(0)
        self._probe_pts: Optional[np.ndarray] = None

    # ------------------------------------------------------------- properties

    @property
    def intrinsics(self) -> np.ndarray:
        return self._k.copy()

    @property
    def principal_point(self) -> Tuple[float, float]:
        return (float(self._cx), float(self._cy))

    @property
    def focal_length_px(self) -> float:
        return float(0.5 * (float(self._p.fx) + float(self._p.fy)))

    @property
    def ground(self) -> GroundTexture:
        return self._ground

    def seed(self, seed: int) -> None:
        self._rng = np.random.default_rng(int(seed))

    # ----------------------------------------------------------------- render

    def world_to_image_homography(self, pose: _Pose, *, plane_height_m: float = 0.0) -> np.ndarray:
        """H mapping points on the plane z = ``plane_height_m`` to image pixels.

        Passing a nonzero height is how the raised-object layer is rendered: the
        objects' top surfaces form a second plane, and projecting the same
        texture through a different plane is exactly what produces parallax.
        """
        r_wb = body_to_world_rotation(pose.roll, pose.pitch, pose.yaw)
        cam_axes_in_world = r_wb @ self._body_to_cam_axes
        r_cw = cam_axes_in_world.T

        c = np.array([pose.x, pose.y, pose.z], dtype=np.float64)
        origin_on_plane = np.array([0.0, 0.0, float(plane_height_m)], dtype=np.float64)
        t = r_cw @ (origin_on_plane - c)

        m = np.column_stack((r_cw[:, 0], r_cw[:, 1], t))
        return self._k @ m

    def texture_to_image_homography(self, pose: _Pose, *, plane_height_m: float = 0.0) -> np.ndarray:
        return self.world_to_image_homography(pose, plane_height_m=plane_height_m) @ self._tex_to_world

    def render(self, state: PhysicsState) -> RenderResult:
        p = self._p
        w, h = int(p.width), int(p.height)

        samples = max(1, int(p.blur_samples))
        exposure = max(0.0, float(p.exposure_sec))
        if samples == 1 or exposure <= 0.0:
            offsets = [0.0]
        else:
            # Sample the exposure window symmetrically around the frame time.
            offsets = [exposure * (i / (samples - 1) - 0.5) for i in range(samples)]

        accum = np.zeros((h, w), dtype=np.float32)
        out_of_bounds = False
        for off in offsets:
            pose = pose_at_offset(state, off)
            layer, oob = self._render_layers(pose, w, h)
            accum += layer
            out_of_bounds = out_of_bounds or oob

        img = accum / float(len(offsets))

        if float(p.brightness_gain) != 1.0:
            img *= float(p.brightness_gain)
        if self._vignette_mask is not None:
            img *= self._vignette_mask
        if float(p.sensor_noise_sigma) > 0.0:
            img += self._rng.normal(0.0, float(p.sensor_noise_sigma), size=img.shape).astype(np.float32)

        gray = np.clip(img, 0.0, 255.0).astype(np.uint8)

        if self._distortion_maps is not None:
            map_x, map_y = self._distortion_maps
            gray = cv2.remap(gray, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        if p.jpeg_quality is not None:
            ok, buf = cv2.imencode(".jpg", gray, [int(cv2.IMWRITE_JPEG_QUALITY), int(p.jpeg_quality)])
            if ok:
                gray = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)

        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        gsd = float(state.z_m) / max(1e-6, self.focal_length_px)
        return RenderResult(frame_bgr=frame, out_of_bounds=bool(out_of_bounds), gsd_m_per_px=gsd)

    def _render_layers(self, pose: _Pose, w: int, h: int) -> Tuple[np.ndarray, bool]:
        """Warp the floor, then the raised-object layer, and composite.

        Both layers use the same texture but different plane heights, so a
        raised object's image motion differs from the floor's by exactly the
        parallax its height implies. That difference is what the RANSAC fit in
        the flow estimator has to reject.
        """
        hom_floor = self.texture_to_image_homography(pose)
        floor = cv2.warpPerspective(
            self._ground.gray, hom_floor, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
        ).astype(np.float32)
        oob = self._footprint_out_of_bounds(hom_floor, w, h)

        if not self._ground.has_relief:
            return floor, oob

        height = float(self._ground.relief_height_m)
        if pose.z - height < 0.05:
            # The camera is at or below the top of the objects; parallax is not
            # meaningful and the homography degenerates. Fall back to the floor.
            return floor, oob

        hom_relief = self.texture_to_image_homography(pose, plane_height_m=height)
        raised = cv2.warpPerspective(
            self._ground.gray, hom_relief, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
        ).astype(np.float32)
        mask = cv2.warpPerspective(
            self._ground.relief_mask, hom_relief, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
        )
        return floor * (1.0 - mask) + raised * mask, oob

    # ------------------------------------------------------------- predictions

    def ground_truth_flow_px(self, prev_state: PhysicsState, state: PhysicsState) -> Tuple[float, float]:
        """Exact image displacement of the floor between two poses, in pixels.

        This is the quantity the flow estimator is trying to measure, defined
        the same way the estimator defines it: the best global translation of
        the *whole frame*, not the motion of one point. It is computed by
        transporting a grid of image points through the plane-induced homography
        (image -> floor -> image) and averaging, so tilt-induced apparent flow,
        vibration and altitude change are all included exactly, with no
        small-angle approximation.

        Scoring the real estimator against this is the core fidelity check: if
        they agree, the renderer's geometry and the estimator's model agree.
        """
        h_prev = self.world_to_image_homography(pose_from_state(prev_state))
        h_cur = self.world_to_image_homography(pose_from_state(state))
        try:
            transport = h_cur @ np.linalg.inv(h_prev)
        except np.linalg.LinAlgError:
            return (0.0, 0.0)

        pts = self._flow_probe_points()
        moved = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), transport).reshape(-1, 2)
        delta = moved - pts
        finite = np.all(np.isfinite(delta), axis=1)
        if not np.any(finite):
            return (0.0, 0.0)
        mean = delta[finite].mean(axis=0)
        return (float(mean[0]), float(mean[1]))

    def ground_truth_translation_flow_px(self, prev_state: PhysicsState, state: PhysicsState) -> Tuple[float, float]:
        """Apparent flow from *translation alone*, holding attitude fixed.

        The motion models disagree about what they are estimating, so they need
        different references:

        - ``affine_translation`` and ``translation_rotation`` report total
          apparent flow, tilt included -> score against
          :meth:`ground_truth_flow_px`.
        - ``derotation`` deliberately subtracts the rotational component and
          reports translation only -> score against this.

        Comparing a derotating estimator to total flow would make a correctly
        working estimator look broken, and vice versa.
        """
        frozen_attitude = dataclasses.replace(
            prev_state, x_m=float(state.x_m), y_m=float(state.y_m), z_m=float(state.z_m)
        )
        return self.ground_truth_flow_px(prev_state, frozen_attitude)

    def _flow_probe_points(self) -> np.ndarray:
        if getattr(self, "_probe_pts", None) is None:
            w, h = int(self._p.width), int(self._p.height)
            xs = np.linspace(0.1 * w, 0.9 * w, 9)
            ys = np.linspace(0.1 * h, 0.9 * h, 9)
            gx, gy = np.meshgrid(xs, ys)
            self._probe_pts = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64)
        return self._probe_pts

    def expected_flow_px_s(self, state: PhysicsState) -> Tuple[float, float]:
        """Ground-truth image-plane velocity at the principal point, px/s.

        This is what a perfect optical-flow estimator would report for this
        state: the apparent motion of ground features, which is the *negative*
        of the camera's own motion projected into the image, plus the apparent
        motion induced by tilting.

        Used two ways: as the analytic sensor in ``fast_loop``, and as the
        reference the vision-in-the-loop tests score the real estimator against.
        """
        z = float(state.z_m)
        if z <= 1e-4:
            return (0.0, 0.0)

        f = self.focal_length_px

        r_wb = body_to_world_rotation(state.roll_rad, state.pitch_rad, state.yaw_rad)
        cam_axes_in_world = r_wb @ self._body_to_cam_axes
        r_cw = cam_axes_in_world.T

        # Camera-frame translational velocity. A feature at the principal point
        # sits at depth z / cos(tilt); using the optical-axis depth is exact at
        # the principal point for a level camera and accurate to O(tilt^2)
        # otherwise, which at hover tilts (<5 deg) is under half a percent.
        v_world = np.array([float(state.vx_m_s), float(state.vy_m_s), float(state.vz_m_s)], dtype=np.float64)
        v_cam = r_cw @ v_world
        depth = z / max(0.2, float(abs(cam_axes_in_world[2, 2])))

        flow_x = -f * float(v_cam[0]) / depth
        flow_y = -f * float(v_cam[1]) / depth

        # Rotation: an angular rate omega about the camera axes moves a feature
        # at the principal point by -f * omega_y (in x) and +f * omega_x (in y).
        omega_world = self._body_rates_to_world(state)
        omega_cam = r_cw @ omega_world
        flow_x += -f * float(omega_cam[1])
        flow_y += f * float(omega_cam[0])

        return (float(flow_x), float(flow_y))

    @staticmethod
    def _body_rates_to_world(state: PhysicsState) -> np.ndarray:
        """Angular velocity in world axes.

        At hover the tilt angles are small, so the Euler rates map to world axes
        essentially one-for-one: roll rate about world x... except that roll is
        defined about the body *forward* axis and pitch about the body *right*
        axis (see physics.py), so roll rate is about world y and pitch rate is
        about world x, with the pitch sign flipped by the same convention.
        """
        yaw = float(state.yaw_rad)
        c, s = math.cos(yaw), math.sin(yaw)
        right = np.array([c, s, 0.0], dtype=np.float64)
        forward = np.array([-s, c, 0.0], dtype=np.float64)
        omega = forward * float(state.roll_rate_rad_s) + right * (-float(state.pitch_rate_rad_s))
        omega = omega + np.array([0.0, 0.0, float(state.yaw_rate_rad_s)], dtype=np.float64)
        return omega

    # --------------------------------------------------------------- internals

    def _footprint_out_of_bounds(self, tex_to_img: np.ndarray, w: int, h: int) -> bool:
        try:
            inv = np.linalg.inv(tex_to_img)
        except np.linalg.LinAlgError:
            return True
        corners = np.array([[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]], dtype=np.float64)
        pts = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), inv).reshape(-1, 2)
        gh, gw = self._ground.gray.shape[:2]
        return bool(
            np.any(pts[:, 0] < 0.0) or np.any(pts[:, 0] > gw - 1.0) or np.any(pts[:, 1] < 0.0) or np.any(pts[:, 1] > gh - 1.0)
        )

    def _build_vignette(self) -> Optional[np.ndarray]:
        strength = float(self._p.vignette)
        if strength <= 0.0:
            return None
        w, h = int(self._p.width), int(self._p.height)
        xs = (np.arange(w, dtype=np.float32) - self._cx) / max(1.0, float(w) * 0.5)
        ys = (np.arange(h, dtype=np.float32) - self._cy) / max(1.0, float(h) * 0.5)
        r2 = (ys[:, None] ** 2) + (xs[None, :] ** 2)
        return (1.0 - strength * np.clip(r2 / 2.0, 0.0, 1.0)).astype(np.float32)

    def _build_distortion_maps(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        p = self._p
        coeffs = np.array([p.k1, p.k2, p.p1, p.p2, p.k3], dtype=np.float64)
        if not np.any(np.abs(coeffs) > 0.0):
            return None

        w, h = int(p.width), int(p.height)
        # For each pixel of the *distorted* output, find where it came from in the
        # ideal pinhole render. undistortPoints does exactly that mapping.
        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        pts = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).reshape(-1, 1, 2)
        ideal = cv2.undistortPoints(pts, self._k, coeffs, P=self._k).reshape(h, w, 2)
        return (np.ascontiguousarray(ideal[:, :, 0]), np.ascontiguousarray(ideal[:, :, 1]))
