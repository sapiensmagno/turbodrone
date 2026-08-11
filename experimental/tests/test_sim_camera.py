import math
import sys
import unittest
from pathlib import Path

import numpy as np

_EXPERIMENTAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXPERIMENTAL_ROOT))

from e88_sim.camera import DownwardCamera, pose_from_state
from e88_sim.config import CameraParams, GroundParams
from e88_sim.ground import make_ground
from e88_sim.physics import PhysicsState


def state(x=0.0, y=0.0, z=0.7, roll=0.0, pitch=0.0, yaw=0.0, **rates) -> PhysicsState:
    defaults = dict(
        vx_m_s=0.0,
        vy_m_s=0.0,
        vz_m_s=0.0,
        roll_rate_rad_s=0.0,
        pitch_rate_rad_s=0.0,
        yaw_rate_rad_s=0.0,
    )
    defaults.update(rates)
    return PhysicsState(
        t=0.0,
        x_m=x,
        y_m=y,
        z_m=z,
        roll_rad=roll,
        pitch_rad=pitch,
        yaw_rad=yaw,
        vibration_roll_rad=0.0,
        vibration_pitch_rad=0.0,
        applied_roll=0.0,
        applied_pitch=0.0,
        applied_yaw=0.0,
        applied_throttle=50.0,
        landed=False,
        **defaults,
    )


# A small, cheap ground for the geometric tests: they project points, they do
# not need texture.
_SMALL_GROUND = GroundParams(extent_m=3.0, m_per_px=0.004, relief_fraction=0.0)


class TestProjectionGeometry(unittest.TestCase):
    """The homography is the load-bearing part of the simulator: if it is wrong,
    the flow the estimator sees is wrong, and every control conclusion with it."""

    def setUp(self):
        self.cam = DownwardCamera(CameraParams(), ground=make_ground(_SMALL_GROUND))
        self.f = self.cam.focal_length_px
        self.cx, self.cy = self.cam.principal_point

    def project(self, world_xy, pose_state):
        h = self.cam.world_to_image_homography(pose_from_state(pose_state))
        p = h @ np.array([world_xy[0], world_xy[1], 1.0])
        return p[:2] / p[2]

    def test_point_below_the_camera_lands_on_the_principal_point(self):
        u, v = self.project((0.3, -0.2), state(x=0.3, y=-0.2, z=0.7))
        self.assertAlmostEqual(u, self.cx, places=6)
        self.assertAlmostEqual(v, self.cy, places=6)

    def test_scale_is_focal_length_over_altitude(self):
        for z in (0.4, 0.7, 2.0):
            offset_m = 0.05
            u, v = self.project((offset_m, 0.0), state(z=z))
            # With the 90 deg mount, world +x lies along the image -y axis, so a
            # point to the drone's right appears *above* the principal point.
            # (The camera moving right therefore slides features downward, which
            # is the +dy convention the flow tests check.)
            self.assertAlmostEqual(self.cy - v, self.f * offset_m / z, delta=1e-6 * self.f)
            self.assertAlmostEqual(u, self.cx, delta=1e-9)

    def test_mount_yaw_maps_world_axes_to_image_axes(self):
        """The 90 deg mount is why the autopilot drives roll from vy and pitch
        from vx (README 4.4.2). Encoding it here is what lets the simulator
        catch an axis swap rather than share one.

        Stated for *points*: a point to the right sits up-image, a point ahead
        sits left-image. Equivalently, for *motion*: moving right slides
        features down-image (+dy), moving forward slides them right (+dx).
        """
        u_right, v_right = self.project((0.05, 0.0), state())
        self.assertLess(v_right - self.cy, -10.0, "world +x (right) should lie along image -y")
        self.assertAlmostEqual(u_right, self.cx, delta=1e-9)

        u_fwd, v_fwd = self.project((0.0, 0.05), state())
        self.assertLess(u_fwd - self.cx, -10.0, "world +y (forward) should lie along image -x")
        self.assertAlmostEqual(v_fwd, self.cy, delta=1e-9)

    def test_camera_motion_signs_match_the_autopilot_remap(self):
        """The convention the controller actually depends on."""
        f, z, d = self.f, 0.7, 0.02
        _, dy_right = self.cam.ground_truth_flow_px(state(z=z), state(x=d, z=z))
        dx_fwd, _ = self.cam.ground_truth_flow_px(state(z=z), state(y=d, z=z))
        self.assertGreater(dy_right, 0.0, "drifting right must show as +dy so roll can be driven from vy")
        self.assertGreater(dx_fwd, 0.0, "drifting forward must show as +dx so pitch can be driven from vx")

    def test_zero_mount_yaw_gives_the_unrotated_convention(self):
        cam = DownwardCamera(CameraParams(mount_yaw_deg=0.0), ground=make_ground(_SMALL_GROUND))
        cx, cy = cam.principal_point
        h = cam.world_to_image_homography(pose_from_state(state()))
        p = h @ np.array([0.05, 0.0, 1.0])
        u, v = p[:2] / p[2]
        self.assertGreater(u - cx, 10.0, "without the mount rotation, world +x is image +x")
        self.assertAlmostEqual(v, cy, delta=1e-9)

    def test_yaw_rotates_the_view(self):
        u0, v0 = self.project((0.05, 0.0), state())
        u90, v90 = self.project((0.05, 0.0), state(yaw=math.radians(90.0)))
        r0 = np.array([u0 - self.cx, v0 - self.cy])
        r90 = np.array([u90 - self.cx, v90 - self.cy])
        self.assertAlmostEqual(float(np.linalg.norm(r0)), float(np.linalg.norm(r90)), delta=1e-6)
        cos = float(np.dot(r0, r90) / (np.linalg.norm(r0) * np.linalg.norm(r90)))
        self.assertAlmostEqual(math.degrees(math.acos(np.clip(cos, -1, 1))), 90.0, delta=1e-3)


class TestGroundTruthFlow(unittest.TestCase):
    def setUp(self):
        self.cam = DownwardCamera(CameraParams(), ground=make_ground(_SMALL_GROUND))
        self.f = self.cam.focal_length_px

    def test_translation_flow_matches_f_times_d_over_z(self):
        z, d = 0.7, 0.02
        dx, dy = self.cam.ground_truth_flow_px(state(z=z), state(x=d, z=z))
        self.assertAlmostEqual(dy, self.f * d / z, delta=0.02)
        self.assertAlmostEqual(dx, 0.0, delta=0.02)

    def test_forward_translation_moves_features_along_image_x(self):
        z, d = 0.7, 0.02
        dx, dy = self.cam.ground_truth_flow_px(state(z=z), state(y=d, z=z))
        self.assertAlmostEqual(dx, self.f * d / z, delta=0.02)
        self.assertAlmostEqual(dy, 0.0, delta=0.02)

    def test_flow_halves_when_altitude_doubles(self):
        d = 0.02
        _, near = self.cam.ground_truth_flow_px(state(z=0.7), state(x=d, z=0.7))
        _, far = self.cam.ground_truth_flow_px(state(z=1.4), state(x=d, z=1.4))
        self.assertAlmostEqual(near / far, 2.0, delta=0.01)

    def test_pure_tilt_produces_apparent_flow_of_about_f_times_angle(self):
        """Tilting the camera moves the image even though the drone has not.
        This is the coupling that lets a controller fight its own output.

        The frame-average slightly exceeds the principal-point value f*tan(a):
        rotational flow grows as (1 + u'^2) away from the optical axis, and the
        probe grid spans |u'| up to ~0.7, whose mean is ~1.11. So the expected
        band is f*tan(a) to ~1.15x that, not the textbook value exactly.
        """
        angle = math.radians(1.0)
        dx, dy = self.cam.ground_truth_flow_px(state(), state(roll=angle))
        centre_value = self.f * math.tan(angle)
        self.assertGreater(abs(dy), 0.98 * centre_value)
        self.assertLess(abs(dy), 1.20 * centre_value)
        self.assertAlmostEqual(dx, 0.0, delta=0.5)

    def test_tilt_flow_scales_linearly_with_angle(self):
        one = abs(self.cam.ground_truth_flow_px(state(), state(roll=math.radians(1.0)))[1])
        two = abs(self.cam.ground_truth_flow_px(state(), state(roll=math.radians(2.0)))[1])
        self.assertAlmostEqual(two / one, 2.0, delta=0.03)

    def test_tilt_flow_does_not_depend_on_altitude(self):
        """Translation flow scales as 1/z; rotation flow does not. That is what
        makes the two separable at all, and what the derotation model exploits."""
        low = abs(self.cam.ground_truth_flow_px(state(z=0.5), state(z=0.5, roll=math.radians(1.0)))[1])
        high = abs(self.cam.ground_truth_flow_px(state(z=2.0), state(z=2.0, roll=math.radians(1.0)))[1])
        self.assertAlmostEqual(low, high, delta=0.05 * low)

    def test_pure_vertical_motion_produces_no_net_translation(self):
        """Climbing zooms the image about the principal point. The mean
        displacement over a centred grid is therefore zero -- any nonzero value
        an estimator reports is fabricated lateral velocity."""
        dx, dy = self.cam.ground_truth_flow_px(state(z=0.7), state(z=0.75))
        self.assertAlmostEqual(dx, 0.0, delta=0.05)
        self.assertAlmostEqual(dy, 0.0, delta=0.05)

    def test_translation_only_truth_ignores_attitude_change(self):
        d = 0.02
        moved_and_tilted = state(x=d, roll=math.radians(2.0))
        total = self.cam.ground_truth_flow_px(state(), moved_and_tilted)
        translation = self.cam.ground_truth_translation_flow_px(state(), moved_and_tilted)
        pure = self.cam.ground_truth_flow_px(state(), state(x=d))

        self.assertAlmostEqual(translation[1], pure[1], delta=0.05)
        self.assertGreater(abs(total[1] - translation[1]), 5.0, "tilt should change total flow")

    def test_stationary_camera_has_zero_flow(self):
        dx, dy = self.cam.ground_truth_flow_px(state(), state())
        self.assertAlmostEqual(dx, 0.0, places=9)
        self.assertAlmostEqual(dy, 0.0, places=9)


class TestRendering(unittest.TestCase):
    def test_render_shape_and_type(self):
        cam = DownwardCamera(CameraParams(), ground=make_ground(_SMALL_GROUND))
        r = cam.render(state())
        self.assertEqual(r.frame_bgr.shape, (480, 640, 3))
        self.assertEqual(r.frame_bgr.dtype, np.uint8)

    def test_ground_sampling_distance_is_reported(self):
        cam = DownwardCamera(CameraParams(), ground=make_ground(_SMALL_GROUND))
        r = cam.render(state(z=0.7))
        self.assertAlmostEqual(r.gsd_m_per_px, 0.7 / cam.focal_length_px, places=9)

    def test_out_of_bounds_is_flagged_when_the_drone_leaves_the_world(self):
        ground = make_ground(GroundParams(extent_m=2.0, m_per_px=0.004, relief_fraction=0.0))
        cam = DownwardCamera(CameraParams(), ground=ground)
        self.assertFalse(cam.render(state(z=0.5)).out_of_bounds)
        self.assertTrue(cam.render(state(x=5.0, z=0.5)).out_of_bounds)

    def test_renders_are_deterministic_for_a_given_seed(self):
        ground = make_ground(_SMALL_GROUND)
        a = DownwardCamera(CameraParams(), ground=ground)
        a.seed(3)
        b = DownwardCamera(CameraParams(), ground=ground)
        b.seed(3)
        np.testing.assert_array_equal(a.render(state()).frame_bgr, b.render(state()).frame_bgr)

    def test_motion_blur_reduces_high_frequency_detail(self):
        ground = make_ground(GroundParams(extent_m=3.0, m_per_px=0.0015, relief_fraction=0.0))
        params = CameraParams(sensor_noise_sigma=0.0, jpeg_quality=None, vignette=0.0)
        sharp = DownwardCamera(params.__class__(**{**params.__dict__, "blur_samples": 1}), ground=ground)
        blurred = DownwardCamera(
            params.__class__(**{**params.__dict__, "blur_samples": 5, "exposure_sec": 0.04}), ground=ground
        )
        fast = state(vx_m_s=1.0)

        def detail(img):
            return float(np.var(np.diff(img[:, :, 0].astype(np.float32), axis=1)))

        self.assertLess(detail(blurred.render(fast).frame_bgr), 0.7 * detail(sharp.render(fast).frame_bgr))

    def test_relief_layer_moves_with_its_own_parallax(self):
        """Raised objects must move differently from the floor, otherwise the
        scene is a plane and RANSAC never sees an outlier.

        Checked geometrically rather than by image difference: the texture is
        deliberately sparse (a mostly-flat floor with ~55 speckles per square
        metre, to match the recorded feature counts), so a whole-image mean
        difference is small even when the parallax is large.
        """
        height, z = 0.15, 0.7
        bumpy = make_ground(
            GroundParams(extent_m=3.0, m_per_px=0.0015, relief_fraction=0.4, relief_height_m=height, seed=1)
        )
        self.assertTrue(bumpy.has_relief)
        self.assertAlmostEqual(float(bumpy.relief_mask.mean()), 0.4, delta=0.05)

        cam = DownwardCamera(CameraParams(), ground=bumpy)
        pose = pose_from_state(state(z=z))
        floor_h = cam.world_to_image_homography(pose)
        relief_h = cam.world_to_image_homography(pose, plane_height_m=height)

        # A point at world (0.1, 0) is 0.1 m off the optical axis. Seen on the
        # floor it subtends f*0.1/z; on top of a 0.15 m object, f*0.1/(z-0.15).
        p = np.array([0.1, 0.0, 1.0])
        floor_px = floor_h @ p
        relief_px = relief_h @ p
        floor_offset = abs((floor_px[1] / floor_px[2]) - cam.principal_point[1])
        relief_offset = abs((relief_px[1] / relief_px[2]) - cam.principal_point[1])
        self.assertAlmostEqual(relief_offset / floor_offset, z / (z - height), delta=0.01)

    def test_relief_is_ignored_when_the_camera_is_below_the_object_tops(self):
        bumpy = make_ground(
            GroundParams(extent_m=3.0, m_per_px=0.004, relief_fraction=0.4, relief_height_m=0.5, seed=1)
        )
        cam = DownwardCamera(CameraParams(), ground=bumpy)
        # Should degrade to a floor-only render rather than divide by ~zero.
        frame = cam.render(state(z=0.5)).frame_bgr
        self.assertTrue(np.all(np.isfinite(frame)))
        self.assertEqual(frame.shape, (480, 640, 3))

    def test_planar_ground_has_no_relief(self):
        flat = make_ground(GroundParams(extent_m=2.0, m_per_px=0.004, relief_fraction=0.0, seed=1))
        self.assertFalse(flat.has_relief)
        self.assertIsNone(flat.relief_mask)


class TestGroundTexture(unittest.TestCase):
    def test_world_to_texture_mapping_is_centred_and_y_flipped(self):
        g = make_ground(GroundParams(extent_m=2.0, m_per_px=0.01, relief_fraction=0.0))
        u0, v0 = g.center_px
        self.assertEqual(g.world_to_texture_px(0.0, 0.0), (u0, v0))

        u, v = g.world_to_texture_px(0.5, 0.5)
        self.assertGreater(u, u0, "world +x is texture +u")
        self.assertLess(v, v0, "world +y is texture -v")

    def test_extent_matches_the_request(self):
        g = make_ground(GroundParams(extent_m=3.0, m_per_px=0.005, relief_fraction=0.0))
        self.assertAlmostEqual(g.extent_m[0], 3.0, delta=0.01)

    def test_low_contrast_texture_has_less_variation(self):
        rich = make_ground(GroundParams(extent_m=2.0, m_per_px=0.002, speckle_contrast=0.85, seed=1))
        poor = make_ground(GroundParams(extent_m=2.0, m_per_px=0.002, speckle_contrast=0.1, seed=1))
        self.assertLess(float(np.std(poor.gray)), float(np.std(rich.gray)))


if __name__ == "__main__":
    unittest.main()
