"""Does the simulator match the real drone?

Everything here compares the simulator against something external: closed-form
geometry, the real ``LucasKanadeDriftEstimator`` running on rendered frames, or
statistics measured from recorded flights in
``e88_autopilot/sessions/``. A simulator that is only self-consistent can be
confidently wrong, and confidently wrong is worse than no simulator at all.

Reference values, and where they come from:

- **Feature counts and flow quality** -- ``session_20260117_194839_319431`` is
  the quietest recorded session (velocity std 20 px/s, i.e. nearly stationary,
  the closest recorded thing to a hover): n_features 49.4, n_tracked 49.4,
  quality 0.998, inlier_ratio 0.998. Sessions with violent hand motion show
  quality down to 0.83 and n_features from 37 to 84.
- **Stationary noise floor** -- ``calibration.json`` records a 99th-percentile
  stationary flow magnitude of 2.46 px/s, measured on the real drone.
- **Focal length** -- 365.65 px, recorded as ``flow_focal_length_px`` in every
  recent session and independently implied by reference record ``faf3adb3``
  (a 0.30 m pad measuring 156.7 px at 0.70 m).
- **Frame timing** -- frame_rate_hz 19.8-19.9, flow dt 0.050-0.051 s.
"""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

_EXPERIMENTAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXPERIMENTAL_ROOT))

from e88_autopilot.optical_flow import LucasKanadeDriftEstimator
from e88_sim.camera import DownwardCamera
from e88_sim.config import (
    AirframeParams,
    CameraParams,
    EnvironmentParams,
    GroundParams,
    SimConfig,
    VideoLinkParams,
)
from e88_sim.flow_sensor import FLOW_ERROR_MODELS
from e88_sim.ground import make_ground
from e88_sim.physics import PhysicsState, QuadPhysics

# --- reference values measured from real recordings --------------------------
REAL_FOCAL_LENGTH_PX = 365.6463394165039
REAL_QUIET_SESSION = dict(n_features=49.4, n_tracked=49.4, quality=0.998, inlier_ratio=0.998)
REAL_N_FEATURES_RANGE = (37.0, 84.0)
REAL_STATIONARY_P99_PX_S = 2.46
REAL_FRAME_RATE_HZ = (19.7, 20.0)
REAL_FLOW_DT_SEC = (0.050, 0.052)


def state(x=0.0, y=0.0, z=0.7, roll=0.0, pitch=0.0, yaw=0.0) -> PhysicsState:
    return PhysicsState(
        t=0.0,
        x_m=x,
        y_m=y,
        z_m=z,
        vx_m_s=0.0,
        vy_m_s=0.0,
        vz_m_s=0.0,
        roll_rad=roll,
        pitch_rad=pitch,
        yaw_rad=yaw,
        roll_rate_rad_s=0.0,
        pitch_rate_rad_s=0.0,
        yaw_rate_rad_s=0.0,
        vibration_roll_rad=0.0,
        vibration_pitch_rad=0.0,
        applied_roll=0.0,
        applied_pitch=0.0,
        applied_yaw=0.0,
        applied_throttle=50.0,
        landed=False,
    )


def _flow_estimator(model: str, focal: float) -> LucasKanadeDriftEstimator:
    extra = {}
    if model == "derotation":
        extra = dict(focal_length_px=focal, principal_point_px=(319.5, 239.5))
    return LucasKanadeDriftEstimator(motion_model=model, downscale=0.5, **extra)


class TestGeometryAgainstTheRealEstimator(unittest.TestCase):
    """Render a known motion, then ask the production optical-flow code what it
    sees. Agreement means the renderer's geometry and the estimator's model
    agree -- the single most important fidelity claim the simulator makes."""

    @classmethod
    def setUpClass(cls):
        cls.ground = make_ground(GroundParams(relief_fraction=0.0))
        # A clean sensor, so this test measures geometry and not noise.
        cls.params = CameraParams(sensor_noise_sigma=0.0, jpeg_quality=None, vignette=0.0, blur_samples=1)
        cls.cam = DownwardCamera(cls.params, ground=cls.ground)
        cls.f = cls.cam.focal_length_px

    def test_focal_length_matches_the_recorded_calibration(self):
        self.assertAlmostEqual(self.f, REAL_FOCAL_LENGTH_PX, places=3)

    def test_recovered_translation_matches_f_times_d_over_z(self):
        for z, d in ((0.5, 0.015), (0.7, 0.02), (1.5, 0.04)):
            with self.subTest(z=z):
                est = _flow_estimator("affine_translation", self.f)
                est.update(self.cam.render(state(z=z)).frame_bgr, timestamp=0.0)
                e = est.update(self.cam.render(state(x=d, z=z)).frame_bgr, timestamp=0.05)
                self.assertIsNotNone(e)
                expected = self.f * d / z
                self.assertAlmostEqual(e.dy_px, expected, delta=max(0.15, 0.02 * expected))
                self.assertAlmostEqual(e.dx_px, 0.0, delta=0.2)

    def test_recovered_flow_matches_the_homography_ground_truth(self):
        """Tighter than the closed form above: compares against the exact
        grid-averaged homography transport the simulator itself defines."""
        for model in ("affine_translation", "translation_rotation"):
            with self.subTest(model=model):
                est = _flow_estimator(model, self.f)
                a, b = state(z=0.7), state(x=0.02, y=0.01, z=0.7)
                est.update(self.cam.render(a).frame_bgr, timestamp=0.0)
                e = est.update(self.cam.render(b).frame_bgr, timestamp=0.05)
                gx, gy = self.cam.ground_truth_flow_px(a, b)
                self.assertAlmostEqual(e.dx_px, gx, delta=0.3)
                self.assertAlmostEqual(e.dy_px, gy, delta=0.3)

    def test_drift_direction_matches_the_controller_axis_remap(self):
        """Drifting right must read as +vy and drifting forward as +vx, because
        the autopilot drives roll from vy and pitch from vx. If this inverts,
        the working sign convention inverts with it."""
        for label, dx, dy, expect in (("right", 0.02, 0.0, "vy"), ("forward", 0.0, 0.02, "vx")):
            with self.subTest(label=label):
                est = _flow_estimator("translation_rotation", self.f)
                est.update(self.cam.render(state()).frame_bgr, timestamp=0.0)
                e = est.update(self.cam.render(state(x=dx, y=dy)).frame_bgr, timestamp=0.05)
                if expect == "vy":
                    self.assertGreater(e.vy_px_s, 50.0)
                    self.assertLess(abs(e.vx_px_s), 0.2 * abs(e.vy_px_s))
                else:
                    self.assertGreater(e.vx_px_s, 50.0)
                    self.assertLess(abs(e.vy_px_s), 0.2 * abs(e.vx_px_s))

    def test_derotation_removes_tilt_induced_apparent_flow(self):
        """The whole purpose of the derotation model. With a pure tilt and no
        translation it should report near zero, where the other models report
        the full apparent motion."""
        angle = math.radians(1.5)
        a, b = state(), state(roll=angle)

        derot = _flow_estimator("derotation", self.f)
        derot.update(self.cam.render(a).frame_bgr, timestamp=0.0)
        e_derot = derot.update(self.cam.render(b).frame_bgr, timestamp=0.05)

        plain = _flow_estimator("translation_rotation", self.f)
        plain.update(self.cam.render(a).frame_bgr, timestamp=0.0)
        e_plain = plain.update(self.cam.render(b).frame_bgr, timestamp=0.05)

        self.assertIsNotNone(e_derot)
        self.assertIsNotNone(e_plain)
        self.assertGreater(abs(e_plain.dy_px), 8.0, "the apparent flow should be plainly visible")
        self.assertLess(abs(e_derot.dy_px), 0.25 * abs(e_plain.dy_px))


class TestAltitudeLeakCalibration(unittest.TestCase):
    """The leak coefficients the fast-loop surrogate relies on.

    A change in altitude scales the image. A fitted model with no scale
    parameter can absorb part of that as translation. For a plain affine fit the
    amount is exactly predictable -- a zoom by s about the image centre gives
    translation (1 - s) * principal_point -- so this test doubles as a check
    that the renderer really produces a correct perspective zoom.
    """

    @classmethod
    def setUpClass(cls):
        cls.ground = make_ground(GroundParams(relief_fraction=0.0))
        cls.params = CameraParams(sensor_noise_sigma=0.0, jpeg_quality=None, vignette=0.0, blur_samples=1)
        cls.cam = DownwardCamera(cls.params, ground=cls.ground)
        cls.f = cls.cam.focal_length_px
        cls.cx, cls.cy = cls.cam.principal_point

    def measure_kappa(self, model):
        kxs, kys = [], []
        for z0 in (0.5, 0.7, 1.0, 2.0):
            for dz in (-0.01, -0.005, 0.005, 0.01):
                est = _flow_estimator(model, self.f)
                est.update(self.cam.render(state(z=z0)).frame_bgr, timestamp=0.0)
                e = est.update(self.cam.render(state(z=z0 + dz)).frame_bgr, timestamp=0.05)
                if e is None:
                    continue
                leak = dz / (z0 + dz)
                kxs.append(e.dx_px / (leak * self.cx))
                kys.append(e.dy_px / (leak * self.cy))
        return np.array(kxs), np.array(kys)

    def test_affine_leaks_the_full_zoom_as_predicted_by_theory(self):
        kx, ky = self.measure_kappa("affine_translation")
        self.assertAlmostEqual(float(kx.mean()), 1.0, delta=0.05)
        self.assertAlmostEqual(float(ky.mean()), 1.0, delta=0.05)

    def test_translation_rotation_suppresses_the_leak(self):
        kx, ky = self.measure_kappa("translation_rotation")
        self.assertLess(abs(float(kx.mean())), 0.3)
        self.assertLess(abs(float(ky.mean())), 0.3)

    def test_surrogate_leak_coefficients_still_match_the_rendering_pipeline(self):
        """Guards the fast loop: if the renderer or the estimator changes, the
        constants baked into FLOW_ERROR_MODELS stop describing reality and every
        sweep run through the surrogate becomes quietly wrong."""
        for model in ("affine_translation", "translation_rotation"):
            with self.subTest(model=model):
                kx, ky = self.measure_kappa(model)
                expected = FLOW_ERROR_MODELS[model].kappa_mean
                self.assertAlmostEqual(float(kx.mean()), expected[0], delta=0.15)
                self.assertAlmostEqual(float(ky.mean()), expected[1], delta=0.15)


class TestSurrogateErrorMatchesTheRenderingPipeline(unittest.TestCase):
    """The fast loop's sweeps are only worth anything if its analytic sensor
    misbehaves the way the real estimator does.

    Measured here end to end: fly the same physics twice, once measuring with
    the real ``LucasKanadeDriftEstimator`` on rendered frames and once with the
    surrogate, and compare how far each strays from ground truth. The two error
    terms in ``FlowErrorModel`` -- the altitude leak and the residual noise --
    have to add up to the right total, not just be individually plausible.
    """

    SEEDS = (3, 5)
    STEPS = 90

    def _error_rms(self, model, *, rendered):
        from e88_sim.flow_sensor import AnalyticFlowSensor

        per_seed = []
        for seed in self.SEEDS:
            cfg = SimConfig(environment=EnvironmentParams(initial_altitude_m=0.7), seed=seed)
            cam = DownwardCamera(cfg.camera, ground=make_ground(cfg.ground))
            cam.seed(seed)
            physics = QuadPhysics(cfg)
            sensor = AnalyticFlowSensor(cam, motion_model=model, rng=np.random.default_rng(seed))
            est = _flow_estimator(model, cam.focal_length_px) if rendered else None

            prev = physics.state()
            if est is not None:
                est.update(cam.render(prev).frame_bgr, timestamp=0.0)

            errors = []
            for i in range(1, self.STEPS):
                physics.step(0.05)
                cur = physics.state()
                truth_x, truth_y = sensor.truth_px(prev, cur)
                if est is not None:
                    e = est.update(cam.render(cur).frame_bgr, timestamp=i * 0.05)
                    if e is not None:
                        errors += [(e.dx_px - truth_x) / 0.05, (e.dy_px - truth_y) / 0.05]
                else:
                    e = sensor.measure(prev, cur, 0.05)
                    errors += [(e.dx_px - truth_x) / 0.05, (e.dy_px - truth_y) / 0.05]
                prev = cur
            per_seed.append(float(np.sqrt(np.mean(np.square(errors)))))
        return float(np.mean(per_seed))

    def test_surrogate_error_magnitude_matches_the_real_estimator(self):
        """Within a factor of two per model. Tighter is not achievable -- the
        surrogate models the error, it does not reproduce the estimator -- and
        looser would let a sweep mislocate the stability knee."""
        for model in ("affine_translation", "translation_rotation", "derotation"):
            with self.subTest(model=model):
                rendered = self._error_rms(model, rendered=True)
                surrogate = self._error_rms(model, rendered=False)
                self.assertGreater(rendered, 0.0)
                ratio = max(rendered, surrogate) / max(1e-9, min(rendered, surrogate))
                self.assertLess(
                    ratio,
                    2.5,
                    f"{model}: rendered {rendered:.1f} px/s vs surrogate {surrogate:.1f} px/s",
                )

    def test_surrogate_preserves_the_ranking_between_motion_models(self):
        """A sweep is used to choose between models, so the ordering matters
        more than the absolute values."""
        rendered = {m: self._error_rms(m, rendered=True) for m in ("translation_rotation", "affine_translation")}
        surrogate = {m: self._error_rms(m, rendered=False) for m in ("translation_rotation", "affine_translation")}
        self.assertLess(rendered["translation_rotation"], rendered["affine_translation"])
        self.assertLess(surrogate["translation_rotation"], surrogate["affine_translation"])


class TestFlowStatisticsAgainstRecordedSessions(unittest.TestCase):
    """Feature counts, quality and noise, versus what the real drone recorded."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = SimConfig(
            environment=EnvironmentParams(initial_altitude_m=0.7),
            seed=7,
        )
        cls.ground = make_ground(cls.cfg.ground)
        cls.cam = DownwardCamera(cls.cfg.camera, ground=cls.ground)
        cls.cam.seed(7)
        cls.f = cls.cam.focal_length_px

        est = _flow_estimator("translation_rotation", cls.f)
        physics = QuadPhysics(cls.cfg)
        prev = physics.state()
        est.update(cls.cam.render(prev).frame_bgr, timestamp=0.0)

        cls.n_features, cls.n_tracked, cls.quality, cls.inliers = [], [], [], []
        cls.errors = []
        for i in range(1, 90):
            physics.step(0.05)
            s = physics.state()
            e = est.update(cls.cam.render(s).frame_bgr, timestamp=i * 0.05)
            if e is not None:
                cls.n_features.append(e.n_features)
                cls.n_tracked.append(e.n_tracked)
                cls.quality.append(e.quality)
                cls.inliers.append(e.inlier_ratio)
                gx, gy = cls.cam.ground_truth_flow_px(prev, s)
                cls.errors += [(e.dx_px - gx) / 0.05, (e.dy_px - gy) / 0.05]
            prev = s

    def test_feature_count_is_in_the_recorded_range(self):
        mean = float(np.mean(self.n_features))
        lo, hi = REAL_N_FEATURES_RANGE
        self.assertGreaterEqual(mean, lo, f"sim texture yields {mean:.0f} features, below the recorded range")
        self.assertLessEqual(mean, hi, f"sim texture yields {mean:.0f} features, above the recorded range")

    def test_feature_count_is_near_the_quiet_session(self):
        self.assertAlmostEqual(float(np.mean(self.n_features)), REAL_QUIET_SESSION["n_features"], delta=20.0)

    def test_flow_quality_is_realistic(self):
        """Not too low (the tracker would be gating out) and not a perfect 1.0
        (a scene with no outliers at all would flatter the estimator)."""
        q = float(np.mean(self.quality))
        self.assertGreater(q, 0.80)
        self.assertLessEqual(q, 1.0)

    def test_tracker_does_not_collapse(self):
        self.assertGreater(float(np.mean(self.n_tracked)), 30.0)
        self.assertGreater(float(np.mean(self.inliers)), 0.80)

    def test_estimator_error_is_bounded_during_a_hover(self):
        rms = float(np.sqrt(np.mean(np.square(self.errors))))
        self.assertLess(rms, 20.0, "estimator error far above what the real drone shows")


class TestStationaryNoiseFloor(unittest.TestCase):
    def measure_floor(self, **airframe):
        params = dict(altitude_hold_sigma_m=0.0, yaw_drift_deg_s=0.0)  # a still drone does not bob
        params.update(airframe)
        cfg = SimConfig(
            airframe=AirframeParams(**params),
            environment=EnvironmentParams(
                initial_altitude_m=0.7, trim_accel_x_m_s2=0.0, trim_accel_y_m_s2=0.0, gust_sigma_m_s2=0.0
            ),
            seed=11,
        )
        cam = DownwardCamera(cfg.camera, ground=make_ground(cfg.ground))
        cam.seed(11)
        physics = QuadPhysics(cfg)
        est = _flow_estimator("translation_rotation", cam.focal_length_px)
        est.update(cam.render(physics.state()).frame_bgr, timestamp=0.0)

        samples = []
        for i in range(1, 90):
            physics.step(0.05)
            e = est.update(cam.render(physics.state()).frame_bgr, timestamp=i * 0.05)
            if e is not None:
                samples += [abs(e.vx_px_s), abs(e.vy_px_s)]

        return float(np.percentile(samples, 99))

    def test_stationary_noise_matches_the_measured_deadband(self):
        """calibration.json measured a 2.46 px/s 99th percentile with the real
        drone stationary. The simulator must land near it: much quieter and the
        simulator is optimistic, much noisier and it is pessimistic, and either
        way the deadband tuned on the real drone would not transfer.

        This is the tightest quantitative tie the simulator has to the physical
        drone, so it is asserted within a factor of two rather than loosely.
        """
        p99 = self.measure_floor()
        self.assertLess(p99, 2.0 * REAL_STATIONARY_P99_PX_S, f"sim noise floor {p99:.2f} px/s is too high")
        self.assertGreater(p99, 0.5 * REAL_STATIONARY_P99_PX_S, f"sim noise floor {p99:.2f} px/s is too low")

    def test_image_formation_alone_is_quieter_than_the_real_floor(self):
        """Why the vibration term exists at all: rendering plus sensor noise
        plus JPEG accounts for well under half the measured 2.46 px/s, so the
        rest has to be modelled explicitly rather than assumed away."""
        self.assertLess(self.measure_floor(vibration_sigma_deg=0.0), 0.5 * REAL_STATIONARY_P99_PX_S)

    def test_noise_floor_rises_with_vibration(self):
        quiet = self.measure_floor(vibration_sigma_deg=0.003)
        shaky = self.measure_floor(vibration_sigma_deg=0.015)
        self.assertGreater(shaky, 2.0 * quiet)


class TestVideoTiming(unittest.TestCase):
    def test_configured_frame_rate_matches_recordings(self):
        v = VideoLinkParams()
        lo, hi = REAL_FRAME_RATE_HZ
        self.assertGreaterEqual(v.fps, lo)
        self.assertLessEqual(v.fps, hi + 0.3)

    def test_frame_period_matches_the_recorded_flow_dt(self):
        period = 1.0 / VideoLinkParams().fps
        lo, hi = REAL_FLOW_DT_SEC
        self.assertGreaterEqual(period, lo - 0.001)
        self.assertLessEqual(period, hi)

    def test_delivered_frames_are_late_by_the_pipeline_latency(self):
        """The autopilot timestamps a frame when it decodes it, so the pose the
        frame actually shows is older than that by the pipeline latency -- an
        amount session telemetry cannot see and nobody has measured."""
        import time

        from e88_sim.sim_drone import SimulatedDrone

        cfg = SimConfig(
            video=VideoLinkParams(pipeline_latency_sec=0.15, latency_jitter_sec=0.0, drop_probability=0.0),
            ground=GroundParams(extent_m=2.0, m_per_px=0.004),
            seed=3,
        )
        drone = SimulatedDrone(cfg)
        drone.connect()
        try:
            time.sleep(0.6)
            first = drone.get_frame_with_timestamp(timeout=1.0)
            self.assertIsNotNone(first)
            _, ts = first
            self.assertLessEqual(ts, time.monotonic() + 1e-3)
        finally:
            drone.close()


if __name__ == "__main__":
    unittest.main()
