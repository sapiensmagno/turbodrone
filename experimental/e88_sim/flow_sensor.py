"""An analytic stand-in for the optical-flow estimator.

Rendering a frame and running Lucas-Kanade on it costs ~15 ms, which caps the
vision-in-the-loop simulator at real time. That is fine for validating the
wiring, and useless for asking "at what gain does this become unstable, across
five altitudes, three latencies and twenty seeds". This sensor answers the same
control question hundreds of times faster by computing what the estimator
*would* have reported, straight from the pose pair, and corrupting it with
measurement noise calibrated against the real thing.

It is a surrogate, and it is only trustworthy because it is checked:
``tests/test_sim_fast_loop.py`` runs the same scenario through both the analytic
sensor and the full rendering pipeline and asserts the resulting flight
statistics agree. Any conclusion drawn from a sweep should be confirmed with one
vision-in-the-loop run before it is acted on.
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Optional, Tuple

import numpy as np

from e88_autopilot.optical_flow import FlowEstimate
from e88_sim.camera import DownwardCamera
from e88_sim.physics import PhysicsState


# Typical flow quality during a nominal hover, matching recorded sessions
# (0.83-0.91 across sessions with motion; 0.998 in the quietest one).
DEFAULT_QUALITY: float = 0.95


class FlowErrorModel(NamedTuple):
    """How a motion model's output departs from truth.

    Two separate effects, because they behave completely differently in a
    control loop:

    ``kappa`` -- **altitude leak**. A change in altitude scales the image, and a
    fitted model that has no scale parameter can absorb some of that scaling as
    translation. For an affine fit the amount is exactly predictable: a zoom by
    ``s`` about the image centre gives translation ``(1 - s) * c``, where ``c``
    is the principal point, so ``dx = (dz/z) * cx``. Measured on rendered
    frames, ``affine_translation`` shows kappa = 1.000 +- 0.014 -- the textbook
    value, confirming this is the mechanism and not a rendering artifact.
    This error is *driven by vertical motion*, so it is correlated with the
    drone's own altitude-hold bobbing rather than being white noise.

    ``noise`` -- everything else (sub-pixel matching error, tilt cross-coupling,
    compression). Measured with altitude bobbing switched **off**, precisely so
    it does not double-count the leak: the two terms are summed at runtime, and
    together they reproduce the error measured on rendered frames under full
    hover conditions.

    That decomposition is worth checking, because the leak dominates by
    different amounts per model. Under 30 mm of altitude bobbing, subtracting
    the mean leak removes 96% of affine's error variance -- the leak is
    essentially all of it -- but only 6% of derotation's, because derotation's
    leak is near-zero on average and enormous in variance (kappa_y = 0.11 with a
    standard deviation of 1.00). Its error is not a bias to correct for; it is
    unpredictable frame to frame.

    Calibration procedure and numbers live in tests/test_sim_realism.py, which
    re-measures them against the rendering pipeline and fails if they drift.
    """

    kappa_mean: Tuple[float, float]
    kappa_std: Tuple[float, float]
    noise_px_s: float


# kappa measured at 640x480, f=365.6, principal point (319.5, 239.5), over
# altitudes 0.5-2.0 m and steps of +-5/10 mm. noise_px_s measured over a
# simulated hover with altitude bobbing disabled.
#
# End-to-end, the two terms together reproduce the error the real estimator
# makes on rendered frames under a full hover (30 mm bobbing, gusts, trim):
#   affine_translation    rendered 52.9 px/s RMS
#   translation_rotation  rendered  7.7 px/s RMS
#   derotation            rendered 64.1 px/s RMS
FLOW_ERROR_MODELS: Dict[str, FlowErrorModel] = {
    # The full zoom leaks straight through as translation.
    "affine_translation": FlowErrorModel(kappa_mean=(1.00, 1.00), kappa_std=(0.014, 0.021), noise_px_s=9.8),
    # Fitting an explicit rotation leaves little room for scale to masquerade
    # as translation: the leak drops by a factor of ~11.
    "translation_rotation": FlowErrorModel(kappa_mean=(0.092, 0.055), kappa_std=(0.086, 0.088), noise_px_s=1.8),
    # The tilt terms and the zoom compete for the same radial signal, so the
    # leak is not just larger than translation_rotation's, it is erratic: the
    # y-axis standard deviation of 1.0 dwarfs its 0.11 mean, meaning individual
    # frames can invert or amplify it.
    "derotation": FlowErrorModel(kappa_mean=(0.236, 0.108), kappa_std=(0.315, 0.996), noise_px_s=4.5),
}


def error_model_for(motion_model: str) -> FlowErrorModel:
    return FLOW_ERROR_MODELS.get(str(motion_model), FLOW_ERROR_MODELS["translation_rotation"])


class AnalyticFlowSensor:
    """Predicts ``LucasKanadeDriftEstimator`` output without rendering."""

    def __init__(
        self,
        camera: DownwardCamera,
        *,
        motion_model: str = "translation_rotation",
        error_model: Optional[FlowErrorModel] = None,
        quality: float = DEFAULT_QUALITY,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self._camera = camera
        self._motion_model = str(motion_model)
        self._err = error_model if error_model is not None else error_model_for(self._motion_model)
        self._quality = float(quality)
        self._rng = rng if rng is not None else np.random.default_rng(0)
        self._cx, self._cy = camera.principal_point

    @property
    def derotating(self) -> bool:
        """Whether the modelled estimator subtracts tilt-induced apparent flow.

        This is not cosmetic. A derotating estimator reports translation only,
        so commanded tilt does not feed back into the measurement; a
        non-derotating one reports the tilt as apparent motion, closing an extra
        feedback path from the controller's own output back into its input. The
        two behave very differently near the stability limit, so the surrogate
        has to get this right.
        """
        return self._motion_model == "derotation"

    def truth_px(self, prev_state: PhysicsState, state: PhysicsState) -> tuple:
        if self.derotating:
            return self._camera.ground_truth_translation_flow_px(prev_state, state)
        return self._camera.ground_truth_flow_px(prev_state, state)

    def measure(self, prev_state: PhysicsState, state: PhysicsState, dt_sec: float) -> Optional[FlowEstimate]:
        dt = float(dt_sec)
        if dt <= 1e-6:
            return None

        dx, dy = self.truth_px(prev_state, state)

        # Altitude leak: the relative image scale change between the two frames,
        # times the per-model leak coefficient, times the principal point offset.
        z_cur = float(state.z_m)
        if z_cur > 1e-3:
            leak = (z_cur - float(prev_state.z_m)) / z_cur
            kx = float(self._rng.normal(self._err.kappa_mean[0], self._err.kappa_std[0]))
            ky = float(self._rng.normal(self._err.kappa_mean[1], self._err.kappa_std[1]))
            dx += kx * leak * self._cx
            dy += ky * leak * self._cy

        if self._err.noise_px_s > 0.0:
            # Residual noise enters as a per-frame displacement error, not a
            # velocity error: that is how it actually arises (sub-pixel matching
            # error), so shortening dt amplifies the resulting velocity noise
            # exactly as it does in the real estimator.
            sigma_px = float(self._err.noise_px_s) * dt
            dx += float(self._rng.normal(0.0, sigma_px))
            dy += float(self._rng.normal(0.0, sigma_px))

        return FlowEstimate(
            dt_sec=dt,
            dx_px=float(dx),
            dy_px=float(dy),
            vx_px_s=float(dx / dt),
            vy_px_s=float(dy / dt),
            quality=float(self._quality),
            n_features=60,
            n_tracked=57,
            inlier_ratio=0.95,
            fallback_used=False,
            motion_model=self._motion_model,
            raw_dx_px=float(dx),
            raw_dy_px=float(dy),
        )
