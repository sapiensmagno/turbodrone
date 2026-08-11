"""Pre-flight scenarios.

Each scenario is a question you would want answered before letting the drone
leave the ground, expressed as a simulator configuration. Two kinds:

- **Conditions** -- nominal hover, an untrimmed airframe, a gust, an altitude
  the gains were not tuned at, a frozen video feed. A configuration must survive
  all of these, not just the easy one.
- **Falsification** -- scenarios that are *supposed* to fail, above all inverted
  sign conventions. A simulator that passes everything is not measuring
  anything; ``wrong_signs`` diverging is the evidence that a PASS elsewhere
  means something.

Several scenarios sweep parameters marked [ESTIMATED] in ``config.py`` --
maximum lean angle, video latency, altitude bobbing. Those are the numbers
nobody has measured on this airframe, so a verdict that only holds for one value
of them is not worth much. ``robustness_suite`` exists to check that the
conclusion survives the whole plausible range.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from e88_autopilot.autostabilizer import StabilizerConfig
from e88_sim.config import (
    AirframeParams,
    CameraParams,
    EnvironmentParams,
    GroundParams,
    SimConfig,
    VideoLinkParams,
)


# The configuration recorded in the most recent real sessions
# (sessions/session_20260208_*/meta.json), which is what the drone would fly
# today. Deadband and sigma_v come from calibration.json.
def default_stabilizer_config(**overrides) -> StabilizerConfig:
    base = dict(
        enable_takeoff=False,
        base_throttle=50.0,
        cmd_rate_hz=20.0,
        use_kalman=True,
        kalman_sigma_a=25.0,
        kalman_sigma_v=50.0,
        min_quality=0.15,
        max_cmd=0.9,
        flow_motion_model="translation_rotation",
        flow_focal_length_px=365.6463394165039,
        flow_principal_point_px=(319.5, 239.5),
        deadband_px_s=2.46,
        estimator_deadband_px_s=2.46,
        kp_vx=0.003,
        kp_vy=0.003,
        ki_vx=0.0005,
        ki_vy=0.0005,
        roll_sign=-1.0,
        pitch_sign=-1.0,
        enable_visual_scale=False,
    )
    base.update(overrides)
    return StabilizerConfig(**base)


@dataclass(frozen=True)
class Scenario:
    name: str
    question: str
    sim: SimConfig
    stab: StabilizerConfig
    duration_sec: float = 20.0
    # Seconds excluded from the steady-state statistics. The default skips the
    # estimator's cold start; a disturbance scenario pushes it past the
    # disturbance, so the score measures recovery rather than the shove itself.
    # Divergence checks always cover the whole run regardless.
    settle_sec: float = 2.0
    # Scenarios that are expected to fail. If one of these ever passes, the
    # simulator has stopped discriminating and its other verdicts are worthless.
    expect_failure: bool = False
    seeds: Tuple[int, ...] = (1, 2, 3)


def _sim(
    *,
    altitude_m: float = 0.7,
    seed: int = 1,
    airframe: Optional[AirframeParams] = None,
    environment: Optional[EnvironmentParams] = None,
    video: Optional[VideoLinkParams] = None,
    ground: Optional[GroundParams] = None,
    camera: Optional[CameraParams] = None,
) -> SimConfig:
    env = environment or EnvironmentParams()
    env = replace(env, initial_altitude_m=float(altitude_m))
    return SimConfig(
        airframe=airframe or AirframeParams(),
        environment=env,
        video=video or VideoLinkParams(),
        ground=ground or GroundParams(),
        camera=camera or CameraParams(),
        seed=int(seed),
    )


def build_scenarios(stab: Optional[StabilizerConfig] = None) -> List[Scenario]:
    """The standard pre-flight battery for a given autopilot configuration."""
    s = stab or default_stabilizer_config()

    return [
        Scenario(
            name="nominal_hover",
            question="Does it hold a hover at the altitude it was tuned for?",
            sim=_sim(altitude_m=0.7),
            stab=s,
        ),
        Scenario(
            name="untrimmed_airframe",
            question="Can the integrator cancel a real trim bias, or does it wind up?",
            sim=_sim(
                altitude_m=0.7,
                environment=EnvironmentParams(trim_accel_x_m_s2=0.25, trim_accel_y_m_s2=-0.15),
            ),
            stab=s,
        ),
        Scenario(
            name="wind_gust",
            question=(
                "Does it recover from a shove, or overshoot into an oscillation? "
                "0.8 m/s^2 for 0.6 s is an indoor draft or a downwash bounce, not a storm."
            ),
            sim=_sim(
                altitude_m=0.7,
                environment=EnvironmentParams(
                    gust_pulse_accel_m_s2=(0.8, 0.0),
                    gust_pulse_start_sec=6.0,
                    gust_pulse_duration_sec=0.6,
                ),
            ),
            stab=s,
            # Score the recovery, not the gust: the peak excursion still shows up
            # in speed_max and would still trip the divergence check.
            settle_sec=9.0,
        ),
        Scenario(
            name="low_altitude",
            question=(
                "At 0.4 m the pixel-per-metre gain is 1.75x what it is at 0.7 m. "
                "Does a px/s-tuned loop stay stable when it is flown lower?"
            ),
            sim=_sim(altitude_m=0.4),
            stab=s,
        ),
        Scenario(
            name="high_altitude",
            question="At 2 m the loop gain halves. Does it still correct, or go sluggish?",
            sim=_sim(altitude_m=2.0),
            stab=s,
        ),
        Scenario(
            name="high_latency",
            question=(
                "Video latency has never been measured on this drone. If it is 250 ms "
                "rather than the assumed 120 ms, is the loop still stable?"
            ),
            sim=_sim(altitude_m=0.7, video=VideoLinkParams(pipeline_latency_sec=0.25, latency_jitter_sec=0.03)),
            stab=s,
        ),
        Scenario(
            name="low_texture",
            question="Over a near-featureless floor, does it gate out safely or chase noise?",
            sim=_sim(
                altitude_m=0.7,
                ground=GroundParams(speckle_density_per_m2=12.0, speckle_contrast=0.25, background_contrast=0.05),
            ),
            stab=s,
        ),
        Scenario(
            name="frozen_feed",
            question=(
                "If the RTSP feed stalls for 2 s mid-flight, does max_stale_frame_hold_sec "
                "neutralize before the latched command flies it away?"
            ),
            sim=_sim(
                altitude_m=0.7,
                video=VideoLinkParams(freeze_start_sec=8.0, freeze_duration_sec=2.0),
            ),
            stab=s,
        ),
        Scenario(
            name="lossy_link",
            question="Does heavier packet loss and frame tearing destabilize it?",
            sim=_sim(
                altitude_m=0.7,
                video=VideoLinkParams(drop_probability=0.05, tear_probability=0.03),
            ),
            stab=s,
        ),
        Scenario(
            name="weak_airframe",
            question=(
                "max_tilt_deg is an estimate. If the E88 only leans 10 deg at full stick, "
                "the loop gain halves -- does it still hold?"
            ),
            sim=_sim(altitude_m=0.7, airframe=AirframeParams(max_tilt_deg=10.0)),
            stab=s,
        ),
        Scenario(
            name="strong_airframe",
            question="And if it leans 30 deg, does the doubled gain tip it into oscillation?",
            sim=_sim(altitude_m=0.7, airframe=AirframeParams(max_tilt_deg=30.0)),
            stab=s,
        ),
        Scenario(
            name="bouncy_altitude_hold",
            question=(
                "Baro altitude hold bobs. At 8 cm of bobbing, how much phantom lateral "
                "velocity does the flow model invent, and does the loop chase it?"
            ),
            sim=_sim(altitude_m=0.7, airframe=AirframeParams(altitude_hold_sigma_m=0.08)),
            stab=s,
        ),
        Scenario(
            name="noisy_camera",
            question=(
                "The stationary noise floor was calibrated with the drone still, probably "
                "with motors off. If a hovering drone is 5x noisier, does the loop chase it?"
            ),
            sim=_sim(altitude_m=0.7, airframe=AirframeParams(vibration_sigma_deg=0.015)),
            stab=s,
        ),
        # --- falsification -------------------------------------------------
        Scenario(
            name="wrong_signs",
            question="FALSIFICATION: inverted sign conventions must diverge.",
            sim=_sim(altitude_m=0.7),
            stab=replace(s, roll_sign=+1.0, pitch_sign=+1.0),
            expect_failure=True,
            seeds=(1,),
        ),
        Scenario(
            name="wrong_roll_sign_only",
            question="FALSIFICATION: one inverted axis must diverge on that axis.",
            sim=_sim(altitude_m=0.7),
            stab=replace(s, roll_sign=+1.0),
            expect_failure=True,
            seeds=(1,),
        ),
        Scenario(
            name="excessive_gain",
            question="FALSIFICATION: 5x the gain must produce visible instability.",
            sim=_sim(altitude_m=0.7),
            stab=replace(s, kp_vx=0.015, kp_vy=0.015),
            expect_failure=True,
            seeds=(1,),
        ),
    ]


def gain_sweep(
    *,
    stab: Optional[StabilizerConfig] = None,
    kp_values: Sequence[float] = (0.0002, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005),
    ki_ratio: float = 1.0 / 6.0,
    altitude_m: float = 0.7,
    seeds: Sequence[int] = (1, 2, 3),
) -> List[Scenario]:
    """Locate the stability knee in proportional gain.

    ``ki`` is tied to ``kp`` by a fixed ratio so the sweep moves along a
    one-dimensional family rather than exploring a grid where the two gains
    trade off against each other.
    """
    base = stab or default_stabilizer_config()
    out: List[Scenario] = []
    for kp in kp_values:
        out.append(
            Scenario(
                name=f"kp={kp:.4f}",
                question=f"Is kp={kp:.4f} stable at {altitude_m:.1f} m?",
                sim=_sim(altitude_m=altitude_m),
                stab=replace(base, kp_vx=kp, kp_vy=kp, ki_vx=kp * ki_ratio, ki_vy=kp * ki_ratio),
                seeds=tuple(seeds),
            )
        )
    return out


def latency_sweep(
    *,
    stab: Optional[StabilizerConfig] = None,
    latencies_sec: Sequence[float] = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40),
    altitude_m: float = 0.7,
    seeds: Sequence[int] = (1, 2, 3),
) -> List[Scenario]:
    """How much video latency can the current gains tolerate?

    This is the single most valuable sweep available, because the true
    photon-to-command latency of this drone is unknown: session telemetry only
    measures from decode onward. The sweep converts that unknown into a stated
    tolerance -- "these gains are stable up to X ms" -- which can be checked
    against a one-off latency measurement instead of guessed at.
    """
    base = stab or default_stabilizer_config()
    out: List[Scenario] = []
    for lat in latencies_sec:
        out.append(
            Scenario(
                name=f"latency={int(lat * 1000)}ms",
                question=f"Stable with {int(lat * 1000)} ms of video latency?",
                sim=_sim(
                    altitude_m=altitude_m,
                    video=VideoLinkParams(pipeline_latency_sec=float(lat), latency_jitter_sec=0.015),
                ),
                stab=base,
                seeds=tuple(seeds),
            )
        )
    return out


def motion_model_comparison(
    *, stab: Optional[StabilizerConfig] = None, altitude_m: float = 0.7, seeds: Sequence[int] = (1, 2, 3)
) -> List[Scenario]:
    """Which optical-flow motion model gives the most usable velocity estimate?

    They differ by an order of magnitude in how much altitude change they let
    leak into apparent lateral translation -- see ``flow_sensor.FLOW_ERROR_MODELS``
    -- and altitude change is continuous on a barometric hover.
    """
    base = stab or default_stabilizer_config()
    out: List[Scenario] = []
    for model in ("affine_translation", "translation_rotation", "derotation"):
        out.append(
            Scenario(
                name=f"model={model}",
                question=f"How well does the loop fly on {model}?",
                sim=_sim(altitude_m=altitude_m),
                stab=replace(base, flow_motion_model=model),
                seeds=tuple(seeds),
            )
        )
    return out


def robustness_suite(
    *, stab: Optional[StabilizerConfig] = None, seeds: Sequence[int] = (1, 2, 3)
) -> List[Scenario]:
    """Cross the unmeasured parameters: lean angle x latency x altitude.

    A configuration that passes here is stable across the whole space of
    plausible drones, not just the one we happened to assume.
    """
    base = stab or default_stabilizer_config()
    out: List[Scenario] = []
    for tilt in (10.0, 20.0, 30.0):
        for lat in (0.08, 0.15, 0.25):
            for alt in (0.4, 0.7, 1.5):
                out.append(
                    Scenario(
                        name=f"tilt{int(tilt)}_lat{int(lat * 1000)}_alt{alt:g}",
                        question=f"tilt={tilt:g} deg, latency={int(lat * 1000)} ms, altitude={alt:g} m",
                        sim=_sim(
                            altitude_m=alt,
                            airframe=AirframeParams(max_tilt_deg=tilt),
                            video=VideoLinkParams(pipeline_latency_sec=lat),
                        ),
                        stab=base,
                        duration_sec=15.0,
                        seeds=tuple(seeds),
                    )
                )
    return out


SUITES: Dict[str, Callable[..., List[Scenario]]] = {
    "preflight": build_scenarios,
    "gain": gain_sweep,
    "latency": latency_sweep,
    "models": motion_model_comparison,
    "robustness": robustness_suite,
}
