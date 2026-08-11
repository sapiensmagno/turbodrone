"""Simulator parameters, and where each number comes from.

A simulator is only worth trusting if you can tell measured values from guesses.
Every field below is tagged in its comment:

  [MEASURED]  derived from recorded sessions, reference records, or the protocol
              code in ``e88/``. Changing it means the sim stops matching reality.
  [ESTIMATED] a physically-reasonable value for a ~90 g toy quad that we have not
              measured on this airframe. Sweep it before trusting any conclusion
              that depends on it.
  [POLICY]    a choice about what to simulate, not a property of the drone.

The [ESTIMATED] fields are exactly the ones ``scenarios.py`` sweeps, so that a
"stable hover" verdict is a statement about a range of plausible drones rather
than about one lucky parameter set.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Tuple


GRAVITY_M_S2: float = 9.80665


@dataclass(frozen=True)
class AirframeParams:
    """Rigid-body and flight-controller behaviour of the E88.

    The E88 runs a self-levelling ("angle mode") inner loop on board. We never
    see rates or motor commands -- a roll stick deflection asks for a *lean
    angle*, and the onboard controller gets there on its own. So the airframe
    model is: stick -> target angle -> second-order angle response -> horizontal
    acceleration from the tilted thrust vector -> drag.
    """

    # --- Stick to lean angle -------------------------------------------------
    # [ESTIMATED] Full stick deflection commands this lean angle. Toy quads in
    # angle mode are usually limited to 15-30 deg. This is the single most
    # important unknown for control gain: it sets the loop gain from command to
    # acceleration (a = g*tan(max_tilt * cmd)).
    max_tilt_deg: float = 20.0

    # [ESTIMATED] Natural frequency / damping of the onboard attitude loop.
    # A small quad reaches a commanded angle in ~0.15-0.35 s. omega_n = 15 rad/s
    # with zeta = 0.8 settles in ~0.25 s.
    attitude_omega_n_rad_s: float = 15.0
    attitude_zeta: float = 0.8
    # [ESTIMATED] Angular rate limit of the inner loop (deg/s).
    max_tilt_rate_deg_s: float = 300.0

    # --- Command path --------------------------------------------------------
    # [MEASURED] e88/config.py: control_interval_sec = 0.03 -> the RC state is
    # re-transmitted at ~33 Hz, and that is the rate at which a new command can
    # actually reach the drone. Commands issued between transmissions are held.
    control_interval_sec: float = 0.03
    # [ESTIMATED] Delay from packet transmission to the drone acting on it.
    # Measured net RTT in session meta.json is 3-6 ms; the rest is the drone's
    # own receive/processing cadence. Small compared to the video latency.
    command_latency_sec: float = 0.02
    # [MEASURED] e88/drone.py::_axis_to_byte -- roll/pitch are quantized to
    # byte = round(128 + 127*cmd), i.e. a resolution of 1/127 in command units.
    # With kp_vx=0.003, a 2.6 px/s error is below one quantization step, so this
    # is not a rounding detail: it sets the smallest correction the drone can
    # be asked to make.
    quantize_commands: bool = True
    # [ESTIMATED] Stick counts around centre that the flight controller ignores.
    # Unknown for the E88. 0 = no deadzone; sweep this, because a deadzone
    # interacts with the PI integrator to produce limit cycles.
    stick_deadzone_counts: int = 0

    # --- Command sign convention --------------------------------------------
    # [POLICY] Which way the drone actually moves for a positive command. This is
    # the convention the autopilot's roll_sign/pitch_sign have to match, and
    # getting it wrong turns the controller into positive feedback. The sim can
    # run all four combinations so you learn the right one on the bench.
    # +1 for roll: positive roll command banks right -> accelerates +x (right).
    # +1 for pitch: positive pitch command noses down -> accelerates +y (forward).
    cmd_roll_to_right: float = 1.0
    cmd_pitch_to_forward: float = 1.0

    # --- Translational drag --------------------------------------------------
    # [ESTIMATED] a_drag = -(k_lin*v + k_quad*|v|*v). With these values a 20 deg
    # lean (3.57 m/s^2) settles at roughly 4 m/s, which is the right order for a
    # light indoor quad.
    drag_linear_1_s: float = 0.6
    drag_quadratic_1_m: float = 0.05

    # --- Vertical axis -------------------------------------------------------
    # [MEASURED] e88/drone.py::_throttle_to_byte maps 0..100 -> 0..255, and the
    # autostabilizer holds base_throttle=50 during hover, so 50 is the neutral
    # "hold altitude" stick by construction of the existing config.
    hover_throttle_pct: float = 50.0
    # [ESTIMATED] Climb rate at full throttle deflection from neutral, and the
    # lag of the altitude-hold loop reaching it.
    max_climb_rate_m_s: float = 1.2
    climb_tau_sec: float = 0.35
    # [ESTIMATED] Altitude-hold is barometric on this class of drone and bobs.
    # Standard deviation and correlation time of that bobbing.
    altitude_hold_sigma_m: float = 0.03
    altitude_hold_tau_sec: float = 2.0
    # [POLICY] Throttle deflection below this (in percent) counts as "hold".
    throttle_deadband_pct: float = 2.0
    # [ESTIMATED] Altitude the one-shot takeoff command climbs to before handing
    # control back to the throttle stick. Toy quads settle around 1 m.
    takeoff_altitude_m: float = 1.0
    # [ESTIMATED] Proportional gain of the onboard altitude capture during an
    # auto takeoff or landing, in (m/s) per m of error.
    auto_altitude_kp_1_s: float = 1.5

    # --- Vibration -----------------------------------------------------------
    # Motor and prop imbalance shakes the airframe, and the camera rides on it.
    # A tiny wobble matters a lot: at f = 365.6 px, 0.003 deg between frames is
    # 365.6 * 5.2e-5 = 0.019 px, and at 20 Hz that is already ~0.4 px/s of
    # apparent velocity.
    #
    # Vibration is applied to the *camera's* attitude only, not to the thrust
    # direction: it is a fast oscillation whose net effect on translational
    # acceleration averages to zero over a control period, but whose effect on
    # the image does not average out, because each frame is a snapshot.
    #
    # [CALIBRATED, with a caveat] Set so the simulator's stationary flow noise
    # reproduces the one real anchor available: calibration.json measured a
    # 99th-percentile stationary flow magnitude of 2.46 px/s. The rendered image
    # alone yields only 0.6 px/s, so ~2 px/s of the real figure comes from
    # something the image model does not capture, and this term stands in for
    # all of it -- true airframe wobble, auto-exposure flicker, H.264 temporal
    # artifacts. It cannot be decomposed from the available data.
    #
    # The caveat: that calibration was taken with the drone *still*, most likely
    # with motors off. A hovering drone vibrates more, so this value may be an
    # underestimate for flight. The `noisy_camera` scenario runs at 5x to check
    # the conclusion survives if it is.
    vibration_sigma_deg: float = 0.003
    # [ESTIMATED] Correlation time. Prop frequency is far above the 20 Hz frame
    # rate, so successive frames see essentially uncorrelated wobble.
    vibration_tau_sec: float = 0.01

    # --- Yaw -----------------------------------------------------------------
    # [ESTIMATED] Full yaw stick rate, and the slow heading drift a toy quad
    # shows with the yaw stick centred. Yaw drift matters because it rotates the
    # camera, which the flow model has to cope with.
    max_yaw_rate_deg_s: float = 120.0
    yaw_drift_deg_s: float = 2.0

    def max_tilt_rad(self) -> float:
        import math

        return math.radians(float(self.max_tilt_deg))


@dataclass(frozen=True)
class EnvironmentParams:
    """Disturbances acting on the airframe."""

    # [ESTIMATED] Constant acceleration bias, i.e. an untrimmed drone. Every toy
    # quad has some. This is what the PI integrator exists to cancel, so a hover
    # test with zero trim bias proves almost nothing.
    trim_accel_x_m_s2: float = 0.05
    trim_accel_y_m_s2: float = 0.03

    # [ESTIMATED] Turbulence as an Ornstein-Uhlenbeck acceleration: sigma is the
    # steady-state standard deviation, tau its correlation time. Indoors this is
    # mostly the drone's own downwash bouncing off the floor.
    gust_sigma_m_s2: float = 0.12
    gust_tau_sec: float = 0.8

    # [POLICY] A deterministic gust: constant acceleration over a time window.
    # Used by the wind-gust scenario to probe recovery, not disturbance rejection.
    gust_pulse_accel_m_s2: Tuple[float, float] = (0.0, 0.0)
    gust_pulse_start_sec: float = 0.0
    gust_pulse_duration_sec: float = 0.0

    # [POLICY] Initial condition.
    initial_altitude_m: float = 0.7
    initial_xy_m: Tuple[float, float] = (0.0, 0.0)
    initial_velocity_m_s: Tuple[float, float] = (0.0, 0.0)
    initial_yaw_deg: float = 0.0


@dataclass(frozen=True)
class CameraParams:
    """Downward camera geometry and image formation.

    Resolution and focal length are pinned to reality:
    ``optical_flow.DerotationMotionModel`` documents the real downscaled tracking
    geometry as 320x240 with f=183, and the estimator downscales by 0.5, so the
    delivered frames are 640x480 with f~366. Independently, reference record
    ``faf3adb3`` records a 0.30 m pad measuring 156.7 px at 0.70 m, which gives
    f = 156.7 * 0.70 / 0.30 = 365.6 px -- the value recorded as
    ``flow_focal_length_px`` in every recent session meta.json.
    """

    # [MEASURED]
    width: int = 640
    height: int = 480
    fx: float = 365.6463394165039
    fy: float = 365.6463394165039
    # [ESTIMATED] The principal point has never been calibrated on this camera;
    # the autopilot assumes the image midpoint. Offsetting it here is how you
    # find out how much that assumption costs (it biases derotation).
    cx: Optional[float] = None  # None -> (width - 1) / 2
    cy: Optional[float] = None  # None -> (height - 1) / 2

    # [ESTIMATED] Lens distortion. Never calibrated on this camera. Default zero
    # so nominal runs are clean; the distortion scenario turns it on, because
    # barrel distortion is a radial-quadratic perturbation and therefore
    # corrupts exactly the (1 + u'^2) term the derotation model uses to tell
    # tilt from translation.
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0

    # [MEASURED, sign convention] The camera is mounted rotated relative to the
    # airframe: the autopilot compensates with a fixed 90 deg remap (roll is
    # driven from vy, pitch from vx -- see README 4.4.2). This is that rotation,
    # about the optical axis, in degrees.
    mount_yaw_deg: float = 90.0

    # [ESTIMATED] Exposure time. Drives motion blur: a 0.5 m/s translation at
    # 0.7 m altitude moves 261 px/s, so an 8 ms exposure smears ~2 px.
    exposure_sec: float = 0.008
    # [CALIBRATED] Additive sensor noise, in 8-bit counts. Above ~2 counts the
    # noise itself starts generating corners: goodFeaturesToTrack returns its
    # full 250-corner budget of noise maxima, which is nothing like the recorded
    # n_features of 37-84. 1.5 keeps the corner field texture-driven.
    sensor_noise_sigma: float = 1.5
    # [CALIBRATED] The real stream is H.264, whose inter-frame prediction leaves
    # static regions nearly identical between frames. Independent per-frame JPEG
    # is harsher than that, so the quality is set high (85) to avoid inventing
    # frame-to-frame noise the real codec does not produce. None disables it.
    jpeg_quality: Optional[int] = 85
    # [ESTIMATED] Vignetting strength, 0 = none, 1 = corners fully dark.
    vignette: float = 0.15
    # [POLICY] Overall brightness gain applied to the rendered ground.
    brightness_gain: float = 1.0
    # [POLICY] Sub-exposure samples averaged to produce motion blur. 1 disables
    # it. 3 is enough to smear the few-pixel motions seen at hover, and it gets
    # rotation-induced blur right (which a single directional kernel would not).
    blur_samples: int = 3


@dataclass(frozen=True)
class VideoLinkParams:
    """How rendered frames reach the control loop.

    Recorded sessions show frame_rate_hz ~19.8-19.9 and flow dt ~0.050 s, so the
    stream runs at 20 fps. ``frame_age_ms`` -- decode time to flow start --
    averages 33-52 ms with a p95 of 64-100 ms across the 2026-02-08 sessions.

    The important caveat: ``frame_age_ms`` is measured from the moment OpenCV
    *decoded* the frame, so it excludes camera exposure, onboard encoding, WiFi
    transport and decoder buffering. The true photon-to-command latency is
    larger and has never been measured on this drone. ``pipeline_latency_sec``
    models that unmeasured part, and it is the parameter most likely to decide
    whether the real hover is stable -- which is why the latency scenario sweeps
    it rather than trusting the default.
    """

    # [MEASURED]
    fps: float = 20.0
    # [MEASURED] Frame-to-frame timing jitter (std dev, seconds).
    fps_jitter_sec: float = 0.004
    # [ESTIMATED] Photon-to-decoded-frame latency: exposure, onboard encode,
    # WiFi, jitter buffer, decode. NOT measured. 0.12 s is typical for RTSP over
    # UDP on this class of camera.
    pipeline_latency_sec: float = 0.12
    # [ESTIMATED] Jitter on that latency (std dev, seconds).
    latency_jitter_sec: float = 0.015
    # [MEASURED] Recorded sessions lose 1-8 frames out of 250-800, i.e. well
    # under 1%, in the size-1 buffer. This is the *link* drop rate (frames that
    # never arrive), kept small to match.
    drop_probability: float = 0.005
    # [MEASURED, qualitatively] README 8.2: RTSP over UDP produces torn frames
    # where two halves show different scene slices, and optical flow reads that
    # as a huge jump. Probability per frame.
    tear_probability: float = 0.004
    # [POLICY] Freeze the feed entirely for this long, starting at this time.
    # Exercises max_stale_frame_hold_sec.
    freeze_start_sec: float = 0.0
    freeze_duration_sec: float = 0.0


@dataclass(frozen=True)
class GroundParams:
    """The scene the camera looks at.

    Texture density is what optical flow actually lives on. Recorded sessions
    report n_features 37-84 and quality 0.83-0.91, so a realistic ground has to
    produce features in that range at 0.7 m -- not a checkerboard that tracks
    perfectly, and not a blank floor that tracks not at all.
    """

    # [POLICY] Ground texture resolution, metres per texture pixel. At 0.7 m the
    # camera's ground sampling distance is 0.7/365.6 = 1.91 mm/px, so a 1.5 mm
    # texture is still finer than the sensor can resolve: the render samples the
    # texture *down*, never up, and no synthetic sharpness is invented.
    m_per_px: float = 0.0015
    # [POLICY] Extent of the generated ground patch, metres. At 2 m altitude the
    # camera sees 640 * 2 / 365.6 = 3.5 m across, so 6 m leaves room to drift.
    extent_m: float = 6.0
    # [CALIBRATED] Speckles per square metre. This is the knob that sets
    # ``n_features``: the camera sees 1.13 m^2 at 0.7 m, so 55/m^2 puts ~62
    # trackable blobs in frame, inside the recorded 37-84 range.
    speckle_density_per_m2: float = 55.0
    # [CALIBRATED] Physical size of a speckle. At 0.7 m, 6 mm is ~3 image px --
    # big enough to survive downsampling to the 320x240 tracking resolution.
    speckle_size_m: float = 0.006
    # [POLICY] Contrast of the speckles, 0..1. Lowering this is what the
    # low-texture scenario does to starve the tracker.
    speckle_contrast: float = 0.85
    # [CALIBRATED] Contrast of the broadband background (grain, weave). Kept low
    # deliberately: a strong fractal background makes goodFeaturesToTrack return
    # its full 250-corner budget of weak, self-similar corners, which is both
    # unlike the recorded n_features and unrealistically noisy to track.
    background_contrast: float = 0.10

    # --- Non-planarity -------------------------------------------------------
    # Real floors are not planes. Cables, rug edges, toys and furniture feet sit
    # above the floor, and their parallax is what actually produces the outlier
    # tracks behind the recorded inlier_ratio of ~0.89. Without this the sim
    # tracks at inlier_ratio 1.00 and flatters the estimator.
    # [ESTIMATED] Fraction of the ground covered by raised objects, and how far
    # above the floor they sit.
    relief_fraction: float = 0.15
    relief_height_m: float = 0.05
    # [POLICY] Typical footprint of one raised object.
    relief_patch_size_m: float = 0.08
    # [POLICY] Optional pad composited onto the floor: image path, centre in
    # world metres, physical size in metres. Enables the visual-scale path.
    pad_image_path: Optional[str] = None
    pad_center_m: Tuple[float, float] = (0.0, 0.0)
    pad_size_m: Tuple[float, float] = (0.3, 0.3)
    seed: int = 0


@dataclass(frozen=True)
class SimConfig:
    airframe: AirframeParams = field(default_factory=AirframeParams)
    environment: EnvironmentParams = field(default_factory=EnvironmentParams)
    camera: CameraParams = field(default_factory=CameraParams)
    video: VideoLinkParams = field(default_factory=VideoLinkParams)
    ground: GroundParams = field(default_factory=GroundParams)

    # [POLICY] Physics integration step. Must be well below the 0.03 s command
    # interval and the attitude time constant.
    physics_dt_sec: float = 0.002
    seed: int = 0

    def with_(self, **kwargs) -> "SimConfig":
        """Shallow override helper: ``cfg.with_(seed=3)``."""
        return replace(self, **kwargs)
