# E88 Autopilot (Experimental)

This folder contains an **experimental** “autostabilizer” for the E88-class drone.

The goal is simple:

- Keep the drone visually stable (hovering in place) using only the **onboard camera stream**.
- Estimate the drone’s planar motion (image-space drift) using **optical flow**.
- Smooth that motion estimate using a **Kalman filter**.
- Convert the estimated drift into small **roll/pitch compensation commands**.
- Provide real-time **telemetry** for a Qt UI (optical-flow overlay + 2D motion/compensation plot).

**Phase 0 status (implemented): observability-first**

Phase 0 focuses on making the estimator and control loop inspectable:

- A **session recorder** writes `meta.json` + `samples.jsonl` for every run.
- Rich **per-iteration timing** (flow time, total loop time, estimated latency).
- **Frame acquisition is decoupled** from the control loop via a background thread and a size-1 “latest frame” buffer.
- **Frame staleness** and **dropped/overwritten frames** are surfaced in telemetry.
- A Qt UI shows diagnostics, an optical-flow overlay, and a trajectory widget.
- Stationary **calibration** estimates `estimator_deadband_px_s` and `kalman_sigma_v` and persists them.

---

## Index

- [1. Quick start](#1-quick-start)
- [2. What “stabilization” means here](#2-what-stabilization-means-here)
- [3. Core pipeline overview](#3-core-pipeline-overview)
- [4. Concepts primer](#4-concepts-primer)
  - [4.1 Optical flow (Lucas–Kanade) introduction](#41-optical-flow-lucas–kanade-in-plain-language)
  - [4.2 Robust motion via RANSAC](#42-robust-motion-via-ransac)
  - [4.3 Kalman filtering introduction](#43-kalman-filtering-in-plain-language)
  - [4.4 Controller: turning drift into roll/pitch](#44-controller-turning-drift-into-rollpitch)
- [5. Software architecture](#5-software-architecture)
- [6. Configuration reference (all parameters)](#6-configuration-reference-all-parameters)
  - [6.1 Takeoff / climb / settle phase parameters](#61-takeoff--climb--settle-phase-parameters)
  - [6.2 Control loop parameters](#62-control-loop-parameters)
  - [6.3 Estimation parameters (Kalman)](#63-estimation-parameters-kalman)
  - [6.4 Motion-quality / safety gates](#64-motion-quality--safety-gates)
  - [6.5 PID-like controller parameters](#65-pid-like-controller-parameters)
  - [6.6 Sign conventions (roll_sign, pitch_sign)](#66-sign-conventions-roll_sign-pitch_sign)
  - [6.7 Visual scale (reference detection)](#67-visual-scale-reference-detection)
- [7. Telemetry and visualization](#7-telemetry-and-visualization)
- [7.1 Understanding the key metrics (recommended reading)](#71-understanding-the-key-metrics-recommended-reading)
- [8. Known challenges and mitigations](#8-known-challenges-and-mitigations)
  - [8.1 Camera orientation and axis confusion](#81-camera-orientation-and-axis-confusion)
  - [8.2 “Torn / split frames” (RTSP/UDP artifacts)](#82-torn--split-frames-rtspudp-artifacts)
  - [8.3 Lighting and low-texture scenes](#83-lighting-and-low-texture-scenes)
  - [8.4 Latency, timing, and control rate](#84-latency-timing-and-control-rate)
  - [8.5 Stationary drift (“ghost motion”) from integrating noisy flow](#85-stationary-drift-ghost-motion-from-integrating-noisy-flow)
  - [8.6 Conditioning before Kalman: deadband + gating telemetry](#86-conditioning-before-kalman-deadband--gating-telemetry)
  - [8.7 Stationary calibration and persistence](#87-stationary-calibration-and-persistence)
  - [8.8 Optical flow scaling with downscale](#88-optical-flow-scaling-with-downscale)
  - [8.9 Robust fallback for large displacements (phase correlation)](#89-robust-fallback-for-large-displacements-phase-correlation)
- [9. How to tune safely](#9-how-to-tune-safely)
- [10. File guide](#10-file-guide)

---

## 1. Quick start

### 1.1 Run from the Qt UI (recommended)

From the repo root:

```bash
python -m e88_qt.app
```

In the UI:

- Chose camera. Camera 2 is facing down so it is preferred. Switch between them to reset the stream if you experience video issues. They may happen due to RTSP over UDP (packet loss/jitter)
- Configure parameters in the **Autostabilizer** section.
- In **Visual Scale**:
  - Select a saved Reference (or Capture one).
  - Choose **Detection** method if you're not using ArUco markers:
    - `ORB + Contours` (default)
    - `Closed Quad`
  - Click **Calibrate altitude** for the selected Reference.
- Click **Start autostabilizer**.
- **Emergency land**: click **EMERGENCY LAND** (or press `Esc`) to stop the autostabilizer and then request a land command to the drone.

### 1.2 Run from CLI (headless-ish)

From the repo root:

```bash
python -m e88_autopilot.run_autostabilizer --protocol E88
```

You can optionally set:

- `--camera 1`
- `--enable-takeoff`
- `--duration <seconds>`

---

## 2. What “stabilization” means here

This stabilizer is *not* using IMU state estimation, GPS, or SLAM.

Instead it stabilizes **visual drift**:

- If the camera view appears to slide left/right/up/down between frames, we interpret that as drift.
- We command small roll/pitch corrections in the opposite direction to reduce that drift.

Important practical implications:

- This approach works best when the camera sees a **static environment** with enough texture (corners/edges).
- It estimates motion in **image space (pixels)**; mapping to real-world meters is not part of this experimental module.

---

## 3. Core pipeline overview

At a high level, every control step does:

1. **Acquire a new video frame** from the drone.
   - Frame acquisition runs in a background thread and the control loop consumes the latest frame without blocking.
2. **Track feature points** across consecutive frames (Lucas–Kanade optical flow).
3. **Fit a global motion model** to the tracked point displacements.
   - Default is `flow_motion_model="translation_rotation"`, which estimates translation + in-plane rotation.
   - A simpler `"affine_translation"` option is available.
   - The fit is done robustly (RANSAC / residual gating) to reduce the impact of outlier tracks.
4. Convert the fitted motion into drift velocity `(vx, vy)` in pixels/second.
5. Optionally estimate **visual scale** (meters-per-pixel) and convert `(vx, vy)` to m/s.
   - When enabled and stable, we compute `vx_m_s`, `vy_m_s` from `(vx, vy)`.
   - When stability is lost, the system can optionally keep using the last stable scale for a short hold window.
6. Apply measurement conditioning (deadband/gating), then optionally run a **Kalman filter**.
   - When enabled (`use_kalman=True`), the controller uses Kalman-smoothed `vx/vy` and the Kalman `x/y` state provides the primary position estimate.
7. Choose the active controller:
   - Default: velocity hold (drive measured velocity toward `0`).
   - Optional: position leash (outer-loop position hold that produces a velocity setpoint to pull back toward the start point).
8. If m/s control is enabled and scale is stable (or held), feed controllers with m/s; otherwise use px/s.
9. Feed the selected controller and compute `(roll_cmd, pitch_cmd)`.
9. Send `(roll_cmd, pitch_cmd, throttle)` to the drone.
10. Emit **telemetry** (and optionally record `samples.jsonl`) so the UI can display and you can evaluate:
   - the camera frame,
   - flow tracks,
   - fitted flow model quality and rotation estimate,
   - estimated velocity/position (px and optionally m/s; raw and/or Kalman),
   - command outputs,
   - leash state (when enabled).

---

## 4. Concepts primer

### 4.1 Optical flow (Lucas–Kanade) introduction

**Optical flow** measures how pixels move from one frame to the next.

Lucas–Kanade (LK) is a classic method:

- We pick a set of “good” points to track (corners are ideal).
- For each point, LK finds where that point likely moved in the next frame.

In this project:

- Feature detection uses `cv2.goodFeaturesToTrack`.
- Tracking uses `cv2.calcOpticalFlowPyrLK`.
- We then try to summarize many point motions into one global drift estimate.

Why feature tracking instead of comparing whole frames?

- It’s lighter weight.
- It often works well in indoor scenes.
- It’s easy to visualize (you can draw the track vectors).

### 4.2 Robust motion via RANSAC

Real video is messy:

- Some tracked points jump due to blur.
- Some points belong to moving objects (e.g., your hand).
- Some points are tracked incorrectly.

If we simply average all displacements, outliers can dominate.

**RANSAC** (Random Sample Consensus) is a technique that:

- repeatedly fits a model to random subsets of points,
- measures which points agree with that model (“inliers”),
- chooses the model with the best support.

Here we use:

- `cv2.estimateAffinePartial2D(..., method=cv2.RANSAC)`

and extract only the translation component as `(dx, dy)`.

### 4.3 Kalman filtering introduction

Even with RANSAC, `vx/vy` from optical flow is noisy.

A **Kalman filter** is a lightweight probabilistic estimator that:

- assumes a simple motion model (e.g., “velocity changes smoothly”),
- combines a noisy measurement (optical flow) with that model,
- outputs a smoothed estimate (and confidence).

In this project:

- We use a 2D velocity Kalman filter (`VelocityKalman2D`).
- It maintains a state that includes **position and velocity** in pixels.

When enabled (`use_kalman=True`):

- The controller receives **Kalman-smoothed velocities**.
- The plotted trajectory uses **Kalman position** (`x_px`, `y_px`).

When disabled (`use_kalman=False`):

- The controller uses raw `vx/vy`.
- The plotted position is simple integration: `pos += v * dt`.

### 4.4 Controller: turning drift into roll/pitch

This project controls **image-plane drift**.

#### 4.4.1 Coordinate conventions (what `vx`/`vy` actually mean)

The optical flow layer (`optical_flow.py`) estimates a translation `(dx_px, dy_px)` between consecutive frames:

- `dx_px > 0`: tracked features moved **to the right** in the raw camera frame.
- `dy_px > 0`: tracked features moved **down** in the raw camera frame.

Then:

- `vx_px_s = dx_px / dt_sec`
- `vy_px_s = dy_px / dt_sec`

Important implications:

- These are in **raw camera pixel coordinates** (OpenCV convention), not “world forward/right”.
- The Qt video preview is rotated for display, but the stabilizer computes flow on the raw frame.
- The trajectory widget flips the vertical axis for display and also swaps the plotted axes so the chart is more intuitive (right/left on X, front/back on Y).
  - This is a visualization-only convenience; the underlying `vx`/`vy` values are still raw OpenCV pixel drift.

#### 4.4.2 Mapping from drift velocity to roll/pitch commands

The mapping is implemented in `controller.py` (`VelocityHoldController`) and wired in `autostabilizer.py`.
Because the E88’s downward-facing camera is effectively rotated by ~90° relative to the intuitive “forward/right” axes,
the stabilizer applies a fixed 90° remap when feeding the controller:

- **Roll is driven from measured `vy`**
- **Pitch is driven from measured `vx`**

Specifically (PI controller per axis):

- Apply deadband:
  - `vx = 0` if `abs(vx_px_s) < deadband`
  - `vy = 0` if `abs(vy_px_s) < deadband`
- Integrate (with clamp):
  - `ivx = clamp(ivx + vx * dt, -integrator_limit, +integrator_limit)`
  - `ivy = clamp(ivy + vy * dt, -integrator_limit, +integrator_limit)`
- PI “raw” commands:
  - `u_roll  = kp_vx * vy + ki_vx * ∫ vy dt`
  - `u_pitch = kp_vy * vx + ki_vy * ∫ vx dt`
- Apply configurable sign flips (to handle camera/drone orientation differences):
  - `roll  = clamp(roll_sign  * u_roll,  -max_cmd, +max_cmd)`
  - `pitch = clamp(pitch_sign * u_pitch, -max_cmd, +max_cmd)`

The sign flips `roll_sign` / `pitch_sign` still apply after this remap, and exist to handle remaining direction inversions
from camera mounting, headless mode, or protocol conventions.

#### 4.4.3 When and for how long commands are applied

Commands are recomputed and sent continuously in the main loop (see `autostabilizer.py`) at approximately:

- `cmd_rate_hz` (default `20 Hz`)

How long compensation persists:

- The PI controller runs every loop and outputs the current roll/pitch.
- The **integrator has memory**: even if `vx`/`vy` later fall into the deadband (become 0), the integrator terms (`ivx`/`ivy`) keep their current value and can keep producing some command.
- The integrators are reset only when:
  - `quality < min_quality` (controller returns `None` and calls `reset()`), or
  - the stabilizer is (re)activated (`activate()` resets controllers).

So you should expect commands to be applied continuously while drift is present, and to “linger” according to the integral term unless quality drops or you reset.

---

## 5. Software architecture

### 5.1 Modules

- `autostabilizer.py`
  - Orchestrates the stabilization loop.
  - Owns the phase logic (optional takeoff/climb/settle, then hold loop).
  - Emits `StabilizerTelemetry`.

- `optical_flow.py`
  - Implements `LucasKanadeDriftEstimator`.
  - Produces `FlowEstimate` and track data (`FlowTracks`) for visualization.
  - Applies robustness (RANSAC + outlier/corruption rejection).

- `kalman.py`
  - Implements a Kalman filter that smooths the velocity estimates.

- `controller.py`
  - Implements `VelocityHoldController` mapping velocity -> roll/pitch.

- `run_autostabilizer.py`
  - CLI entrypoint that builds a `StabilizerConfig` from flags.

### 5.2 Design decisions

- **Telemetry sink callback**
  - The stabilizer has an optional `telemetry_sink` callback.
  - This keeps UI code out of the autopilot loop and reduces coupling.

- **Threading model (Qt)**
  - The stabilizer runs in a background `QThread`.
  - The UI periodically polls the latest telemetry and updates widgets.
  - This keeps the UI responsive without sharing complex objects across threads.

- **Visualization-first workflow**
  - The system emits frame + tracks + estimates explicitly, so you can debug
    when the estimator is wrong (e.g., torn frames, low texture).

---

## 6. Configuration reference (all parameters)

The central configuration object is `StabilizerConfig` in `autostabilizer.py`.

Values are shown with their **current defaults**.

### 6.1 Takeoff / climb / settle phase parameters

These phases run *before* the main hold loop if `enable_takeoff=True`.

- `enable_takeoff: bool = False`
  - Enables a scripted takeoff+climb prelude.

- `takeoff_throttle: float = 100.0`
  - Throttle command during the takeoff phase.

- `takeoff_duration_sec: float = 2.0`
  - Duration to keep sending takeoff throttle.

- `climb_throttle: float = 70.0`
  - Throttle after takeoff.

- `climb_duration_sec: float = 1.5`
  - Duration of climb phase.

- `settle_good_frames: int = 5`
  - During settle, we wait until we see N consecutive “good enough” frames
    (based on flow quality) before entering hold.

### 6.2 Control loop parameters

- `base_throttle: float = 50.0`
  - Throttle during hold phase.

- `cmd_rate_hz: float = 20.0`
  - How often to send commands.
  - Higher values respond faster but can amplify noise.

### 6.3 Estimation parameters (Kalman)

- `use_kalman: bool = True`
  - Enables Kalman smoothing.

- `kalman_sigma_a: float = 25.0`
  - Process noise related to acceleration.
  - Larger values allow velocity to change more rapidly.

- `kalman_sigma_v: float = 460.0`
  - Measurement noise for velocity.
  - Larger values trust measurements less and smooth more.

### 6.4 Motion-quality / safety gates

- `min_quality: float = 0.15`
  - Minimum quality required to produce a control command.
  - If quality drops below this, controller output is disabled (and integrator reset).

### 6.4.1 Optical flow motion model (optional rotation compensation)

The optical flow layer estimates a per-frame translation `(dx_px, dy_px)` from many tracked feature point motions.

Initially, it used an affine model and took only the translation terms. This would work well for pure translation but can be biased by small camera rotations (e.g., roll/pitch “inclination” while hovering) which may be misinterpreted as opposite-sign translation.

The stabilizer now supports an optional motion model that explicitly fits translation + in-plane rotation:

- `flow_motion_model: str = "translation_rotation"`
  - Options:
    - `"affine_translation"`: legacy behavior (use affine fit translation terms).
    - `"translation_rotation"`: estimate translation and rotation and use only the translation components.

- `flow_tr_residual_thresh_px: float = 3.0`
  - Used only when `flow_motion_model="translation_rotation"`.
  - Per-track residual threshold (in px) for a second-pass refit. Set to `0.0` to disable residual filtering.

- `flow_tr_min_points: int = 20`
  - Used only when `flow_motion_model="translation_rotation"`.
  - Minimum number of tracked points required to run the translation+rotation model.

**Practical tuning guidance**

These are robustness knobs (not calibration constants). Tune them based on how stable the tracking is in your environment:

- If you see noisy `omega` or frequent sign flips:
  - increase `flow_tr_min_points` (e.g., `20 -> 40` or `60`) to require a stronger support set.

- If `translation_rotation` almost never activates (it frequently fails to fit):
  - decrease `flow_tr_min_points` (but expect more noise), and/or
  - increase `flow_tr_residual_thresh_px` slightly.

- If `translation_rotation` activates but seems to overreact (bad correction, high `model_rmse_px`):
  - decrease `flow_tr_residual_thresh_px` (e.g., `3.0 -> 2.0` or `1.5`) to reject inconsistent tracks.

What to watch while tuning:

- `motion_model` (how often it chooses `translation_rotation`)
- `model_rmse_px` (should be low and stable when the model fits)
- `n_tracked` / `inlier_ratio` / `quality` (should not be collapsing)

Additional flow telemetry fields (in `samples.jsonl` under `flow`) help A/B testing:

- `motion_model`
  - Which estimator path was used for this frame.

- `raw_dx_px`, `raw_dy_px`
  - The translation returned by the legacy affine/median path before any translation+rotation correction.

- `omega_rad`, `omega_rad_s`
  - Estimated in-plane rotation per frame and per second.
  - These are expressed in raw OpenCV image coordinates (x right, y down), so interpret sign accordingly.

- `model_rmse_px`
  - RMSE of the translation+rotation model fit (lower is better).

**How `model_rmse_px` is computed**

When `flow_motion_model="translation_rotation"` is enabled, we fit a small-motion model to all tracked point displacements:

- unknowns: translation `(tx, ty)` and in-plane rotation `omega` (radians per frame)
- for each tracked point we predict its displacement `(dx, dy)` from `(tx, ty, omega)`

For each point, we compute the residual magnitude:

- `r_i = sqrt((dx_i - dx_pred_i)^2 + (dy_i - dy_pred_i)^2)`

Then:

- `model_rmse_px = sqrt(mean(r_i^2))`

This is computed over the points used in the (final) fit.

**How to interpret high `model_rmse_px` (what’s “wrong”)**

High RMSE means the observed flow does not look like “translation + a single global in-plane rotation”. Common causes:

- Non-planar scene / parallax (close objects, strong depth variation)
- Moving objects in the scene
- LK tracking failures / bad matches
- Corrupted frames (torn frames, compression artifacts)
- Strong lens distortion (the simple model is less accurate near the edges)

Practical debugging signals in `samples.jsonl`:

- If `model_rmse_px` spikes while `n_tracked` drops, it often indicates tracking collapse.
- If `model_rmse_px` spikes while `n_tracked` stays high and the frame looks “busy”, it often indicates parallax/moving objects.

**Why RMSE is not a perfect “source of truth”**

RMSE is a self-consistency score for a particular model. It does not guarantee the estimated translation is correct.

Two important limitations:

- A wrong model can still fit well: if the true motion is not translation+rotation (e.g., perspective effects), the solver can sometimes explain it with a plausible `(tx, ty, omega)` and still produce a low RMSE.
- RMSE is scene-dependent: low-texture scenes can yield low RMSE on a small set of points even when the estimate is biased.

So RMSE is best treated as a *diagnostic* and an *additional gate*, not a ground-truth label.

**How to use RMSE as an additional gate (recommended)**

If you find cases where `translation_rotation` produces worse commands, you can tighten acceptance by gating on RMSE.

For example, accept translation+rotation only if:

- `model_rmse_px <= threshold_px`
- and `n_used >= flow_tr_min_points`

Otherwise fall back to the legacy affine translation estimate.

**How `quality` is defined**

`LucasKanadeDriftEstimator` defines:

- `tracked_ratio = n_tracked / max(1, n_features)`
- `inlier_ratio = n_inliers / max(1, n_tracked)`
- `quality = clamp(tracked_ratio * inlier_ratio, 0..1)`

Interpretation:

- `quality` measures “how much of the frame’s feature tracking is usable”.
- A `min_quality` of `0.15` means we accept updates where the estimator thinks at least ~15% of the potential feature evidence is consistent.
  - Example: `tracked_ratio=0.5` and `inlier_ratio=0.3` gives `quality=0.15`.
  - If either tracking collapses (few tracked points) or RANSAC consistency collapses (few inliers), `quality` drops.

In `LucasKanadeDriftEstimator` (optical flow) there are additional *internal* robustness gates
that protect against corrupted frames (e.g., torn frames):

- `max_translation_frac: float = 0.25` (internal default)
  - If estimated per-frame translation exceeds this fraction of the frame size,
    the frame is treated as corrupted and the tracker resets.

**How `max_translation_frac` is computed and used**

In `LucasKanadeDriftEstimator.update()`:

- The estimator works on a (possibly downscaled) grayscale frame `gray`.
- It computes:
  - `max_step_px = max_translation_frac * min(gray.height, gray.width)`
- If `abs(dx_px) > max_step_px` or `abs(dy_px) > max_step_px`, the estimate is rejected.

This is meant as a “corrupted frame / torn frame / huge jump” guardrail.
It is not a physical limit of the drone.

- `min_inlier_ratio: float = 0.6` (internal default)
  - If RANSAC inlier ratio drops below this, the frame is treated as corrupted
    and the tracker resets.

These are not currently exposed in `StabilizerConfig`.

### 6.5 PID-like controller parameters

These live in `VelocityHoldController` and are surfaced through `StabilizerConfig`:

- `max_cmd: float = 0.8`
  - Clamp for absolute roll/pitch output.

**What `max_cmd` means**

The controller output is a normalized command in the range `[-1.0, +1.0]`.
`max_cmd` clamps roll and pitch to `[-max_cmd, +max_cmd]`.

So with `max_cmd=0.8`, the controller can command up to ~80% of full stick deflection on roll/pitch.
This is not “80% of force” in a physical sense, but it is a large command.

Practical guidance:

- Higher `max_cmd` increases authority (can correct faster) but also increases risk of oscillation and aggressive motion.

- `kp_vx: float = 0.003`
  - Proportional gain for x velocity.

- `kp_vy: float = 0.003`
  - Proportional gain for y velocity.

- `ki_vx: float = 0.0005`
  - Integral gain for x velocity.

- `ki_vy: float = 0.0005`
  - Integral gain for y velocity.

- `deadband_px_s: float = 3.0`
  - Velocities below this magnitude are treated as 0.

**What `kp_*` and `ki_*` do**

The controller’s job is to turn an observed drift velocity (`vx`, `vy` in px/s) into a correcting roll/pitch command.

It uses two terms:

- **Proportional (P)**: “react to what is happening right now”
  - `u_p = kp * v`
  - If drift velocity doubles, the correction doubles.
  - Too much P causes oscillation/jitter; too little P feels lazy and won’t correct.

- **Integral (I)**: “react to what has been happening for a while”
  - The controller accumulates velocity over time: `iv += v * dt` (clamped by an internal limit).
  - Then adds `u_i = ki * iv`.
  - I helps cancel steady, persistent drift that P alone doesn’t remove.
  - Too much I causes slow oscillations and can “wind up” (integrator grows while the drone can’t respond).

Tuning workflow (safe):

1. Start with `ki_* = 0` (or very small), tune `kp_*` until drift is reduced.
2. Add a small `ki_*` only if you still see steady drift.
3. If you see oscillation, first reduce `max_cmd`, then reduce `kp_*`, then reduce `ki_*`.

### 6.5.1 Position leash (optional position hold)

The default controller is a velocity hold: it tries to drive measured velocity toward `0`.
This stops drift *rate* but does not necessarily return the drone to its original position.

The optional **position leash** adds an outer-loop that estimates displacement and generates a velocity setpoint that “pulls back” toward the start point.
The existing velocity controller remains the inner loop.

- `enable_position_leash: bool = False`
  - Enables the position leash wrapper.

There are two parameter sets:

- *px-domain* leash (used when m/s control is not available):
  - `leash_kp_pos_px: float = 0.02`
  - `leash_ki_pos_px: float = 0.0`
  - `leash_pos_deadband_px: float = 1.0`
  - `leash_max_v_sp_px_s: float = 10.0`
  - `leash_pos_integrator_limit_px_s: float = 1000.0`

- *m-domain* leash (used when visual scale is stable and m/s control is enabled):
  - `leash_kp_pos_m: float = 0.5`
  - `leash_ki_pos_m: float = 0.0`
  - `leash_pos_deadband_m: float = 0.02`
  - `leash_max_v_sp_m_s: float = 0.25`
  - `leash_pos_integrator_limit_m_s: float = 10.0`

**How it works (conceptually)**

- Use a displacement estimate `pos`:
  - If Kalman is enabled, `pos` comes from the Kalman `x/y` state (preferred).
  - Otherwise, `pos` is obtained by integrating measured velocity: `pos += v * dt`.
- Convert displacement to a velocity setpoint: `v_sp = -(kp_pos * pos + ki_pos * ∫pos)`.
- Feed the inner velocity controller the velocity error: `v_err = v_meas - v_sp`.

**Practical tuning guidance**

- Start with `ki_pos=0.0`.
- Increase `leash_kp_pos_*` until it reliably pulls back, then back off if you see oscillation.
- Use `leash_max_v_sp_*` as the primary safety/comfort limiter (how aggressively it is allowed to “pull”).
- Use `leash_pos_deadband_*` to stop tiny jittering near the origin.

**Session logging signals (samples.jsonl)**

When enabled, session samples include:

- `leash_enabled` / `leash_units`
  - Whether leash is enabled and whether the controller is currently in `"px"` or `"m"` units.

- `leash_state`
  - `pos_x`, `pos_y`: the displacement estimate used by the leash (Kalman position when available; otherwise velocity-integrated).
  - `err_x`, `err_y`: deadbanded error used for control.
  - `v_sp_x`, `v_sp_y`: the velocity setpoint generated by the leash.

What to watch:

- `pos_x_px` / `pos_y_px` (and/or Kalman `x_px/y_px`) vs `leash_state.pos_*`
- `cmd_roll` / `cmd_pitch` continuing to command when velocity is near zero but `leash_state.err_*` is not.
- If you see oscillation, reduce `max_cmd` before touching gains.

### 6.6 Sign conventions (roll_sign, pitch_sign)

- `roll_sign: float = -1.0`
- `pitch_sign: float = -1.0`

These exist because “positive image drift” does not necessarily correspond to
“positive command” on a given drone/camera orientation.

If the drone corrects in the wrong direction:

- flip `roll_sign` from `-1` to `+1`, or
- flip `pitch_sign` from `-1` to `+1`.

#### 6.6.1 UI sign verification (Flow sign OK / Control sign OK)

The Qt UI includes two tri-state checkboxes in **Diagnostics**:

- **Flow sign OK**
  - Meaning: you have manually verified that the *reported* drift direction (from optical flow) matches what you expect when you intentionally move the drone.
  - How to use:
    - With the drone on the ground (motors off) or at a safe low altitude, move it slightly:
      - move the drone to the **right** and confirm the drift indicator/trajectory moves in the expected direction,
      - move the drone **forward** and confirm the drift indicator/trajectory moves in the expected direction.
    - If directions look mirrored or swapped, fix the camera/axis/sign settings before trying to tune gains.

- **Control sign OK**
  - Meaning: you have manually verified that the *controller output* (roll/pitch compensation) pushes the drone in the correct direction to counteract the observed drift.
  - How to use:
    - Start the autostabilizer.
    - Introduce a small drift (or gently translate the drone) and confirm the command arrow/telemetry indicates a corrective command (opposes the drift).
    - If it reinforces the drift (wrong-way correction), adjust `roll_sign` / `pitch_sign` (see above).

Tri-state semantics:

- Checked: `true` (verified OK)
- Unchecked: `false` (verified NOT OK)
- Partially checked: `null` (unknown / not verified yet)

When you click **Save**, these values (plus the **Notes** field) are stored in the session recording `meta.json` under `sign_verification`. This is meant to make recorded sessions self-describing (so you know later whether a run was done with the correct sign conventions).

### 6.7 Visual scale (reference detection)

Visual scale is enabled by `enable_visual_scale=True` and uses a saved **Reference** image to:

- Detect the reference object in the live frame.
- Estimate the reference object’s apparent size in pixels (`ref_size_px`).
- Convert between px/s and m/s using the saved physical pad dimensions.

Key parameters:

- `reference_id: Optional[str] = None`
  - Selected reference record (from the reference store).

- `visual_scale_detection_method: str = "orb_contours"`
  - Selects the detection backend used by `ReferenceDetector`.
  - In the Qt UI this is the **Visual Scale → Detection** dropdown.

- `visual_scale_unstable_cmd_scale: float = 1.0`
  - A multiplier applied to roll/pitch commands **only while visual scale is enabled but not yet stable**.
  - This is a transitional safety/feel knob:
    - `1.0` means full authority during the “unstable scale” period, when the autopilot only knows px/s.
    - Values `< 1.0` reduce authority until the reference becomes stable (and m/s control can take over).
  - In the Qt UI this is **Visual Scale → Unstable cmd scale**.

- `visual_scale_ms_hold_sec: float = 1.0`
  - A hysteresis/grace window for m/s control.
  - When m/s control was active (stable scale) and visual scale briefly becomes unstable or the reference is lost, the stabilizer will keep using the **last stable** m/px scale for up to this duration before falling back to px/s.
  - Set to `0.0` to disable hysteresis.
  - In the UI this is **Visual Scale → m/s hold (s)**.

#### 6.7.1 m/s control

The **Use m/s control** checkbox controls whether the stabilizer is allowed to use the metric velocity estimate from visual scale.

Behavior:

- If **Use m/s control is OFF**:
  - The controller always runs in **px/s** (pixel velocity) mode.
  - Visual scale can still run for diagnostics/telemetry, but it will not change the control units.

- If **Use m/s control is ON**:
  - The controller uses **m/s** *only when* visual scale is considered **stable**.
  - In code, m/s control is enabled only when all of these are true:
    - `enable_visual_scale=True`
    - `use_m_s_control=True`
    - `scale_stable=True` (visual scale stability gate)
    - `vx_m_s` and `vy_m_s` are available (not `None`)

Fallback semantics (important):

- Even if the reference is detected once, **m/s control is not “latched”**.
- If the reference is lost or the detector becomes inconsistent such that `scale_stable` becomes false:
  - the stabilizer will keep using the **last stable** m/px scale for up to `visual_scale_ms_hold_sec`, then
  - it falls back to px/s control.

Command scaling during fallback:

- When **Use m/s control is ON** but m/s control is not active (because scale is not stable yet / was lost), the roll/pitch output is multiplied by `visual_scale_unstable_cmd_scale`.
- This lets you choose between:
  - full authority while scale is unstable (`1.0`), or
  - reduced authority until scale stabilizes (`< 1.0`).

#### 6.7.2 Visual scale stability gate (what `scale_stable` means)

Visual scale exposes a boolean `stable` flag (surfaced in telemetry as `scale_stable`).
This flag is a **gate** that decides whether the system is allowed to use m/s control.

The stability gate is intentionally conservative: it is trying to ensure that the detected reference size is not “jumping around” due to false detections or bad matches.

Step-by-step (per frame):

- A detection is considered **accepted** when:
  - the detector reports `detected=True`, and
  - `ref_size_px > 0`.

- If there was a previous accepted detection, the new detection is additionally checked for **size change consistency**:
  - Compute the per-second relative change of `ref_size_px`:
    - `frac_per_sec = abs(cur - prev) / prev / dt`
  - If `frac_per_sec` exceeds `visual_scale_max_ref_size_frac_per_sec`, the detection is **rejected** for this frame.
    - This rejects sudden scale jumps (often caused by wrong quad, wrong homography, or latching to a different object).

- If the detection is accepted:
  - An internal counter `stable_seen` is incremented.
  - `scale_stable=True` only when `stable_seen >= visual_scale_stable_frames`.

- If the detection is not accepted (not detected or rejected by the jump gate):
  - `stable_seen` is reset to `0`.
  - `scale_stable=False` immediately.

Important implications:

- Stability requires **consecutive** accepted detections.
- If the reference is intermittently lost (or detections are unstable), `scale_stable` will flap and m/s control will repeatedly fall back to px/s.

Tuning knobs (see config reference):

- `visual_scale_stable_frames`
  - How many consecutive accepted frames are required before declaring stable.
  - Larger values reduce false “stable” but increase the time-to-m/s.

- `visual_scale_max_ref_size_frac_per_sec`
  - Maximum allowed relative scale change rate before rejecting the detection.
  - Larger values are more permissive (can stabilize sooner), but may allow bad detections to be treated as stable.

#### 6.7.3 Detection method overview

The detector returns a `ReferenceDetectionResult` with:

- `detected`: whether a reference was found
- `quad_xy`: a 4-point quadrilateral (image coordinates) when detected
- `ref_width_px`, `ref_height_px`, `ref_size_px`: geometric size estimates derived from `quad_xy`
- `mode`: which backend produced the result

The implementation lives in `e88_autopilot/reference_detection.py`.

#### 6.7.4 Method: ORB + Contours (`visual_scale_detection_method="orb_contours"`)

This is the default method and is meant to work with a generic printed reference image.

Step-by-step:

1. **ORB features on the stored reference**
   - When the detector is created, it computes ORB keypoints/descriptors for the saved `reference_bgr`.

2. **ORB features on the live frame**
   - Each update computes ORB keypoints/descriptors on the incoming frame.

3. **Descriptor matching (KNN + ratio test)**
   - Uses `BFMatcher(NORM_HAMMING)` with `knnMatch(k=2)`.
   - Applies a Lowe-style ratio test (keeps matches where `m.distance < 0.75 * n.distance`).

4. **Homography estimation (RANSAC)**
   - Uses `cv2.findHomography(..., cv2.RANSAC, ransacReprojThreshold=5.0)`.
   - Rejects detections if the number of inliers or inlier ratio is too small.

5. **Project the reference corners into the frame**
   - The reference image rectangle corners are projected with `cv2.perspectiveTransform`.
   - The resulting quadrilateral becomes `quad_xy`.

If ORB/homography fails (insufficient matches/inliers), it falls back to a contour-based estimate:

- **Contour fallback (“square-ish board”)**
  - Converts to grayscale, blurs, runs Canny.
  - Dilates edges slightly.
  - Finds external contours.
  - Stacks “significant” contours (by arc-length threshold) and fits a `minAreaRect`.
  - Forces that rectangle to a square (side = min(w, h)) and returns it as `quad_xy`.

Notes:

- ORB is good when the reference image has texture (corners/unique patterns).
- The contour fallback is a heuristic: it can lock onto other large square-ish objects if the scene has them.
- `mode` is reported as `"orb"` when homography succeeds, else `"contours"` when the fallback is used.

#### 6.7.5 Method: Closed Quad (`visual_scale_detection_method="closed_quad"`)

This method does *not* rely on keypoints/descriptors. It tries to find a big quadrilateral “board-like” shape.

Step-by-step:

1. Convert to grayscale and blur
2. Run Canny edge detection
3. **Morphological closing**
   - Applies `cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)` to connect broken edges.
4. Find contours (`cv2.RETR_TREE`)
5. Consider the largest contours (by area) and approximate polygons
   - Uses `cv2.approxPolyDP(contour, eps_frac * perimeter, True)`.
   - Accepts the first contour whose approximation has exactly 4 vertices.
6. Fit a min-area rectangle to that quad and use its `boxPoints` as `quad_xy`

Notes:

- This is often more robust than ORB when the reference has low texture but strong edges.
- It is sensitive to scene clutter: any large quad-like contour may be selected.
- `mode` is reported as `"closed_quad"`.

#### 6.7.4 Method: Markers (`mode="markers"`)

This mode is intended for references that include **ArUco markers**.

Step-by-step:

1. Convert to grayscale
2. Detect markers using OpenCV’s ArUco module
   - `cv2.aruco.detectMarkers(gray, DICT_4X4_50)`
3. If fewer than 4 markers are found, detection fails.
4. Concatenate all marker corner points and compute a convex hull.
5. Fit a minimum-area rectangle to the hull (`minAreaRect` + `boxPoints`) and return it as `quad_xy`.

Important behavior:

- If a saved reference record has `markers_present=True`, the visual-scale pipeline forces marker detection.
  - This overrides the UI’s `orb_contours` / `closed_quad` selection to preserve marker-based references.

---

## 7. Telemetry and visualization

The stabilizer emits `StabilizerTelemetry` containing:

- `phase`: `takeoff`, `climb`, `settle`, `hold`
- `frame_bgr`: the video frame (if available)
- `flow`: `FlowEstimate` (dx/dy/vx/vy, quality, counts)
- `tracks`: feature tracks (`prev_xy`, `next_xy`, `inliers`) for overlay
- `kalman`: Kalman estimate (when enabled)
- `pos_x_px`, `pos_y_px`: the plotted position
- `cmd_roll`, `cmd_pitch`, `cmd_throttle`: the issued commands

Frame-health fields:

- `frame_stale_ms`: how old the latest decoded frame is (time since it was received by the frame thread).
- `frame_seq`: monotonically increasing internal sequence number (every received frame increments it).
- `frame_is_new`: whether the control loop consumed a new frame compared to the previous loop.
- `frames_dropped`: cumulative count of frames that were overwritten in the size-1 buffer before the control loop could consume them.

Timing fields:

- `dt_flow_ms`: time spent inside optical-flow update.
- `dt_total_ms`: time from loop start until command send.
- `estimated_latency_ms`: coarse estimate of sensor-to-command latency computed as `(t_cmd_sent - frame_timestamp)`.

Frame freshness fields:

- `frame_age_ms`: `t_flow_start - timestamp`.
  - Uses the frame’s `timestamp`, which in this project is a **local monotonic timestamp assigned when OpenCV decodes the frame** (not a drone-provided capture timestamp).
  - Measures how long it has been since the frame became available to the process (decode → flow start).
- `frame_stale_ms`: `t_flow_start - t_frame_received`.
  - Uses the time when the background frame thread received/decoded the frame and placed it in the size-1 buffer.
  - Mostly measures *how long the latest decoded frame sat around before the control loop used it*.

Optical-flow timestamp behavior:

- In `hold` phase, the optical-flow estimator’s internal `dt` is computed from local `time.monotonic()` time (control-loop time), not from the decoded frame’s `timestamp`.
  - This avoids `dt == 0` (and thus `flow=None`) when the size-1 frame buffer reuses the same decoded frame for multiple control iterations.
  - This does not change `frame_age_ms`, `estimated_latency_ms`, or the recorded telemetry `timestamp` field, which still refer to the decoded frame timestamp.

In general you should expect `frame_age_ms >= frame_stale_ms`.

Relationship between `timestamp` and `t_frame_received`:

- `timestamp` is recorded in the video decode thread when a frame is read/decoded.
- `t_frame_received` is recorded in the `LatestFrameBuffer` thread when it fetches that decoded frame and writes it into the size-1 buffer.


- `kf_input_vx_px_s`, `kf_input_vy_px_s`: the conditioned velocity actually fed into the Kalman update.
- `kf_gated`: whether the conditioner suppressed (zeroed) at least one component due to deadband.

In the Qt UI:

- The video shows LK tracks and text overlays (phase, velocity, command).
- The 2D widget plots trajectory `(pos_x_px, pos_y_px)` (cyan trace) and draws the latest
  compensation vector (yellow arrow) scaled relative to `max_cmd`.

### 7.1 Understanding the key metrics (recommended reading)

**`dt_total_ms` vs `dt_flow_ms`**

- `dt_flow_ms` is only the optical-flow compute time.
- `dt_total_ms` is the overall time spent in the control iteration until the command is sent.

If `dt_total_ms` grows, the controller becomes sluggish even if `dt_flow_ms` stays small.

**Loop and frame jitter (`loop_dt_ms_std`, `frame_dt_ms_std`)**

The UI also reports two “jitter” metrics computed over the rolling diagnostics window (~1s):

- `loop_dt_ms_std`: standard deviation of `Δt_loop_start`.
  - Measures how much the *control loop period* varies.
  - High values indicate scheduling jitter (UI thread load, Python runtime/GIL effects, sleeping granularity).

- `frame_dt_ms_std`: standard deviation of `Δt_frame_received` (for *new* frames only).
  - Measures how bursty/unstable frame delivery is.
  - High values usually indicate RTSP/network/decode irregularities.

**`kf_in` and `gated`**

The Kalman filter (when enabled) is fed the *conditioned* velocity:

- `kf_in vx/vy` are after applying `estimator_deadband_px_s`.
- `gated=1` means at least one component was forced to 0 because it was below the deadband.

This keeps the Kalman position estimate from drifting when the drone is actually still.

**Why frames are “dropped”**

With the size-1 frame buffer, “dropped” means:

- A newer frame arrived before the control loop consumed the previous one.
- The previous one is overwritten (intentionally) so the controller always uses the freshest data.

Interpretation:

- `frames_dropped` is workload/throughput feedback: if it increases rapidly, the control loop is not keeping up with the camera stream.
- The absolute number depends on your stream FPS and control rate, so it’s often more meaningful to look at a *rate*.

Derived metrics:

- `drop_pct ~= 100 * frames_dropped / frame_seq` (cumulative percent of overwritten frames)

or (per-second) compute `Δframes_dropped / Δframe_seq` over a window.

---

## 8. Known challenges and mitigations

### 8.1 Camera orientation and axis confusion

The raw RTSP frames can have an orientation that does not match what “feels natural”
in the UI.

Mitigation:

- The Qt display code rotates the shown image for readability.
- The autopilot overlay is now drawn in the displayed orientation.

Note:

- The stabilizer’s computation uses the raw camera frame coordinates.
- Use `roll_sign` / `pitch_sign` to fix direction mismatches.

### 8.2 “Torn / split frames” (RTSP/UDP artifacts)

Symptoms:

- The image appears split (two halves don’t correspond to the same scene slice).
- Optical flow interprets this as a huge movement even while the drone is stationary.

Cause:

- RTSP over UDP can produce corrupted decoded frames when packets are lost.
- Switching cameras restarts the stream/decoder and temporarily fixes it.

Mitigation implemented:

- The optical flow estimator rejects suspicious frames using:
  - translation jump threshold (`max_translation_frac`)
  - RANSAC inlier ratio threshold (`min_inlier_ratio`)
- On rejection, it resets tracking and returns `None`, preventing large control outputs.

### 8.3 Lighting and low-texture scenes

If the camera sees a blank wall or low-light scene:

- `goodFeaturesToTrack` finds few corners.
- LK tracking becomes unstable.
- Quality drops and control disables.

Mitigation:

- Require minimum tracked features.
- Periodic reinitialization.

### 8.4 Latency, timing, and control rate

If the video stream stalls or frame timestamps are irregular:

- `dt` can become large, producing misleading velocities.

Mitigation:

- The estimator ignores extremely small `dt`.
- The control loop uses a fixed `cmd_rate_hz` and sends neutral commands when frames are missing.

### 8.5 Stationary drift (“ghost motion”) from integrating noisy flow

Symptom observed:

- With the drone physically still, the plotted position would drift over time.
- When Kalman was enabled, the drift could be worse because the filter integrates velocity into position.

Root cause:

- The optical-flow velocity estimate has small zero-mean noise.
- Even if the controller has a deadband, the Kalman filter was still receiving the *raw* noisy velocity.
- Integrating that velocity into position accumulates error, producing visible “ghost motion”.

Mitigation implemented:

- A velocity conditioner now runs *before* the Kalman update and can zero out small measurements.
- We also expose telemetry showing what velocity was actually fed into the filter.

### 8.6 Conditioning before Kalman: deadband + gating telemetry

Design change:

- Introduced `VelocityMeasurementConditioner` (`measurement_conditioner.py`).
- It applies a deadband in px/s and returns:
  - conditioned velocities (`vx_px_s`, `vy_px_s`)
  - a boolean `gated` flag (whether the measurement was suppressed)

Where it lives in the pipeline:

- Optical flow produces `FlowEstimate.vx_px_s/vy_px_s`.
- Conditioning is applied next.
- The conditioned values are then passed to `VelocityKalman2D.update_velocity(...)`.

Why this matters:

- The controller deadband prevents over-reacting, but it does not prevent the estimator state from drifting.
- Conditioning *before* state estimation keeps both the controller and the internal position estimate stable.

Telemetry support:

- `StabilizerTelemetry` now includes:
  - `kf_input_vx_px_s`, `kf_input_vy_px_s`
  - `kf_gated`

### 8.7 Stationary calibration and persistence

Problem:

- “Good defaults” for `kalman_sigma_v` and an estimator deadband are environment- and stream-dependent.
- We needed a repeatable way to tune these from real data while the drone is still.

Mitigation implemented:

- Added stationary calibration (`calibration.py`) which:
  - samples flow velocities while the drone is stationary,
  - computes a robust deadband using a percentile of `|vx|` and `|vy|`,
  - computes a robust `kalman_sigma_v` using MAD-derived sigma.

Persistence:

- Calibration results are stored in `experimental/e88_autopilot/calibration.json`.
- On startup, the caller can load and apply `estimator_deadband_px_s` and `kalman_sigma_v`.

Why only these parameters:

- This calibration specifically targets the “stationary drift” failure mode and measurement trust.
- It does not attempt to infer higher-level motion models (e.g., `sigma_a`) or map pixels to meters.

### 8.8 Optical flow scaling with downscale

Symptom observed:

- After calibration, motion was only detected reliably when moving extremely slowly.

Root cause:

- When downscaling frames for LK tracking (performance/robustness), displacement was being reported in the downscaled pixel units.
- This effectively shrinks the velocity magnitude, making real motion look too small.

Mitigation implemented:

- The estimator now rescales `dx/dy` back to full-resolution pixel units before producing `FlowEstimate`.
- Internal gating (e.g., translation sanity checks) can still operate in the tracking coordinate system.

### 8.9 Robust fallback for large displacements (phase correlation)

Problem:

- At low effective FPS (e.g., ~9 Hz), the per-frame displacement after takeoff can be large.
- LK + RANSAC can fail (low inlier ratio) or exceed translation gates, causing repeated resets and loss of velocity signal.

Mitigation implemented:

- Added an optional fallback path based on `cv2.phaseCorrelate`.
- When LK tracking fails or is rejected (too-large translation / low inlier ratio), the estimator can:
  - estimate global translation via phase correlation,
  - return a usable `(dx, dy)` and a quality proxy based on response,
  - reinitialize tracking afterward to resume LK once motion becomes trackable.

Why phase correlation:

- It is resilient to larger global translations than local-feature tracking.
- It provides a natural response/quality metric that can be gated.

---

## 9. How to tune safely

Recommended safety approach:

1. Test with the drone restrained or in a safe space.
2. Start with:
   - low `max_cmd` (e.g. 0.10–0.20)
   - conservative gains
3. Verify sign:
   - move the drone slowly, confirm commands push back.
4. Increase `kp_*` gradually until it corrects drift.
5. Add `ki_*` only if there is consistent steady drift that P alone can’t remove.

If you see oscillation:

- decrease `kp_*`
- decrease `cmd_rate_hz`
- increase `kalman_sigma_v` (more smoothing)

---

## 10. File guide

- `autostabilizer.py`
  - `StabilizerConfig`
  - `StabilizerTelemetry`
  - `AutoStabilizer`

- `optical_flow.py`
  - `LucasKanadeDriftEstimator`
  - `FlowEstimate` / `FlowTracks`

- `kalman.py`
  - `VelocityKalman2D` / `KalmanEstimate`

- `controller.py`
  - `VelocityHoldController` / `HoldControlOutput`

- `run_autostabilizer.py`
  - CLI to run without the Qt UI
