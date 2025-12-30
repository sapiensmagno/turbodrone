# E88 Autopilot (Experimental)

This folder contains an **experimental** “autostabilizer” for the E88-class drone.

The goal is simple:

- Keep the drone visually stable (hovering in place) using only the **onboard camera stream**.
- Estimate the drone’s planar motion (image-space drift) using **optical flow**.
- Smooth that motion estimate using a **Kalman filter**.
- Convert the estimated drift into small **roll/pitch compensation commands**.
- Provide real-time **telemetry** for a Qt UI (optical-flow overlay + 2D motion/compensation plot).

---

## Index

- [1. Quick start](#1-quick-start)
- [2. What “stabilization” means here](#2-what-stabilization-means-here)
- [3. Core pipeline overview](#3-core-pipeline-overview)
- [4. Concepts primer](#4-concepts-primer)
  - [4.1 Optical flow (Lucas–Kanade) in plain language](#41-optical-flow-lucas–kanade-in-plain-language)
  - [4.2 Robust motion via RANSAC](#42-robust-motion-via-ransac)
  - [4.3 Kalman filtering in plain language](#43-kalman-filtering-in-plain-language)
  - [4.4 Controller: turning drift into roll/pitch](#44-controller-turning-drift-into-rollpitch)
- [5. Software architecture](#5-software-architecture)
- [6. Configuration reference (all parameters)](#6-configuration-reference-all-parameters)
  - [6.1 Takeoff / climb / settle phase parameters](#61-takeoff--climb--settle-phase-parameters)
  - [6.2 Control loop parameters](#62-control-loop-parameters)
  - [6.3 Estimation parameters (Kalman)](#63-estimation-parameters-kalman)
  - [6.4 Motion-quality / safety gates](#64-motion-quality--safety-gates)
  - [6.5 PID-like controller parameters](#65-pid-like-controller-parameters)
  - [6.6 Sign conventions (roll_sign, pitch_sign)](#66-sign-conventions-roll_sign-pitch_sign)
- [7. Telemetry and visualization](#7-telemetry-and-visualization)
- [8. Known challenges and mitigations](#8-known-challenges-and-mitigations)
  - [8.1 Camera orientation and axis confusion](#81-camera-orientation-and-axis-confusion)
  - [8.2 “Torn / split frames” (RTSP/UDP artifacts)](#82-torn--split-frames-rtspudp-artifacts)
  - [8.3 Lighting and low-texture scenes](#83-lighting-and-low-texture-scenes)
  - [8.4 Latency, timing, and control rate](#84-latency-timing-and-control-rate)
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

- Chose camera. Switch between them to reset the stream if you experience video issues. They may happen due to RTSP over UDP (packet loss/jitter)
- Configure parameters in the **Autostabilizer** section.
- Click **Start autostabilizer**.

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
2. **Track feature points** across consecutive frames (Lucas–Kanade optical flow).
3. Estimate a single “best” global translation `(dx, dy)` between frames using **RANSAC**.
4. Convert `(dx, dy, dt)` into velocity `(vx, vy)` in pixels/second.
5. Optionally run a **Kalman filter** to smooth velocities and derive a more stable position estimate.
6. Feed `(vx, vy)` into a **controller** to compute `(roll_cmd, pitch_cmd)`.
7. Send `(roll_cmd, pitch_cmd, throttle)` to the drone.
8. Emit **telemetry** so the UI can display:
   - the camera frame,
   - flow tracks,
   - estimated velocity/position,
   - command outputs.

---

## 4. Concepts primer

### 4.1 Optical flow (Lucas–Kanade) in plain language

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

### 4.3 Kalman filtering in plain language

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

Once we have an estimate of drift velocity:

- If the scene appears to move right, the drone is drifting left (or equivalently, the camera is moving left relative to scene).
- We send the opposite command to counteract.

The controller in this project (`VelocityHoldController`) behaves like a small PID-family controller:

- Proportional term: `u ~ kp * v`
- Integral term: `u ~ ki * ∫ v dt`
- Deadband: ignore tiny velocities
- Saturation: clamp output to `[-max_cmd, +max_cmd]`

We do not currently include a derivative term.

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

In `LucasKanadeDriftEstimator` (optical flow) there are additional *internal* robustness gates
that protect against corrupted frames (e.g., torn frames):

- `max_translation_frac: float = 0.25` (internal default)
  - If estimated per-frame translation exceeds this fraction of the frame size,
    the frame is treated as corrupted and the tracker resets.

- `min_inlier_ratio: float = 0.6` (internal default)
  - If RANSAC inlier ratio drops below this, the frame is treated as corrupted
    and the tracker resets.

These are not currently exposed in `StabilizerConfig`.

### 6.5 PID-like controller parameters

These live in `VelocityHoldController` and are surfaced through `StabilizerConfig`:

- `max_cmd: float = 0.35`
  - Clamp for absolute roll/pitch output.

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

### 6.6 Sign conventions (roll_sign, pitch_sign)

- `roll_sign: float = -1.0`
- `pitch_sign: float = -1.0`

These exist because “positive image drift” does not necessarily correspond to
“positive command” on a given drone/camera orientation.

If the drone corrects in the wrong direction:

- flip `roll_sign` from `-1` to `+1`, or
- flip `pitch_sign` from `-1` to `+1`.

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

In the Qt UI:

- The video shows LK tracks and text overlays (phase, velocity, command).
- The 2D widget plots trajectory `(pos_x_px, pos_y_px)` and draws the latest
  compensation vector (scaled by `max_cmd`).

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
