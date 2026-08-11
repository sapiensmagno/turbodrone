# E88 Simulator

A simulator for the E88 and its downward-camera stabilization loop, built to
answer one question before a real flight:

> **Will the autostabilizer hold a stable hover, or will it fight itself?**

It answers that by flying the **unmodified** `AutoStabilizer` — real optical
flow, real Kalman, real controller, real threading — against a physically
simulated drone and synthetically rendered camera frames.

---

## What it found

Three results came out of building it. All three are asserted by tests so they
cannot regress silently.

### 1. The shipped proportional gain is past the stability knee

`kp_vx = kp_vy = 0.003` is the value in `StabilizerConfig` and in every recent
session's `meta.json`. Sweeping it at 0.7 m altitude, scoring each run against an
open-loop run of the *same* scenario and seed:

| `kp`       | speed RMS vs no control | position excursion vs no control |
|------------|------------------------|----------------------------------|
| 0.0002     | ×0.75–0.79             | ×0.63–0.65                       |
| 0.0005     | ×0.61–0.68             | ×0.40–0.42                       |
| **0.0010** | **×0.55–0.67**         | **×0.26–0.28**                   |
| **0.0015** | **×0.56–0.74**         | **×0.19–0.22**                   |
| 0.0020     | ×0.61–0.90             | ×0.15–0.19                       |
| 0.0030     | ×1.47–2.46 ✗           | ×0.15–0.25                       |
| 0.0050     | ×8.92–12.52 ✗          | ×0.55–1.51                       |

At `kp = 0.003` the drone stays near its start point but **moves more than it
would with no control at all** — the loop is converting measurement noise into
motion. The knee sits between 0.002 and 0.003.

Why: at 0.7 m the flow gain is `f/z = 365.6/0.7 = 522` px/s per m/s. With
`kp = 0.003` the loop gain is `0.003 × 522 × 20° × g ≈ 5.4 rad/s`, against a
total sensing delay of roughly 150 ms. A crossover that fast with that much
delay has almost no phase margin. Note the loop gain scales with `1/z`, so the
same gain is 1.75× hotter at 0.4 m than at 0.7 m.

**Recommendation: `kp_vx = kp_vy = 0.0012`, `ki_vx = ki_vy = 0.0002`.** That
configuration passes the whole pre-flight battery, confirmed both in the fast
loop and vision-in-the-loop.

But the loop gain scales as `tilt_authority / altitude`, and both of those are
unmeasured. Crossing them (`--suite robustness`, 27 corners of lean angle ×
latency × altitude, 2 seeds each):

| gain | corners passed | where it fails |
|---|---|---|
| `kp = 0.0012` | 22 / 27 | every 0.4 m corner with a strong airframe, plus 30° + 250 ms at 0.7 m |
| **`kp = 0.0008`** | **26 / 27** | only 30° lean **and** 250 ms latency **and** 0.4 m together |

At 0.4 m with a 30° airframe the effective loop gain is 2.6× its value at 0.7 m
with a 20° one — enough on its own to push `kp = 0.0012` back past the knee.

**So: fly the first flight at `kp = 0.0008` and hover at 0.7 m or above.** Once
the real lean angle and latency are known, `kp = 0.0012` gives a tighter hold.
This altitude dependence is exactly what the visual-scale m/s control path
(roadmap Phase 1) exists to remove; until it is trusted, altitude has to be
treated as a flight constraint rather than a free variable.

### 2. Reducing the gain removes the latency sensitivity entirely

The true photon-to-command latency of this drone has never been measured —
session telemetry only times from *decode* onward, which misses exposure,
onboard encoding, WiFi and jitter buffering. At `kp = 0.0012`:

| video latency | 50 ms | 100 ms | 150 ms | 200 ms | 300 ms | 400 ms |
|---------------|-------|--------|--------|--------|--------|--------|
| speed vs open loop | ×0.52–0.65 | ×0.55–0.68 | ×0.58–0.73 | ×0.60–0.77 | ×0.66–0.86 | ×0.72–0.93 |

Stable across the whole range. The unmeasured latency stops being a risk once
the gain is right, which is worth more than measuring it would be.

### 3. `affine_translation` turns altitude change into phantom sideways drift

A change in altitude scales the image. An affine fit has no scale parameter, so
it absorbs the zoom as translation: a zoom by `s` about the image centre gives
translation `(1 − s)·c`, where `c` is the principal point. Measured against the
renderer, `affine_translation` shows a leak coefficient of **1.000 ± 0.014** —
the full textbook value.

In practice, a 10 mm climb at 0.7 m fabricates 4.5 px of lateral displacement,
i.e. **90 px/s of velocity that is not happening** — against a real measured
noise floor of 2.46 px/s. Barometric altitude hold bobs continuously, so this is
not an edge case.

Measured estimator error during a simulated hover, at 30 mm of altitude bobbing:

| motion model           | RMS error @ 0.7 m | leak coefficient |
|------------------------|-------------------|------------------|
| `translation_rotation` | **5.4 px/s**      | 0.09 ± 0.09      |
| `derotation`           | 53.8 px/s         | 0.24 ± **1.00**  |
| `affine_translation`   | 54.8 px/s         | 1.000 ± 0.014    |

`derotation` — the model configured in the most recent sessions — is not merely
worse, it is *erratic*: its y-axis leak standard deviation of 1.0 dwarfs its
0.11 mean, so individual frames can invert or amplify the error. It degrades
sharply at low altitude (100 px/s RMS at 0.5 m).

`derotation` does do its actual job well: with a pure tilt and no translation it
reports near zero where the others report the full apparent motion. The tradeoff
is real, and the altitude-leak weakness is the price. At `kp = 0.0012` the loop
is gentle enough that all three models fly acceptably, so this matters most if
the gain is ever raised.

---

## Running it

```bash
# Pre-flight battery on the shipped configuration
python -m e88_sim.run_sim --suite preflight

# ...and on the recommended one
python -m e88_sim.run_sim --suite preflight --kp 0.0012 --ki 0.0002

# Where is the stability knee?
python -m e88_sim.run_sim --suite gain --seeds 3

# How much latency can these gains take?
python -m e88_sim.run_sim --suite latency --kp 0.0012 --ki 0.0002

# Cross the unmeasured parameters: lean angle x latency x altitude
python -m e88_sim.run_sim --suite robustness --kp 0.0012 --ki 0.0002

# Confirm one scenario with the real autopilot on rendered frames (real time)
python -m e88_sim.run_sim --suite preflight --vision --scenario nominal_hover --kp 0.0012 --ki 0.0002
```

Exit codes: `0` go (possibly with caution), `1` a scenario failed, `3` a
falsification scenario passed when it should have failed.

---

## Two levels of fidelity

**Vision-in-the-loop** (`sim_drone.SimulatedDrone` + `AutoStabilizer`).
`SimulatedDrone` implements the `turbodrone.Drone` interface, so the real
autopilot flies it with nothing stubbed. This is the authoritative check,
because the thing most likely to crash an E88 is not a mistuned gain, it is a
sign convention or an axis swap — and those live in the wiring of
`autostabilizer.py`, not in any component testable in isolation. Runs at real
time.

**Fast loop** (`fast_loop.run_fast`). Mirrors the hold loop step for step using
the same `VelocityMeasurementConditioner`, `VelocityKalman2D` and
`VelocityHoldController`, including the axis swap at `autostabilizer.py:894`.
Only optical flow is replaced, by an analytic sensor that computes what the
estimator *would* have reported. Fast enough for sweeps.

The duplication is the riskiest thing here, so `tests/test_sim_fast_loop.py`
flies the same scenario both ways and asserts they agree. They match closely in
the stable regime (0.108–0.123 vs 0.117 m/s speed RMS at the tuned gain) and the
fast loop is pessimistic past the knee (×3.56 vs ×1.52 at `kp = 0.003`) — a safe
bias for a screening tool, but the reason **any conclusion from a sweep should be
confirmed with one vision-in-the-loop run before acting on it.**

---

## How it is calibrated

`config.py` tags every parameter with its provenance:

- **[MEASURED]** — from recorded sessions, reference records, or the protocol
  code in `e88/`. 640×480 at 20 fps; `f = 365.65` px; 33 Hz RC transmission;
  byte quantization of stick commands.
- **[CALIBRATED]** — tuned so the simulator reproduces a measured statistic.
- **[ESTIMATED]** — physically reasonable for a ~90 g toy quad but not measured
  on this airframe. Maximum lean angle, video latency, altitude bobbing, drag.
- **[POLICY]** — a choice about what to simulate.

The [ESTIMATED] values are the ones `robustness_suite` sweeps, so a verdict is a
statement about a range of plausible drones rather than one lucky parameter set.

**External anchors the simulator is checked against** (`tests/test_sim_realism.py`):

| quantity | real value | source |
|---|---|---|
| focal length | 365.65 px | `flow_focal_length_px` in every recent session; independently, a 0.30 m pad at 156.7 px from 0.70 m |
| frame rate / flow dt | 19.8–19.9 Hz / 0.050–0.051 s | session telemetry |
| features tracked | 49.4 (quiet session), 37–84 overall | `session_20260117_194839_319431` and others |
| flow quality | 0.998 quiet, 0.83–0.91 with motion | same |
| stationary noise floor | 2.46 px/s (p99) | `calibration.json` |
| affine altitude leak | 1.000 ± 0.014 | measured here; matches the closed form exactly |

Three modelling choices exist specifically because a naive simulator would have
been optimistic:

- **Sparse strong speckles, weak background.** A rich fractal texture makes
  `goodFeaturesToTrack` return its full 250-corner budget of weak, self-similar
  corners. Real sessions show 37–84 distinctive features.
- **The floor is not a plane.** A `relief_fraction` of raised objects produces
  genuine parallax, and therefore the outlier tracks behind the recorded
  `inlier_ratio` of ~0.89. A planar scene tracks at 1.00 and flatters RANSAC.
- **Airframe vibration.** Rendering plus sensor noise plus JPEG accounts for only
  0.6 px/s of the measured 2.46 px/s stationary floor. The remainder is modelled
  explicitly rather than assumed away.

### Known limitations

- **The stationary noise anchor may be an underestimate for flight.**
  `calibration.json` was measured with the drone still, most likely motors off.
  The `noisy_camera` scenario runs at 5× to check the conclusion survives.
- **Maximum lean angle is a guess** (20°), and it sets the command-to-
  acceleration gain directly. The `weak_airframe` / `strong_airframe` scenarios
  cover 10° and 30°.
- **The vision-in-the-loop control loop can run below 20 Hz** because rendering
  competes with it for CPU in the same process. `run_sim` prints a note when
  this happens. The real drone achieves ~20 Hz, so the fast loop is *more*
  faithful on timing.
- **No lens distortion by default.** The camera has never been intrinsically
  calibrated (`intrinsics.json` does not exist). Distortion is a
  radial-quadratic perturbation and therefore corrupts exactly the `(1 + u'²)`
  term the derotation model uses to separate tilt from translation. The
  parameters exist in `CameraParams`; run `run_intrinsics_calibration` and set
  them before trusting any derotation result.
- **The visual-scale / m/s control path is not exercised** by the default
  scenarios. `GroundParams.pad_image_path` composites a pad of known physical
  size onto the floor, which makes it testable — altitude ground truth is known,
  so the inferred altitude can be scored — but no scenario uses it yet.

---

## Falsification

A simulator that passes everything is not measuring anything. Three scenarios
are designed to fail, and `run_sim` exits with code 3 if any of them passes:

| scenario | result |
|---|---|
| `wrong_signs` (`roll_sign` and `pitch_sign` inverted) | diverges to 64–70 m |
| `wrong_roll_sign_only` | diverges to 52–55 m |
| `excessive_gain` (5×) | speed exceeds 1.5 m/s |

The sign scenarios matter most. A `roll_sign` error turns the controller into
positive feedback, and the simulator catches it on the bench in twelve seconds.

---

## Module guide

| file | contents |
|---|---|
| `config.py` | every parameter, tagged with provenance |
| `physics.py` | rigid body, angle-mode attitude loop, drag, altitude hold, 33 Hz command path with quantization and latency |
| `ground.py` | procedural floor texture, raised-object relief layer, pad compositing |
| `camera.py` | downward camera as a plane homography; motion blur, noise, vignette, JPEG, distortion; exact ground-truth flow |
| `sim_drone.py` | `SimulatedDrone`, a drop-in `turbodrone.Drone` |
| `flow_sensor.py` | analytic optical-flow surrogate with calibrated error model |
| `fast_loop.py` | deterministic closed loop on a virtual clock |
| `metrics.py` | hover scoring and PASS/MARGINAL/FAIL verdicts |
| `scenarios.py` | the pre-flight battery and the sweeps |
| `run_sim.py` | CLI |

## Before the real flight

The simulator supports a stable hover at `kp = 0.0008`; it cannot prove one. It
is a model, and its two most influential parameters — maximum lean angle and
photon-to-command latency — are still estimates. The highest-value things it
cannot do for you, roughly in order:

1. **Verify the sign conventions on the real drone**, restrained, at low
   throttle. This is the failure that ends flights: simulation shows an inverted
   `roll_sign` diverging to 64 m in twelve seconds, and no amount of gain tuning
   survives it.
2. **Measure the maximum lean angle.** It sets the command-to-acceleration gain
   directly, and it is the difference between 22/27 and 26/27 robustness
   corners. A phone video of a full-stick lean against a level reference is
   enough.
3. **Measure the real photon-to-command latency** once, with a clock or an LED
   in frame. The sweep says the tuned gains tolerate 400 ms, so this becomes a
   confirmation rather than a risk.
4. **Run the intrinsic calibration** (`e88_autopilot.run_intrinsics_calibration`),
   so distortion and the principal point stop being assumptions — particularly
   before trusting anything from the `derotation` model.
5. **Re-run `--suite preflight` after any gain change.** It takes three minutes.

### Suggested first flight

- `kp_vx = kp_vy = 0.0008`, `ki_vx = ki_vy = 0.00013`
- `flow_motion_model = "translation_rotation"`
- hover at 0.7 m or above, not 0.4 m
- keep `max_cmd` low for the first flight; the tuned loop never approaches 0.9

Expected behaviour if the model is right: the drone holds within roughly 0.5 m
over 20 s and drifts slowly rather than buzzing. **Slow drift is the designed
behaviour, not a failure** — this is a velocity-hold loop with no position
feedback, so it can null velocity but cannot return to a point. Visible
high-frequency jitter is the symptom to abort on, because that is what too much
gain for the latency looks like.

---

## Known gaps in this simulator

Found by review after the numbers above were produced. They do not overturn the
direction of the gain result — every mechanism here would make an over-hot loop
look *better* behaved, not worse — but they do mean the specific knee location
and corner counts should be treated as provisional until fixed.

1. **Scenarios hardcode `kalman_sigma_v = 50.0`.** That is the `StabilizerConfig`
   default, but the Qt app loads `calibration.json`, which now supplies `1.109`.
   Less smoothing means a noisier velocity estimate reaching the controller, so
   the real knee is likely *lower* than measured, not higher. The sweep should
   be re-run at the flown value, and the CLI needs a sigma override.
2. **`low_texture` does not degrade flow quality.** `AnalyticFlowSensor` never
   consults `GroundParams`, so it emits the same quality and feature counts as
   nominal. That scenario can report PASS without exercising feature loss or
   quality gating at all.
3. **`lossy_link` tear probability is unused** by `run_fast`, so the advertised
   tearing stress is absent from the non-vision verdict.
4. **`run_fast` does not log state during stale-frame holds**, leaving a hole in
   the recorded trajectory. A freeze lasting to the end truncates the scored
   duration, so RMS and FFT metrics underweight exactly the behaviour under test.
5. **Metrics can return PASS on NaN.** Only `radius_all` is checked for
   finiteness; an isolated NaN in the velocity arrays survives `np.nanmax` and
   makes every threshold comparison false.
6. **Latency jitter can reorder frames.** Sorting by `deliver_at` lets a
   later-captured frame arrive first, so LK sees pose go backwards in time --
   the opposite of the in-order guarantee the code claims, and most likely in
   the high-latency scenario.
7. **Matching noise scales with `dt`**, which makes the resulting velocity noise
   constant across frame intervals instead of amplifying on short ones. This
   flattens exactly the effect the lossy and timing scenarios exist to probe.
8. **`run_fast` builds the full 4000x4000 ground texture** on every run and seed
   even though the analytic sensor never reads a pixel, which dominates the
   runtime of a path documented as ~40 ms.
