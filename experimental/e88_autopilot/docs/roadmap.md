 # Autonomy-First Stabilizer Roadmap
 
 This document is the consolidated, final plan for evolving the current `experimental/` autostabilizer into a repeatable, autonomy-first stabilization platform.
  
 ---
 
 ## 1) Executive Summary
 
 We will build an autopilot that stabilizes the drone using its **downward camera** by treating the system as a real control problem:
 
 - **Perception** estimates drift (optical flow → velocity in px/s).
 - **Estimation** conditions and filters measurements (deadband + Kalman).
 - **Control** commands roll/pitch to counter drift (PI initially).
 - **Supervision** enforces safety constraints and aborts/lands instead of crashing.
 - **Experimentation** autonomously identifies the drone response and tunes itself.
 - **Observability** makes every failure mode explainable through logs and UI.
 
 We implement this through phases **0–5**. Each phase produces reusable infrastructure that will later support higher-level capabilities (e.g., **Bezier-curve flight**) by building on a stable, measurable inner-loop stabilization foundation.
 
 ---
 
 ## 2) Constraints & Assumptions
 
 - **Scope:** work only inside `experimental/`.
 - **Primary UX:** Qt app is the control room; CLI is secondary.
 - **Camera:** downward-facing is the default and assumed for stabilization.
 - **No expensive drone modifications:** no added sensors/hardware.
 - **Optional low-cost aids are allowed:** printable landing pad, optional external phone camera later.
 - **Goal:** “better than a human pilot” means higher bandwidth and repeatability, quantified via metrics and robust across conditions.
 - **Battery telemetry:** not available; if dynamics drift meaningfully we will re-identify the model periodically rather than relying on battery measurements.
 
 ---
 
 ## 3) Current State (What Exists Today)
 
 In `experimental/e88_autopilot/`:
 
 - **Optical flow:** `LucasKanadeDriftEstimator` produces `vx_px_s`, `vy_px_s`, and `quality`.
 - **Conditioning:** `VelocityMeasurementConditioner` applies a velocity deadband.
 - **Kalman filter:** `VelocityKalman2D` smooths velocity using `sigma_a` and `sigma_v_meas`.
 - **Controller:** `VelocityHoldController` is a PI controller on velocity with saturation at `max_cmd`.
 - **Autostabilizer loop:** `AutoStabilizer` runs flow → conditioning → optional Kalman → PI → `drone.send_cmd`.
 - **Stationary calibration:** estimates `estimator_deadband_px_s` and `kalman_sigma_v` while the drone is still.
 - **Qt UI:** starts/stops autostabilizer and shows some telemetry overlay.
 
 Key tuning clarification:
 
 - “Intensity” is primarily controlled by `kp/ki/deadband/max_cmd`.
 - Kalman sigmas affect filtering/trust/lag and can indirectly reduce aggressiveness if too smooth/laggy.
 
 ---
 
 ## 4) Challenges We Must Solve (and Why They Matter)
 
 ### 4.1 Missing “control-room visibility”
 Without observability, we cannot answer basic debugging questions:
 
 - What did optical flow measure?
 - Was it gated?
 - Did Kalman lag?
 - Was the controller saturated?
 - Did the integrator wind up?
 - Why did it crash instead of aborting?
 
 This blocks safe iteration and autonomous tuning.
 
 ### 4.2 Texture degradation and altitude effects
 As the drone rises, ground features become smaller and flow tracking can degrade. This can yield biased/noisy velocity estimates that a controller will “fight.”
 
 ### 4.3 Scale ambiguity (critical)
 Optical flow measures velocity in **px/s**, not **m/s**. The conversion depends on altitude. A controller tuned at one height will not behave correctly at another.
 
 ### 4.4 Latency budget (critical)
 We do not yet know the effective control loop latency:
 
 - RTSP buffering (frame age)
 - processing time (flow + Kalman)
 - network RTT / command delivery
 
 Delay reduces stability margins and must be measured and explicitly considered.
 
 ### 4.5 Coordinate conventions (critical)
 If `roll_sign` or `pitch_sign` is wrong, the controller becomes positive feedback and destabilizes immediately.
 
 ### 4.6 Attitude-induced flow artifacts
 Rolling/pitching tilts the camera and can introduce apparent flow even without translation. This couples control output back into measurement.
 
 ### 4.7 Control loop rate vs frame rate mismatch
 The system loop is currently constrained by incoming frames and timing jitter. This needs to be fixed by a better threading model.
 
 ### 4.8 Quality metric reliability
 Today gating relies mostly on `quality`. We need to validate whether it predicts measurement reliability, and supplement it with additional health metrics (feature counts, inlier ratio, fallback signals).
 
 ### 4.9 Initialization transients
 Estimators start “cold” and early samples can be garbage. The system must initialize safely.
 
 ### 4.10 Throttle/battery drift
 Battery telemetry is unavailable. If dynamics drift meaningfully over time, we will handle this via periodic re-identification.
 
 ---
 
 ## 5) Key Design Decisions (Reasoning)
 
 ### 5.1 Autonomy-first tuning via System Identification
 We will tune stabilization through autonomous excitation + model fitting rather than human demonstration flights.
 
 - Humans are not the gold standard for bandwidth/precision.
 - System ID yields repeatable and measurable gains and stability margins.
 - This matches “too-fast-for-humans” control domains so lessons learned are transferable.
 
 ### 5.2 Invest early in observability (Phase 0)
 We prioritize measurement and explainability before aggressive tuning.
 
 ### 5.3 Printable landing pad as a controlled visual environment
 A standardized pad reduces environmental variability and addresses early texture issues at minimal cost.
 
 ### 5.4 External camera later, initially for scoring only
 External phone camera integration is deferred until the onboard loop is strong. Initially it is used for **validation/scoring**, not control.
 
 ---
 
 ## 6) Architecture Overview
 
 Conceptual modules:
 
 - **Perception (`OpticalFlowEstimator`)**
   - Input: frames + timestamps
   - Output: velocity estimate + health metrics
 - **Measurement conditioning (`VelocityMeasurementConditioner`)**
   - deadband gating, basic outlier handling
 - **State estimation (`VelocityKalman2D`)**
   - filtered velocity + covariance proxies
 - **Control (`VelocityHoldController`)**
   - PI control, saturation, anti-windup
 - **Safety supervisor (`Supervisor`)**
   - hard rules: abort/land thresholds, gating policies
 - **Recording (`SessionRecorder`)**
   - structured JSONL logs + metadata
 - **Experiment runner (`SystemIDRunner`)**
   - inject bounded perturbations, identify model, propose parameters
 - **Qt control room (UI)**
   - configure, start/stop, diagnostics, apply suggestions, rollback
 
 SOLID application (high-level):
 
 - **Single responsibility:** flow vs filtering vs control vs safety vs logging separated.
 - **Open/closed:** allow swapping estimation/control strategies without rewriting UI.
 - **Dependency inversion:** core logic depends on abstractions (e.g., telemetry sink) rather than Qt.
 
 ---
 
 ## 7) Data & Telemetry (Foundation for Debugging and Learning)
 
 We will use session-based logging:
 
 - `meta.json`: environment notes + configuration snapshot + verification results.
 - `samples.jsonl`: append-only per-loop telemetry.
 
 ### 7.1 Per-sample telemetry fields (core)
 
 The final set will evolve, but it must include:
 
 - **Timing and loop budget**
   - `t_frame_received`
   - `t_flow_start`, `t_flow_end`
   - `t_kf_end`, `t_ctrl_end`, `t_cmd_sent`
   - `dt_flow_ms`, `dt_total_ms`
   - `estimated_frame_age_ms` (based on latency estimate)
   - `estimated_latency_ms`
 - **Raw optical flow**
   - `dx_px`, `dy_px`, `vx_px_s`, `vy_px_s`
 - **Flow health**
   - `quality`
   - `n_features`, `n_tracked`
   - `inlier_ratio` (must be surfaced)
   - `fallback_used` (phase correlation fallback)
 - **Conditioned measurement**
   - `cond_vx_px_s`, `cond_vy_px_s`
   - `cond_gated`
 - **Kalman (enabled by default)**
   - `kf_in_vx_px_s`, `kf_in_vy_px_s`
   - `kf_out_vx_px_s`, `kf_out_vy_px_s`
   - `p_vx`, `p_vy`
 - **Controller internals**
   - `p_term_roll`, `i_term_roll`, `u_raw_roll`, `u_sat_roll`, `sat_roll`
   - same for pitch
   - `integrator_status` (running/frozen/decaying)
 - **Supervisor trace**
   - `supervisor_state`
   - `gating_reason` / `transition_reason`
 - **Commands actually sent**
   - `cmd_roll`, `cmd_pitch`, `cmd_throttle`, `cmd_yaw` (normalized)
   - optionally raw stick bytes
 
 ### 7.2 Why JSONL
 - Append-only and crash resilient.
 - Easy offline analysis and replay.
 - Friendly for ML/control tooling.
 
 ---
 
 ## 8) Safety Supervision (Prevent Crash-Driven “Learning”)
 
 The supervisor enforces safety invariants and ensures failures turn into controlled abort/land events.
 
 Example triggers (thresholds tuned later):
 
 - **Quality collapse:** low quality for N frames → neutral; persistent → land.
 - **Runaway drift:** |pos| exceeds threshold → land.
 - **Saturation persistence:** saturated too long → land.
 - **Timing instability:** dt jitter or frame timeouts → neutral/land.
 - **Integrator windup protection:** freeze/decay/reset policy based on state.
 
 ---
 
 # 9) Phased Implementation Plan (0–5)
 
 Each phase includes:
 
 - **Objectives**
 - **Strategy (what we will do and why)**
 - **Implementation checklist** (implementation-ready)
 - **Deliverables** (what exists in the repo at the end)
 - **Milestones / acceptance criteria**
 - **Definition of done**
 
 ---
 
 ## Phase 0 — Observability + Latency Budget Verification
 
 ### Objectives
 - Make the stabilizer explainable in real time.
 - Verify the control loop budget (CPU, frame timing, network RTT, frame age if possible).
 - Establish the canonical session recording format and diagnostics UI.
 
 ### Strategy
 - Add systematic timing instrumentation.
 - Measure network RTT.
 - Log enough health metrics to diagnose whether failures are due to sensing, timing, or control.
 - Add explicit **sign convention verification** as a gated prerequisite for autonomy.
 
 ### Implementation checklist
 - Add per-stage timestamps and duration fields to telemetry.
 - Compute and log `estimated_latency_ms`.
 - Add frame and loop rate estimation (moving averages).
 - Add network RTT measurement (startup or on-demand) and persist to `meta.json`.
 - Add sign verification mode:
   - flow sign convention check (manual push test)
   - control sign convention check (small commanded roll/pitch pulse and observe response)
   - persist result to `meta.json`
 - Surface additional flow health signals:
   - `n_features`, `n_tracked`, `inlier_ratio`, `fallback_used`
 - Add Qt diagnostics panel with the above fields.
 - Implement session recorder:
   - `meta.json` (config snapshot + environment notes)
   - `samples.jsonl` (telemetry per loop)
 
 ### Deliverables
 - **Qt:** Diagnostics panel visible during autopilot run.
 - **Logging:** Session folder with `meta.json` and `samples.jsonl`.
 - **Telemetry schema:** includes timing + flow health + controller internals placeholders.
 
 ### Milestones / acceptance criteria
 - **M0.1:** Latency fields appear in logs; `dt_flow_ms` and `dt_total_ms` measured.
 - **M0.2:** Network RTT is displayed and recorded in `meta.json`.
 - **M0.3:** Actual frame rate and loop rate are displayed and recorded.
 - **M0.4:** Sign verification mode exists and writes results to `meta.json`.
 - **M0.5:** Flow health metrics are visible and logged (`n_features`, `n_tracked`, `inlier_ratio`, `fallback_used`).
 
 ### Definition of done
 Phase 0 is done when a developer can answer, from UI + logs, at least:
 
 - What is the control loop rate and its jitter?
 - What is processing time vs total loop time?
 - What is the network RTT?
 - Are we operating on stale frames (if measurable)?
 - Are signs correct and verified?
 
 ---
 
 ## Phase 1 — Safety Supervisor + Robust Control Loop Behavior
 
 ### Objectives
 - Stop crash-driven iteration.
 - Ensure degraded sensing/timing conditions cause neutralization and/or landing, with logged reasons.
 - Harden PI control behavior (anti-windup, rate limiting, safe initialization).
 
 ### Strategy
 - Introduce a supervisor state machine controlling whether the controller output is applied.
 - Introduce robust integrator policy and command rate limiting.
 - Add latency-aware gating so delay spikes trigger conservative behavior.
 
 ### Implementation checklist
 - Implement supervisor state machine:
   - `INITIALIZING → OK → DEGRADED → LANDING → STOPPED`
   - log all transitions with reasons
 - Add graceful initialization:
   - command neutral output until estimators stable for N frames
 - Anti-windup:
   - freeze integrator on saturation
   - decay integrator toward zero when gated/bad quality
 - Rate limiting:
   - bound `Δcmd_roll/sec` and `Δcmd_pitch/sec`
 - Latency-aware gating:
   - if `estimated_latency_ms` exceeds threshold → degrade
   - if `estimated_frame_age_ms` exceeds threshold (when measurable) → degrade/gate
 - Implement abort/land triggers:
   - quality collapse
   - saturation persistence
   - timing instability
   - frame timeouts
 - Ensure telemetry includes supervisor state and gating reason.
 
 ### Deliverables
 - **Supervisor:** state machine controlling outputs.
 - **Controller robustness:** anti-windup + rate limiting.
 - **Telemetry:** supervisor state + reasons, integrator status.
 
 ### Milestones / acceptance criteria
 - **M1.1:** Supervisor state visible in Qt and logged per sample.
 - **M1.2:** Under bad quality/timeout conditions, system neutralizes and/or lands instead of diverging.
 - **M1.3:** Anti-windup behavior is visible in logs (integrator freezes/decays appropriately).
 - **M1.4:** Rate limiting is visible in command traces.
 - **M1.5:** Latency spikes trigger a conservative supervisor response.
 
 ### Definition of done
 Phase 1 is done when the stabilizer fails safe (neutral/land) under common failure modes and logs a clear root-cause reason.
 
 ---
 
 ## Phase 2 — Landing Pad + Visual Altimeter + Gain Scheduling
 
 ### Objectives
 - Solve early texture reliability issues using a standardized pad.
 - Solve scale ambiguity by estimating altitude and using altitude-aware control.
 - Make behavior consistent across a range of low altitudes.
 
 ### Strategy
 - Use a printable pad with high-frequency texture, a central marker, and known physical size.
 - Use the pad to estimate altitude (visual altimeter) based on its known physical size.
 - Remove scale ambiguity by either:
   - converting velocity from px/s to m/s (preferred), or
   - gain scheduling using altitude.
 
 ### Implementation checklist
 - Define landing pad specification:
   - recommended physical size
   - texture pattern
   - central marker and known physical size
 - Implement marker detection in downward camera.
 - Implement visual altimeter:
   - compute `marker_size_px`
   - estimate `altitude_est_m` via calibrated scale factor
   - log `altitude_source`, `marker_detected`, `marker_size_px`
 - Implement scale handling:
   - **Approach A (preferred):** convert `vx_px_s`/`vy_px_s` to `vx_m_s`/`vy_m_s`
   - **Approach B:** gain scheduling via `gain_scale_factor`
 - Add scale-aware telemetry fields.
 - Add takeoff/climb policy:
   - conservative gains while altitude changes rapidly
   - transition to normal control only after altitude stabilizes
 
 ### Deliverables
 - **Pad spec:** documented design for printing and use.
 - **Visual altimeter:** `altitude_est_m` available and logged.
 - **Scale-aware control:** either m/s control or gain scheduling implemented.
 
 ### Milestones / acceptance criteria
 - **M2.1:** Pad design is available, printed, and used as the standard test surface.
 - **M2.2:** Marker detection works reliably near hover; `altitude_est_m` is logged.
 - **M2.3:** Scale-aware telemetry exists: `altitude_est_m`, `altitude_source`, and either `vx_m_s/vy_m_s` or `gain_scale_factor`.
 - **M2.4:** Hover over pad at a fixed altitude is stable and repeatable.
 - **M2.5:** Behavior is consistent at multiple altitudes (e.g., 0.3m, 0.5m, 1.0m), avoiding the “too aggressive low / too sluggish high” failure mode.
 
 ### Definition of done
 Phase 2 is done when the system has an operational altitude estimate and a defined method (velocity scaling or gain scheduling) that makes stabilization behavior consistent across altitude.
 
 ---
 
 ## Phase 3 — Autonomous System Identification (Latency- and Scale-Aware)
 
 ### Objectives
 - Identify the drone response from roll/pitch commands to measured velocity.
 - Identify effective delay (including frame age + processing + network).
 - Produce a stable model fit usable for controller synthesis.
 
 ### Strategy
 - Run bounded excitation experiments at a known, stable altitude.
 - Fit a simple interpretable model per axis (FOPDT).
 - Use logged timing to validate identified delay.
 
 ### Implementation checklist
 - Build a Qt-driven experiment runner:
   - settle
   - excite roll axis
   - excite pitch axis
   - stop
 - Excitation types:
   - PRBS or sine sweep
 - Maintain strict bounds:
   - amplitude limits
   - rate limiting
   - supervisor always active
 - Fit per-axis FOPDT model:
 
   ```
   v(s) / u(s) = K * exp(-θ*s) / (τ*s + 1)
   ```
 
   where:
   - `K`: steady-state gain (m/s per command unit)
   - `τ`: time constant
   - `θ`: delay
 
 - Delay estimation:
   - cross-correlation
   - grid search during fitting
 - Exclude from fitting:
   - low quality segments
   - saturated segments
   - missing/timeout segments
 - Save and version model artifacts per session.
 
 ### Deliverables
 - **Qt:** “System ID Experiment” workflow.
 - **Artifacts:** `identified_model.json` stored with the session.
 - **Summary:** model parameters per axis and fit quality metrics.
 
 ### Milestones / acceptance criteria
 - **M3.1:** Experiment runner completes safely under supervisor control.
 - **M3.2:** FOPDT parameters (`K`, `τ`, `θ`) are produced for roll and pitch.
 - **M3.3:** Identified `θ` aligns with measured latency from Phase 0 (same order of magnitude).
 - **M3.4:** Repeated runs on the pad produce similar model parameters.
 
 ### Definition of done
 Phase 3 is done when system identification produces stable, repeatable model parameters and delay estimates usable for controller synthesis.
 
 ---
 
 ## Phase 4 — Controller Synthesis + Constrained Iteration (“Learn Between Flights”)
 
 ### Objectives
 - Compute PI gains from the identified model with conservative robustness.
 - Enable safe, reversible improvement between runs.
 - Track improvements using quantitative metrics.
 
 ### Strategy
 - Use IMC/SIMC-style PI tuning for FOPDT.
 - Start conservative (choose `λ` such that stability is robust under delay).
 - Apply bounded parameter updates and allow rollback.
 
 ### Implementation checklist
 - Compute PI gains from identified `K`, `τ`, `θ` using:
 
   ```
   Kp = τ / (K * (λ + θ))
   Ki = Kp / τ
   ```
 
 - Choose `λ` conservatively:
   - `λ ≈ 2θ` or `λ ≈ 3θ` initially
 - Optional later enhancement:
   - consider Smith predictor if delay is large
 - Constrained update policy:
   - per-iteration change limits (e.g., ±15%)
   - parameter history
   - auto-rollback on regression
 - Score flights consistently:
   - drift RMS / peak drift
   - time saturated
   - gating fraction
   - abort count
 - Qt integration:
   - show current vs suggested gains
   - show confidence/fit quality
   - “Apply suggested” / “Rollback”
 
 ### Deliverables
 - **Controller synthesis:** computed PI gains from model.
 - **Iteration framework:** bounded updates + rollback.
 - **Metrics:** session scoring stored and displayed.
 
 ### Milestones / acceptance criteria
 - **M4.1:** Suggested gains are computed and presented in Qt.
 - **M4.2:** Applying suggested gains improves defined metrics over baseline.
 - **M4.3:** Rollback works and is used if a regression occurs.
 - **M4.4:** Over multiple iterations, gains converge toward stable hover on the pad.
 
 ### Definition of done
 Phase 4 is done when the system can safely improve gains between runs using measurable criteria, without human demonstration flights.
 
 ---
 
 ## Phase 5 — Generalization Beyond the Pad + Optional External Camera Scoring
 
 ### Objectives
 - Maintain stability when pad/marker is unavailable.
 - Improve robustness to changing textures and altitude.
 - Validate onboard sensing against an external reference when needed.
 
 ### Strategy
 - Implement graceful degradation when marker not detected.
 - Add secondary heuristics for altitude when possible.
 - Introduce an external phone camera as a scoring/validation tool (not control loop).
 
 ### Implementation checklist
 - Graceful degradation without marker:
   - use last known altitude with increasing uncertainty
   - widen deadbands
   - reduce `max_cmd`
   - log `altitude_source = "last_known"`
 - Explore feature-based altitude heuristics:
   - average feature size
   - feature density
   - correlation with marker-based altitude when available
 - Attitude artifact mitigation:
   - detect coupling between command and measured flow
   - gate or reduce authority when coupling is strong
 - External phone camera scoring:
   - ingest RTSP/HTTP stream in Qt
   - track drone via color blob / tag
   - compute external drift metric
   - align timestamps for comparison
 - Throttle/battery drift handling:
   - since battery telemetry isn’t available, consider periodic re-identification if drift is significant
 
 ### Deliverables
 - **Generalized behavior:** stabilization works with degraded performance when pad is absent.
 - **External scoring tool:** phone camera drift metric integrated into Qt.
 - **Validation workflow:** compare onboard drift estimates to external drift metric.
 
 ### Milestones / acceptance criteria
 - **M5.1:** Stabilizer remains safe and functional beyond pad (with graceful degradation).
 - **M5.2:** External camera tracking produces a stable drift metric.
 - **M5.3:** Onboard vs external drift correlation analysis identifies failure cases.
 - **M5.4:** Robustness improvements reduce failure cases across varying textures/lighting.
 
 ### Definition of done
 Phase 5 is done when the stabilizer is measurably robust beyond the pad and external scoring can validate onboard sensing reliability.
 
 ---
 
 ## 10) How This Infrastructure Enables Future Capabilities (Bezier Curves, etc.)
 
 This roadmap builds a layered control stack:
 
 - **Inner loop (this project):** stabilize drift (velocity/position proxy).
 - **Outer loop (future):** trajectory tracking (Bezier curves, waypoints).
 
 Why the roadmap enables future capabilities:
 
 - Logging + replay validate trajectory execution offline.
 - Supervisor prevents experimental trajectories from becoming crashes.
 - System ID provides responsiveness models needed for smooth tracking.
 - Qt becomes mission control for both stabilization and higher-level autonomy.
 
 ---
 
 ## 11) One-Line Outcomes per Phase
 
 - **Phase 0:** “I can see and record everything that matters, including latency.”
 - **Phase 1:** “Failures neutralize/land with clear reasons; no crash-driven iteration.”
 - **Phase 2:** “Scale ambiguity is addressed; pad + altimeter yields repeatable sensing and control.”
 - **Phase 3:** “The drone runs its own identification experiments and estimates its dynamics and delay.”
 - **Phase 4:** “It tunes itself safely between runs with bounded updates and rollback.”
 - **Phase 5:** “It generalizes beyond the pad; external camera validates when needed.”
 
 ---
 
 ## 12) Stabilizer v1 — Definition of Done (Project-Level)
 
 Stabilizer v1 is achieved when:
 
 1. **Latency is characterized:** total loop latency is measured and acceptable (target <200ms).
 2. **Signs verified:** coordinate conventions confirmed and recorded.
 3. **Scale-aware:** altitude-corrected velocity or gain scheduling is implemented.
 4. **Safety supervised:** failures land/neutralize instead of crashing.
 5. **Autonomously tuned:** model is identified and gains are derived and iterated safely.
 6. **Observable:** failures are explainable via logs/UI.
 7. **Repeatable:** stable hover over the pad across sessions.
