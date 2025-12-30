# Real-drone incremental test checklist

## Safety preflight

1. Have an immediate way to land/cut throttle (your normal controller/script).
2. Test in a clear area.
3. Start with conservative values:

- `--max-cmd 0.15`
- `--min-quality 0.20`
- keep `--duration` short initially

If the drone behaves unexpectedly, stop the stabilizer and land.

## Step 0: Confirm you have frames

Run from the `experimental/` folder.

- `python -c "import sys; from pathlib import Path; sys.path.insert(0, str(Path('..').resolve()/'experimental')); from turbodrone import Drone; d=Drone('E88'); d.connect(); f=d.get_frame(timeout=5.0); print(None if f is None else f.shape); d.close()"`

If this prints `None`, fix video/networking first.

## Step 1: Optical flow probe (no control)

This validates LK quality, effective FPS, and texture starvation behavior.

- `python -m e88_autopilot.probe_optical_flow --camera 2 --print-hz 5`

Optional window:

- `python -m e88_autopilot.probe_optical_flow --camera 2 --show`

What to look for:

- When the drone is **still on the ground**: `dx/dy` should hover around 0 and `q` should be stable.
- Slowly move the drone by hand over textured ground: `dx/dy` should follow the motion.
- If you point to featureless surfaces (plain carpet / glossy floor): `tracked` will drop and `q` will fall.

## Step 2: Add Kalman smoothing (still no control)

- `python -m e88_autopilot.probe_optical_flow --camera 2 --kalman --sigma-a 25 --sigma-v 60`

You should see `kvx/kvy` less noisy than `vx/vy`.

Tuning hints:

- Increase `--sigma-v` if the estimator is too “trusting” of noisy LK.
- Increase `--sigma-a` if you want the filter to adapt faster to real changes (but it will be noisier).

## Step 3: Stabilizer in “hold” mode (no takeoff)

This step is only practical if the drone will spin motors while held. If yours will not, skip to Step 4.

Expected:

- With good texture, it will generate small `roll/pitch` corrections.
- If `q` drops below threshold, it should output neutral roll/pitch.

If commands oscillate:

- Reduce `--max-cmd`
- Increase `--min-quality`
- Reduce controller gains (next step once we expose knobs)

## Step 4: Hover hold (manual takeoff)

If manual takeoff makes it hard to start the program at the right time, use Step 5 which includes an initial boost.

Suggested:

- `python -m e88_autopilot.run_autostabilizer --camera 2 --duration 15 --max-cmd 0.15 --min-quality 0.25 --base-throttle 50`

If it drifts in the *wrong* direction:

- Swap the signs in `VelocityHoldController` (`roll_sign` / `pitch_sign`). Different camera orientations can invert axes.

## Step 5: Optional takeoff assistance (use carefully)

- `python -m e88_autopilot.run_autostabilizer --camera 2 --enable-takeoff --takeoff-throttle 100 --takeoff-duration 0.5 --climb-throttle 70 --climb-duration 1.5 --settle-good-frames 5 --duration 15 --max-cmd 0.15 --min-quality 0.25`

Note: This does **not** guarantee 1m altitude; it’s a timed takeoff/throttle bump only.

If the motors spin briefly then stop without taking off:

- Increase `--climb-throttle` (e.g. 75-85)
- Increase `--climb-duration` (e.g. 2.0)
- Ensure `--duration` is long enough (it starts after takeoff/settle now)

## Axis/sign sanity check

From your probe results:

- left is negative, right is positive
- front is positive, back is negative

In the stabilizer, **x drift** drives `roll` and **y drift** drives `pitch`.

If it corrects in the wrong direction, flip one sign at a time:

- `--roll-sign 1` (or `-1`)
- `--pitch-sign 1` (or `-1`)

## Known limitations / next engineering steps

- No altitude estimate: “go up 1m” cannot be guaranteed without additional sensing.
- Camera latency: timestamping is monotonic on frame arrival, not true sensor time.
- Need better grounded/flying detection: could use optical-flow magnitude + motor spin-up heuristics.
- Need texture starvation handling: should add a “hold last command” vs “neutral” policy and explicit fallback to manual.
