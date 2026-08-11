"""Pre-flight simulation runner.

    python -m e88_sim.run_sim --suite preflight
    python -m e88_sim.run_sim --suite gain --json gains.json
    python -m e88_sim.run_sim --suite preflight --vision --scenario nominal_hover

By default every scenario runs on the fast analytic-sensor loop, several
hundred times faster than real time, and each is scored against an open-loop
run of the *same* scenario with the same seed -- because "it only drifted a
metre" means nothing until you know what it would have done untouched.

``--vision`` instead flies the unmodified ``AutoStabilizer`` against rendered
frames and the real optical-flow stack. That runs at real time and is the
authoritative check: use the fast suite to find the interesting configurations,
then confirm the one you intend to fly.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

_EXPERIMENTAL_ROOT = Path(__file__).resolve().parents[1]
if str(_EXPERIMENTAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTAL_ROOT))

from e88_sim.fast_loop import run_fast
from e88_sim.metrics import HoverMetrics, evaluate, format_report
from e88_sim.scenarios import SUITES, Scenario, default_stabilizer_config


def run_scenario_fast(scenario: Scenario, seed: int) -> HoverMetrics:
    sim = scenario.sim.with_(seed=int(seed))
    open_loop = run_fast(
        sim_cfg=sim, stab_cfg=scenario.stab, duration_sec=scenario.duration_sec, control=False, seed=seed
    ).metrics(max_cmd=float(scenario.stab.max_cmd), settle_sec=float(scenario.settle_sec))
    closed = run_fast(
        sim_cfg=sim, stab_cfg=scenario.stab, duration_sec=scenario.duration_sec, control=True, seed=seed
    )
    return closed.metrics(
        max_cmd=float(scenario.stab.max_cmd), open_loop=open_loop, settle_sec=float(scenario.settle_sec)
    )


def run_scenario_vision(scenario: Scenario, seed: int) -> HoverMetrics:
    """Fly the unmodified AutoStabilizer against rendered frames, in real time.

    The open-loop baseline is produced by the fast loop rather than by a second
    real-time flight: it needs no perception at all, so rendering it would cost
    another ``duration_sec`` of wall clock for an identical answer.
    """
    from e88_autopilot.autostabilizer import AutoStabilizer
    from e88_sim.sim_drone import SimulatedDrone

    sim = scenario.sim.with_(seed=int(seed))
    open_loop = run_fast(
        sim_cfg=sim, stab_cfg=scenario.stab, duration_sec=scenario.duration_sec, control=False, seed=seed
    ).metrics(max_cmd=float(scenario.stab.max_cmd), settle_sec=float(scenario.settle_sec))

    drone = SimulatedDrone(sim)
    drone.connect()
    # Let the video pipeline fill before the estimator starts, so the run is not
    # scored on its cold start.
    time.sleep(0.5)
    telemetry: List = []
    try:
        stabilizer = AutoStabilizer(drone, cfg=scenario.stab, telemetry_sink=telemetry.append)
        stabilizer.run(duration_sec=float(scenario.duration_sec))
        truth = drone.truth
    finally:
        drone.close()

    if not truth:
        raise RuntimeError("vision run produced no telemetry")

    metrics = evaluate(
        t=[s.t for s in truth],
        x_m=[s.x_m for s in truth],
        y_m=[s.y_m for s in truth],
        vx_m_s=[s.vx_m_s for s in truth],
        vy_m_s=[s.vy_m_s for s in truth],
        cmd_roll=[s.cmd_roll for s in truth],
        cmd_pitch=[s.cmd_pitch for s in truth],
        max_cmd=float(scenario.stab.max_cmd),
        open_loop=open_loop,
        settle_sec=float(scenario.settle_sec),
    )

    loop_rates = [t.loop_rate_hz for t in telemetry if t.loop_rate_hz > 0.0]
    if loop_rates:
        achieved = statistics.median(loop_rates)
        target = float(scenario.stab.cmd_rate_hz)
        if achieved < 0.9 * target:
            print(
                f"    note: control loop achieved {achieved:.1f} Hz against a {target:.0f} Hz target. "
                f"Rendering competes with the loop for CPU in this process; the real drone "
                f"reaches ~20 Hz. Treat this run as slightly slower than reality."
            )
    return metrics


def summarize(results: Dict[str, List[HoverMetrics]], scenarios: Sequence[Scenario]) -> int:
    """Print the report. Returns a process exit code."""
    by_name = {s.name: s for s in scenarios}
    unexpected: List[str] = []
    silent_falsifiers: List[str] = []

    print()
    print("=" * 78)
    print("RESULTS")
    print("=" * 78)

    for name, runs in results.items():
        scenario = by_name[name]
        verdicts = [m.verdict for m in runs]
        worst = "FAIL" if "FAIL" in verdicts else ("MARGINAL" if "MARGINAL" in verdicts else "PASS")

        speeds = [m.speed_rms_m_s for m in runs]
        ratios = [m.speed_ratio_vs_open_loop for m in runs if m.speed_ratio_vs_open_loop is not None]
        pos = [m.pos_max_m for m in runs]
        pos_ratios = [m.pos_ratio_vs_open_loop for m in runs if m.pos_ratio_vs_open_loop is not None]

        tag = worst
        if scenario.expect_failure:
            if worst == "FAIL":
                tag = "FAIL (expected)"
            else:
                tag = f"{worst} -- EXPECTED FAILURE DID NOT OCCUR"
                silent_falsifiers.append(name)
        elif worst == "FAIL":
            unexpected.append(name)

        print()
        print(f"[{tag}] {name}   ({len(runs)} seed{'s' if len(runs) != 1 else ''})")
        print(f"    {scenario.question}")
        print(
            f"    speed rms {min(speeds):.3f}-{max(speeds):.3f} m/s"
            + (f"   vs open loop x{min(ratios):.2f}-{max(ratios):.2f}" if ratios else "")
        )
        print(
            f"    position max {min(pos):.3f}-{max(pos):.3f} m"
            + (f"   vs open loop x{min(pos_ratios):.2f}-{max(pos_ratios):.2f}" if pos_ratios else "")
        )
        seen = []
        for m in runs:
            for r in m.reasons:
                if r not in seen:
                    seen.append(r)
        for r in seen[:4]:
            print(f"    - {r}")

    print()
    print("=" * 78)
    if silent_falsifiers:
        print("SIMULATOR NOT DISCRIMINATING")
        print(
            "  These scenarios are designed to fail and did not: "
            + ", ".join(silent_falsifiers)
            + "\n  Until that is explained, treat the other verdicts as unproven."
        )
        return 3
    if unexpected:
        print("NO-GO")
        print("  Failing scenarios: " + ", ".join(unexpected))
        print("  Do not fly this configuration.")
        return 1

    marginal = [n for n, runs in results.items() if any(m.verdict == "MARGINAL" for m in runs)]
    if marginal:
        print("GO WITH CAUTION")
        print("  Marginal scenarios: " + ", ".join(marginal))
        return 0

    print("GO -- every scenario passed, and every falsification scenario failed as designed.")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", default="preflight", choices=sorted(SUITES), help="which scenario set to run")
    parser.add_argument("--scenario", action="append", help="run only these scenarios (repeatable)")
    parser.add_argument("--vision", action="store_true", help="fly the real AutoStabilizer on rendered frames (real time)")
    parser.add_argument("--duration", type=float, default=None, help="override scenario duration, seconds")
    parser.add_argument("--seeds", type=int, default=None, help="override the number of seeds per scenario")
    parser.add_argument("--kp", type=float, default=None, help="override kp_vx/kp_vy")
    parser.add_argument("--ki", type=float, default=None, help="override ki_vx/ki_vy")
    parser.add_argument("--max-cmd", type=float, default=None, help="override max_cmd")
    parser.add_argument("--motion-model", default=None, help="override flow_motion_model")
    parser.add_argument("--json", type=Path, default=None, help="write full metrics to this JSON file")
    parser.add_argument("--verbose", action="store_true", help="print the full per-run report")
    args = parser.parse_args(argv)

    overrides = {}
    if args.kp is not None:
        overrides.update(kp_vx=args.kp, kp_vy=args.kp)
    if args.ki is not None:
        overrides.update(ki_vx=args.ki, ki_vy=args.ki)
    if args.max_cmd is not None:
        overrides.update(max_cmd=args.max_cmd)
    if args.motion_model is not None:
        overrides.update(flow_motion_model=args.motion_model)

    stab = default_stabilizer_config(**overrides)
    scenarios = SUITES[args.suite](stab=stab)
    if args.scenario:
        wanted = set(args.scenario)
        scenarios = [s for s in scenarios if s.name in wanted]
        if not scenarios:
            parser.error(f"no scenario matched {sorted(wanted)}")

    runner = run_scenario_vision if args.vision else run_scenario_fast
    mode = "vision-in-the-loop (real time)" if args.vision else "analytic sensor (fast)"
    print(f"suite={args.suite}  mode={mode}  scenarios={len(scenarios)}")
    if overrides:
        print(f"overrides: {overrides}")

    results: Dict[str, List[HoverMetrics]] = {}
    started = time.time()
    for scenario in scenarios:
        if args.duration is not None:
            scenario = Scenario(**{**asdict_shallow(scenario), "duration_sec": float(args.duration)})
        seeds = list(scenario.seeds)
        if args.seeds is not None:
            seeds = list(range(1, int(args.seeds) + 1))
        if args.vision:
            # Real time: one seed unless the caller insists.
            seeds = seeds[: (int(args.seeds) if args.seeds is not None else 1)]

        print(f"  running {scenario.name} ({len(seeds)} seed(s), {scenario.duration_sec:.0f}s each)...", flush=True)
        runs = []
        for seed in seeds:
            m = runner(scenario, seed)
            runs.append(m)
            if args.verbose:
                print(format_report(f"    {scenario.name} seed={seed}", m))
        results[scenario.name] = runs

    print(f"\ncompleted in {time.time() - started:.1f} s wall clock")
    code = summarize(results, scenarios)

    if args.json:
        payload = {
            "suite": args.suite,
            "mode": "vision" if args.vision else "fast",
            "overrides": overrides,
            "results": {name: [m.to_dict() for m in runs] for name, runs in results.items()},
        }
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.json}")

    return code


def asdict_shallow(scenario: Scenario) -> dict:
    """Shallow field copy. ``dataclasses.asdict`` would recurse into the config
    objects and turn them into plain dicts, which the Scenario cannot rebuild."""
    return {f: getattr(scenario, f) for f in scenario.__dataclass_fields__}


if __name__ == "__main__":
    raise SystemExit(main())
