"""Scoring a simulated flight.

The autopilot runs a **velocity**-hold loop: it drives measured drift velocity
toward zero and has no position feedback at all. That has a consequence worth
stating plainly, because it decides which numbers mean anything:

    A velocity-hold controller cannot hold position. Any residual velocity
    integrates into position error forever, so position drift is bounded only
    by how well velocity is nulled and for how long.

So the primary score is **speed**, not position. And speed has to be judged
against the right baseline: a controller that leaves the drone drifting at
0.1 m/s is doing well if the untrimmed airframe would have reached 0.4 m/s on
its own, and doing badly if it would have reached 0.05 m/s. Every evaluation
here therefore reports the open-loop comparison, because "the drone moved less
than a metre" is not evidence that the controller helped.

The second thing that matters is *how* the drone moves. A loop with too much
gain for its latency does not diverge dramatically -- it buzzes, converting
measurement noise into motion. That shows up as speed RMS above the open-loop
baseline and as energy in a 0.5-5 Hz band, which is what ``oscillation_*``
measures.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# Verdict thresholds. These are judgements about what counts as a hover worth
# risking a real airframe on, not measurements -- tune them deliberately.
DIVERGENCE_POS_M = 5.0
DIVERGENCE_SPEED_M_S = 1.5
GOOD_SPEED_RMS_M_S = 0.10
MARGINAL_SPEED_RMS_M_S = 0.20
# A controller must beat open loop by at least this factor to count as helping.
REQUIRED_IMPROVEMENT = 0.9
OSCILLATION_BAND_HZ = (0.5, 5.0)
BAD_OSCILLATION_FRACTION = 0.5


@dataclass(frozen=True)
class HoverMetrics:
    duration_sec: float
    n_samples: int

    speed_rms_m_s: float
    speed_p95_m_s: float
    speed_max_m_s: float

    pos_max_m: float
    pos_final_m: float
    pos_drift_rate_m_s: float

    oscillation_hz: float
    oscillation_fraction: float

    cmd_rms: float
    cmd_max: float
    saturation_fraction: float

    diverged: bool
    divergence_reason: str = ""

    # Populated when an open-loop run of the same scenario is supplied.
    open_loop_speed_rms_m_s: Optional[float] = None
    speed_ratio_vs_open_loop: Optional[float] = None
    open_loop_pos_max_m: Optional[float] = None
    pos_ratio_vs_open_loop: Optional[float] = None

    # Populated when estimator output is supplied alongside ground truth.
    flow_err_rms_px_s: Optional[float] = None
    flow_err_p95_px_s: Optional[float] = None

    verdict: str = "UNKNOWN"
    reasons: Tuple[str, ...] = ()

    def to_dict(self) -> Dict:
        return asdict(self)


def _dominant_oscillation(signal: np.ndarray, dt: float) -> Tuple[float, float]:
    """Peak frequency in the oscillation band, and that band's share of power.

    Detrended first: a steadily drifting drone has enormous power at DC, which
    would swamp the buzz we are actually looking for.
    """
    n = int(signal.size)
    if n < 16 or dt <= 0.0:
        return (0.0, 0.0)

    x = np.asarray(signal, dtype=np.float64)
    x = x - x.mean()
    # Remove a linear trend too, so a monotonic run-away is not read as a
    # very-low-frequency oscillation.
    t = np.arange(n, dtype=np.float64)
    slope, intercept = np.polyfit(t, x, 1)
    x = x - (slope * t + intercept)
    x = x * np.hanning(n)

    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(n, d=float(dt))
    total = float(spec[1:].sum())
    if total <= 1e-18:
        return (0.0, 0.0)

    lo, hi = OSCILLATION_BAND_HZ
    band = (freqs >= lo) & (freqs <= hi)
    if not np.any(band):
        return (0.0, 0.0)

    band_power = float(spec[band].sum())
    peak_idx = int(np.argmax(np.where(band, spec, 0.0)))
    return (float(freqs[peak_idx]), float(band_power / total))


def evaluate(
    *,
    t: Sequence[float],
    x_m: Sequence[float],
    y_m: Sequence[float],
    vx_m_s: Sequence[float],
    vy_m_s: Sequence[float],
    cmd_roll: Sequence[float],
    cmd_pitch: Sequence[float],
    max_cmd: float,
    open_loop: Optional["HoverMetrics"] = None,
    flow_error_px_s: Optional[Sequence[float]] = None,
    settle_sec: float = 2.0,
) -> HoverMetrics:
    """Score one run.

    ``settle_sec`` of the start is excluded from the steady-state statistics:
    the estimator initializes cold, the Kalman filter has no state and the
    integrator is empty, so the first couple of seconds say nothing about hover
    quality. Divergence checks still cover the whole run.
    """
    ta = np.asarray(t, dtype=np.float64)
    xa = np.asarray(x_m, dtype=np.float64)
    ya = np.asarray(y_m, dtype=np.float64)
    vxa = np.asarray(vx_m_s, dtype=np.float64)
    vya = np.asarray(vy_m_s, dtype=np.float64)
    cra = np.asarray(cmd_roll, dtype=np.float64)
    cpa = np.asarray(cmd_pitch, dtype=np.float64)

    n = int(min(ta.size, xa.size, ya.size, vxa.size, vya.size, cra.size, cpa.size))
    if n < 2:
        return HoverMetrics(
            duration_sec=0.0,
            n_samples=int(n),
            speed_rms_m_s=0.0,
            speed_p95_m_s=0.0,
            speed_max_m_s=0.0,
            pos_max_m=0.0,
            pos_final_m=0.0,
            pos_drift_rate_m_s=0.0,
            oscillation_hz=0.0,
            oscillation_fraction=0.0,
            cmd_rms=0.0,
            cmd_max=0.0,
            saturation_fraction=0.0,
            diverged=True,
            divergence_reason="insufficient samples",
            verdict="FAIL",
            reasons=("run produced no usable samples",),
        )

    ta, xa, ya, vxa, vya, cra, cpa = (a[:n] for a in (ta, xa, ya, vxa, vya, cra, cpa))
    ta = ta - float(ta[0])
    duration = float(ta[-1])

    radius_all = np.hypot(xa - xa[0], ya - ya[0])
    speed_all = np.hypot(vxa, vya)

    diverged = False
    reason = ""
    if float(np.nanmax(radius_all)) > DIVERGENCE_POS_M:
        diverged, reason = True, f"position exceeded {DIVERGENCE_POS_M:.1f} m"
    elif float(np.nanmax(speed_all)) > DIVERGENCE_SPEED_M_S:
        diverged, reason = True, f"speed exceeded {DIVERGENCE_SPEED_M_S:.1f} m/s"
    elif not np.all(np.isfinite(radius_all)):
        diverged, reason = True, "non-finite state"

    steady = ta >= float(settle_sec)
    if np.count_nonzero(steady) < 8:
        steady = np.ones_like(ta, dtype=bool)

    speed = speed_all[steady]
    radius = radius_all[steady]
    ts = ta[steady]

    dt = float(np.median(np.diff(ts))) if ts.size > 2 else 0.05
    osc_hz, osc_frac = _dominant_oscillation(vxa[steady], dt)
    osc_hz_y, osc_frac_y = _dominant_oscillation(vya[steady], dt)
    if osc_frac_y > osc_frac:
        osc_hz, osc_frac = osc_hz_y, osc_frac_y

    drift_rate = 0.0
    if ts.size > 4 and (ts[-1] - ts[0]) > 1e-6:
        drift_rate = float(np.polyfit(ts, radius, 1)[0])

    cmd_mag = np.hypot(cra[steady], cpa[steady])
    sat_limit = max(1e-9, float(max_cmd)) * 0.99
    saturation = float(np.mean((np.abs(cra[steady]) >= sat_limit) | (np.abs(cpa[steady]) >= sat_limit)))

    flow_rms = flow_p95 = None
    if flow_error_px_s is not None:
        fe = np.abs(np.asarray(flow_error_px_s, dtype=np.float64))
        fe = fe[np.isfinite(fe)]
        if fe.size:
            flow_rms = float(np.sqrt(np.mean(fe**2)))
            flow_p95 = float(np.percentile(fe, 95))

    speed_rms = float(np.sqrt(np.mean(speed**2)))
    pos_max = float(np.max(radius_all))

    ol_rms = None if open_loop is None else float(open_loop.speed_rms_m_s)
    ol_pos = None if open_loop is None else float(open_loop.pos_max_m)
    ratio = float(speed_rms / ol_rms) if (ol_rms is not None and ol_rms > 1e-9) else None
    pos_ratio = float(pos_max / ol_pos) if (ol_pos is not None and ol_pos > 1e-9) else None

    verdict, reasons = _verdict(
        diverged=diverged,
        divergence_reason=reason,
        speed_rms=speed_rms,
        ratio=ratio,
        pos_ratio=pos_ratio,
        oscillation_fraction=float(osc_frac),
        saturation=saturation,
    )

    return HoverMetrics(
        duration_sec=duration,
        n_samples=int(n),
        speed_rms_m_s=speed_rms,
        speed_p95_m_s=float(np.percentile(speed, 95)),
        speed_max_m_s=float(np.max(speed_all)),
        pos_max_m=pos_max,
        pos_final_m=float(radius_all[-1]),
        pos_drift_rate_m_s=drift_rate,
        oscillation_hz=float(osc_hz),
        oscillation_fraction=float(osc_frac),
        cmd_rms=float(np.sqrt(np.mean(cmd_mag**2))) if cmd_mag.size else 0.0,
        cmd_max=float(np.max(cmd_mag)) if cmd_mag.size else 0.0,
        saturation_fraction=saturation,
        diverged=bool(diverged),
        divergence_reason=reason,
        open_loop_speed_rms_m_s=ol_rms,
        speed_ratio_vs_open_loop=ratio,
        open_loop_pos_max_m=ol_pos,
        pos_ratio_vs_open_loop=pos_ratio,
        flow_err_rms_px_s=flow_rms,
        flow_err_p95_px_s=flow_p95,
        verdict=verdict,
        reasons=tuple(reasons),
    )


def _verdict(
    *,
    diverged: bool,
    divergence_reason: str,
    speed_rms: float,
    ratio: Optional[float],
    pos_ratio: Optional[float],
    oscillation_fraction: float,
    saturation: float,
) -> Tuple[str, List[str]]:
    reasons: List[str] = []

    if diverged:
        return ("FAIL", [f"diverged: {divergence_reason}"])

    failed = False
    marginal = False

    if speed_rms > MARGINAL_SPEED_RMS_M_S:
        failed = True
        reasons.append(f"speed RMS {speed_rms:.3f} m/s exceeds {MARGINAL_SPEED_RMS_M_S:.2f}")
    elif speed_rms > GOOD_SPEED_RMS_M_S:
        marginal = True
        reasons.append(f"speed RMS {speed_rms:.3f} m/s above the {GOOD_SPEED_RMS_M_S:.2f} target")

    if ratio is not None:
        if ratio > 1.0:
            failed = True
            reasons.append(
                f"controller made the drone move more than no control at all (speed x{ratio:.2f})"
            )
        elif ratio > REQUIRED_IMPROVEMENT:
            marginal = True
            reasons.append(f"speed barely better than open loop (x{ratio:.2f})")

    if pos_ratio is not None:
        if pos_ratio > 1.0:
            failed = True
            reasons.append(f"drifted further than with no control at all (position x{pos_ratio:.2f})")
        elif pos_ratio < 0.7:
            reasons.append(f"position excursion cut to x{pos_ratio:.2f} of open loop")

    if oscillation_fraction > BAD_OSCILLATION_FRACTION:
        failed = True
        reasons.append(
            f"{100.0 * oscillation_fraction:.0f}% of motion energy is in the "
            f"{OSCILLATION_BAND_HZ[0]}-{OSCILLATION_BAND_HZ[1]} Hz band (buzzing)"
        )

    if saturation > 0.10:
        marginal = True
        reasons.append(f"command saturated {100.0 * saturation:.0f}% of the time")

    if failed:
        return ("FAIL", reasons)
    if marginal:
        return ("MARGINAL", reasons)
    return ("PASS", reasons or ["stable hover"])


def format_report(name: str, m: HoverMetrics) -> str:
    """One-scenario human-readable block."""
    lines = [f"{name}: {m.verdict}"]
    lines.append(
        f"  speed   rms={m.speed_rms_m_s:.3f}  p95={m.speed_p95_m_s:.3f}  max={m.speed_max_m_s:.3f} m/s"
    )
    if m.speed_ratio_vs_open_loop is not None:
        lines.append(
            f"  vs open loop  rms={m.open_loop_speed_rms_m_s:.3f} m/s  ->  x{m.speed_ratio_vs_open_loop:.2f}"
        )
    pos_line = f"  position  max={m.pos_max_m:.3f}  final={m.pos_final_m:.3f} m  drift={m.pos_drift_rate_m_s:+.3f} m/s"
    if m.pos_ratio_vs_open_loop is not None:
        pos_line += f"  (open loop max={m.open_loop_pos_max_m:.3f} -> x{m.pos_ratio_vs_open_loop:.2f})"
    lines.append(pos_line)
    lines.append(
        f"  command   rms={m.cmd_rms:.3f}  max={m.cmd_max:.3f}  saturated={100.0 * m.saturation_fraction:.0f}%"
    )
    lines.append(
        f"  oscillation  {m.oscillation_hz:.2f} Hz carrying {100.0 * m.oscillation_fraction:.0f}% of the energy"
    )
    if m.flow_err_rms_px_s is not None:
        lines.append(
            f"  estimator vs truth  rms={m.flow_err_rms_px_s:.1f}  p95={m.flow_err_p95_px_s:.1f} px/s"
        )
    for r in m.reasons:
        lines.append(f"  - {r}")
    return "\n".join(lines)
