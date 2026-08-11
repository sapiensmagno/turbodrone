import math
import sys
import unittest
from pathlib import Path

import numpy as np

_EXPERIMENTAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXPERIMENTAL_ROOT))

from e88_sim.metrics import DIVERGENCE_POS_M, evaluate, format_report


def synth(*, duration=20.0, dt=0.05, vx=None, vy=None, cmd=0.0, max_cmd=0.9):
    """Build a run from a velocity function, integrating it into position."""
    n = int(duration / dt)
    t = [i * dt for i in range(n)]
    vxs = [(vx(x) if callable(vx) else (vx or 0.0)) for x in t]
    vys = [(vy(x) if callable(vy) else (vy or 0.0)) for x in t]
    xs, ys, px, py = [], [], 0.0, 0.0
    for a, b in zip(vxs, vys):
        px += a * dt
        py += b * dt
        xs.append(px)
        ys.append(py)
    return dict(
        t=t,
        x_m=xs,
        y_m=ys,
        vx_m_s=vxs,
        vy_m_s=vys,
        cmd_roll=[cmd] * n,
        cmd_pitch=[0.0] * n,
        max_cmd=max_cmd,
    )


class TestBasicStatistics(unittest.TestCase):
    def test_perfect_hover_passes(self):
        m = evaluate(**synth(vx=0.0, vy=0.0))
        self.assertEqual(m.verdict, "PASS")
        self.assertAlmostEqual(m.speed_rms_m_s, 0.0, places=9)
        self.assertFalse(m.diverged)

    def test_constant_drift_is_measured(self):
        m = evaluate(**synth(vx=0.05, vy=0.0))
        self.assertAlmostEqual(m.speed_rms_m_s, 0.05, places=6)
        self.assertAlmostEqual(m.pos_drift_rate_m_s, 0.05, delta=0.002)

    def test_runaway_position_is_flagged_as_divergence(self):
        m = evaluate(**synth(vx=1.0))
        self.assertTrue(m.diverged)
        self.assertEqual(m.verdict, "FAIL")
        self.assertIn("position", m.divergence_reason)

    def test_excessive_speed_is_flagged_as_divergence(self):
        # Oscillates fast enough to exceed the speed limit without travelling far.
        m = evaluate(**synth(vx=lambda x: 3.0 * math.sin(20.0 * x)))
        self.assertTrue(m.diverged)
        self.assertIn("speed", m.divergence_reason)

    def test_settle_window_excludes_the_cold_start(self):
        # Wild for the first two seconds, still afterwards.
        m = evaluate(**synth(vx=lambda x: 0.5 if x < 2.0 else 0.0), settle_sec=2.0)
        self.assertAlmostEqual(m.speed_rms_m_s, 0.0, places=6)
        self.assertGreater(m.speed_max_m_s, 0.4, "peak should still cover the whole run")


class TestOscillationDetection(unittest.TestCase):
    def test_buzzing_is_detected_and_fails(self):
        """A loop with too much gain for its latency does not run away, it
        buzzes. That has to be caught, or an unflyable tune reads as fine."""
        m = evaluate(**synth(vx=lambda x: 0.15 * math.sin(2.0 * math.pi * 2.0 * x)))
        self.assertAlmostEqual(m.oscillation_hz, 2.0, delta=0.2)
        self.assertGreater(m.oscillation_fraction, 0.8)
        self.assertEqual(m.verdict, "FAIL")

    def test_steady_drift_is_not_mistaken_for_oscillation(self):
        m = evaluate(**synth(vx=0.05))
        self.assertLess(m.oscillation_fraction, 0.3)

    def test_slow_wander_is_not_counted_as_buzzing(self):
        """0.1 Hz is below the oscillation band: that is drift, not a limit
        cycle, and should not be reported as one."""
        m = evaluate(**synth(vx=lambda x: 0.05 * math.sin(2.0 * math.pi * 0.1 * x)))
        self.assertLess(m.oscillation_fraction, 0.5)


class TestOpenLoopComparison(unittest.TestCase):
    def test_a_controller_that_helps_is_rewarded(self):
        ol = evaluate(**synth(vx=0.20))
        cl = evaluate(**synth(vx=0.05), open_loop=ol)
        self.assertAlmostEqual(cl.speed_ratio_vs_open_loop, 0.25, places=3)
        self.assertEqual(cl.verdict, "PASS")

    def test_a_controller_that_adds_motion_fails_even_if_it_stays_close(self):
        """The failure mode that matters: the drone stays near its start point
        but thrashes to do it, converting measurement noise into motion."""
        ol = evaluate(**synth(vx=0.05))
        cl = evaluate(**synth(vx=lambda x: 0.12 * math.sin(2.0 * math.pi * 1.5 * x)), open_loop=ol)
        self.assertGreater(cl.speed_ratio_vs_open_loop, 1.0)
        self.assertEqual(cl.verdict, "FAIL")
        self.assertTrue(any("more than no control" in r or "band" in r for r in cl.reasons))

    def test_position_improvement_is_reported(self):
        ol = evaluate(**synth(vx=0.20))
        cl = evaluate(**synth(vx=0.05), open_loop=ol)
        self.assertAlmostEqual(cl.pos_ratio_vs_open_loop, 0.25, delta=0.02)
        self.assertEqual(cl.open_loop_pos_max_m, ol.pos_max_m)


class TestCommandStatistics(unittest.TestCase):
    def test_saturation_is_reported(self):
        m = evaluate(**synth(vx=0.0, cmd=0.9, max_cmd=0.9))
        self.assertAlmostEqual(m.saturation_fraction, 1.0, places=6)
        self.assertEqual(m.verdict, "MARGINAL")

    def test_unsaturated_commands_report_zero(self):
        m = evaluate(**synth(vx=0.0, cmd=0.1, max_cmd=0.9))
        self.assertAlmostEqual(m.saturation_fraction, 0.0, places=6)


class TestRobustness(unittest.TestCase):
    def test_empty_run_fails_rather_than_crashing(self):
        m = evaluate(
            t=[], x_m=[], y_m=[], vx_m_s=[], vy_m_s=[], cmd_roll=[], cmd_pitch=[], max_cmd=0.9
        )
        self.assertEqual(m.verdict, "FAIL")
        self.assertTrue(m.diverged)

    def test_non_finite_state_is_treated_as_divergence(self):
        data = synth(vx=0.0)
        data["x_m"][10] = float("nan")
        m = evaluate(**data)
        self.assertTrue(m.diverged)

    def test_flow_error_statistics_are_summarized(self):
        m = evaluate(**synth(vx=0.0), flow_error_px_s=[1.0, -2.0, 3.0, -4.0])
        self.assertAlmostEqual(m.flow_err_rms_px_s, math.sqrt((1 + 4 + 9 + 16) / 4.0), places=6)

    def test_report_renders_without_optional_fields(self):
        text = format_report("scenario", evaluate(**synth(vx=0.0)))
        self.assertIn("scenario", text)
        self.assertIn("speed", text)


if __name__ == "__main__":
    unittest.main()
