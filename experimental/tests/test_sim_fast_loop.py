"""The fast loop is a second implementation of the autopilot's hold loop.

That duplication is the most dangerous thing in this package: if it drifts away
from ``AutoStabilizer.run``, gain sweeps produce confident, wrong advice. These
tests pin it down -- against the real loop flown on rendered frames, against
theory, and against scenarios whose answers are known in advance.
"""

import sys
import time
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

_EXPERIMENTAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXPERIMENTAL_ROOT))

from e88_autopilot.autostabilizer import AutoStabilizer
from e88_sim.config import AirframeParams, EnvironmentParams, GroundParams, SimConfig, VideoLinkParams
from e88_sim.fast_loop import run_fast
from e88_sim.metrics import evaluate
from e88_sim.scenarios import default_stabilizer_config

# The gain the simulator recommends, not the one currently shipped: at
# kp = 0.003 the loop sits past its stability knee, so a comparison there would
# be a comparison of two different chaotic trajectories rather than of two
# implementations. Agreement is only a meaningful claim in the stable regime.
TUNED = default_stabilizer_config(kp_vx=0.0012, kp_vy=0.0012, ki_vx=0.0002, ki_vy=0.0002)


def _sim(seed=1, altitude=0.7, **kwargs) -> SimConfig:
    env = kwargs.pop("environment", EnvironmentParams())
    return SimConfig(environment=replace(env, initial_altitude_m=altitude), seed=seed, **kwargs)


def _metrics(log, stab, open_loop=None):
    return log.metrics(max_cmd=float(stab.max_cmd), open_loop=open_loop)


class TestFastLoopBehaviour(unittest.TestCase):
    def test_control_reduces_position_drift_versus_open_loop(self):
        cfg = _sim(seed=2)
        ol = _metrics(run_fast(sim_cfg=cfg, stab_cfg=TUNED, duration_sec=15.0, control=False), TUNED)
        cl = _metrics(run_fast(sim_cfg=cfg, stab_cfg=TUNED, duration_sec=15.0, control=True), TUNED, ol)
        self.assertLess(cl.pos_max_m, 0.6 * ol.pos_max_m)
        self.assertFalse(cl.diverged)

    def test_inverted_signs_diverge(self):
        """The falsification the whole exercise rests on. If a simulator cannot
        tell a positive-feedback loop from a working one, its PASS means
        nothing."""
        cfg = _sim(seed=1)
        bad = replace(TUNED, roll_sign=+1.0, pitch_sign=+1.0)
        m = _metrics(run_fast(sim_cfg=cfg, stab_cfg=bad, duration_sec=15.0), bad)
        self.assertTrue(m.diverged)
        self.assertEqual(m.verdict, "FAIL")

    def test_one_inverted_axis_diverges(self):
        cfg = _sim(seed=1)
        bad = replace(TUNED, roll_sign=+1.0)
        m = _metrics(run_fast(sim_cfg=cfg, stab_cfg=bad, duration_sec=15.0), bad)
        self.assertTrue(m.diverged)

    def test_excessive_gain_destabilizes(self):
        cfg = _sim(seed=1)
        hot = replace(TUNED, kp_vx=0.015, kp_vy=0.015, ki_vx=0.0025, ki_vy=0.0025)
        ol = _metrics(run_fast(sim_cfg=cfg, stab_cfg=hot, duration_sec=15.0, control=False), hot)
        m = _metrics(run_fast(sim_cfg=cfg, stab_cfg=hot, duration_sec=15.0), hot, ol)
        self.assertEqual(m.verdict, "FAIL")
        self.assertGreater(m.speed_rms_m_s, 3.0 * ol.speed_rms_m_s)

    def test_stability_knee_lies_between_the_tuned_and_shipped_gains(self):
        """The headline finding, asserted so a future change cannot erase it
        silently: kp = 0.0012 is stable and kp = 0.003 is not."""
        cfg = _sim(seed=1)
        results = {}
        for kp in (0.0012, 0.003):
            stab = replace(TUNED, kp_vx=kp, kp_vy=kp, ki_vx=kp / 6.0, ki_vy=kp / 6.0)
            ol = _metrics(run_fast(sim_cfg=cfg, stab_cfg=stab, duration_sec=20.0, control=False), stab)
            results[kp] = _metrics(run_fast(sim_cfg=cfg, stab_cfg=stab, duration_sec=20.0), stab, ol)

        self.assertLess(results[0.0012].speed_ratio_vs_open_loop, 1.0, "tuned gain should beat open loop")
        self.assertGreater(results[0.003].speed_ratio_vs_open_loop, 1.2, "shipped gain should be past the knee")

    def test_determinism(self):
        cfg = _sim(seed=9)
        a = run_fast(sim_cfg=cfg, stab_cfg=TUNED, duration_sec=8.0)
        b = run_fast(sim_cfg=cfg, stab_cfg=TUNED, duration_sec=8.0)
        self.assertEqual(a.x_m, b.x_m)
        self.assertEqual(a.cmd_roll, b.cmd_roll)

    def test_different_seeds_give_different_trajectories(self):
        a = run_fast(sim_cfg=_sim(seed=1), stab_cfg=TUNED, duration_sec=8.0)
        b = run_fast(sim_cfg=_sim(seed=2), stab_cfg=TUNED, duration_sec=8.0)
        self.assertNotEqual(a.x_m[-1], b.x_m[-1])

    def test_frozen_feed_neutralizes_rather_than_latching(self):
        """A frozen feed must not leave a nonzero correction latched: the E88
        re-transmits the last RC state at 33 Hz, so a latched command flies the
        aircraft away."""
        cfg = _sim(seed=1, video=VideoLinkParams(freeze_start_sec=5.0, freeze_duration_sec=4.0))
        log = run_fast(sim_cfg=cfg, stab_cfg=TUNED, duration_sec=14.0)
        m = _metrics(log, TUNED)
        self.assertFalse(m.diverged, "a frozen feed should not run the drone away")

    def test_records_truth_alongside_every_estimate(self):
        log = run_fast(sim_cfg=_sim(seed=1), stab_cfg=TUNED, duration_sec=6.0)
        self.assertGreater(len(log), 50)
        self.assertEqual(len(log.meas_vx_px_s), len(log.truth_vx_px_s))
        errors = np.abs(log.flow_error_px_s())
        self.assertTrue(np.all(np.isfinite(errors)))


class TestSurrogateAgreesWithTheRealLoop(unittest.TestCase):
    """Fly the same scenario twice: once through the analytic sensor, once
    through the unmodified ``AutoStabilizer`` on rendered frames.

    These run at real time, so there are few of them and they are short.
    Agreement is asserted loosely -- the two use different flow estimators and
    different clocks, so trajectories cannot match sample for sample. What must
    match is the conclusion: same order of drift, same stability verdict.
    """

    DURATION = 12.0

    def _vision(self, cfg, stab):
        from e88_sim.sim_drone import SimulatedDrone

        drone = SimulatedDrone(cfg)
        drone.connect()
        try:
            time.sleep(0.5)
            AutoStabilizer(drone, cfg=stab).run(duration_sec=self.DURATION)
            truth = drone.truth
        finally:
            drone.close()

        self.assertTrue(truth, "vision run produced no telemetry")
        return evaluate(
            t=[s.t for s in truth],
            x_m=[s.x_m for s in truth],
            y_m=[s.y_m for s in truth],
            vx_m_s=[s.vx_m_s for s in truth],
            vy_m_s=[s.vy_m_s for s in truth],
            cmd_roll=[s.cmd_roll for s in truth],
            cmd_pitch=[s.cmd_pitch for s in truth],
            max_cmd=float(stab.max_cmd),
        )

    def test_tuned_gain_agrees_on_a_stable_hover(self):
        cfg = _sim(seed=5)
        fast = _metrics(run_fast(sim_cfg=cfg, stab_cfg=TUNED, duration_sec=self.DURATION), TUNED)
        vision = self._vision(cfg, TUNED)

        self.assertFalse(fast.diverged)
        self.assertFalse(vision.diverged)
        # Within a factor of two on drift speed is all that can be claimed, and
        # all that a screening tool needs.
        self.assertLess(max(fast.speed_rms_m_s, vision.speed_rms_m_s), 2.0 * min(fast.speed_rms_m_s, vision.speed_rms_m_s))
        self.assertLess(vision.speed_rms_m_s, 0.25)
        self.assertLess(fast.speed_rms_m_s, 0.25)

    def test_both_agree_that_inverted_signs_diverge(self):
        cfg = _sim(seed=1)
        bad = replace(TUNED, roll_sign=+1.0, pitch_sign=+1.0)
        fast = _metrics(run_fast(sim_cfg=cfg, stab_cfg=bad, duration_sec=self.DURATION), bad)
        vision = self._vision(cfg, bad)
        self.assertTrue(fast.diverged)
        self.assertTrue(vision.diverged)


if __name__ == "__main__":
    unittest.main()
