"""Simulator for the E88 drone and its downward-camera stabilization loop.

The point of this package is to answer one question before a real flight:
*will the autostabilizer hold a stable hover, or will it fight itself?*

It answers that at two levels of fidelity:

- ``e88_sim.sim_drone.SimulatedDrone`` is a drop-in replacement for
  ``turbodrone.Drone``. It renders synthetic downward-camera frames from a
  physically simulated pose, so the **unmodified** ``AutoStabilizer`` -- real
  optical flow, real Kalman, real controller, real threading -- can fly it.
  This runs in real time and validates the whole stack, signs included.

- ``e88_sim.fast_loop`` runs the same physics against an *analytic* flow sensor
  (no rendering). It is hundreds of times faster than real time, which is what
  makes gain sweeps, latency stability margins and Monte-Carlo runs practical.

Both share ``e88_sim.physics.QuadPhysics``, so a conclusion drawn from a sweep
can be re-checked with vision in the loop.
"""

from e88_sim.config import (
    AirframeParams,
    CameraParams,
    EnvironmentParams,
    SimConfig,
    VideoLinkParams,
)
from e88_sim.physics import PhysicsState, QuadPhysics

__all__ = [
    "AirframeParams",
    "CameraParams",
    "EnvironmentParams",
    "PhysicsState",
    "QuadPhysics",
    "SimConfig",
    "VideoLinkParams",
]
