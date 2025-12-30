from __future__ import annotations

import argparse
import threading
import time

from e88_autopilot.autostabilizer import AutoStabilizer, StabilizerConfig
from turbodrone import Drone


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="E88")
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--no-interactive", action="store_true")

    parser.add_argument("--enable-takeoff", action="store_true")
    parser.add_argument("--takeoff-throttle", type=float, default=None)
    parser.add_argument("--takeoff-duration", type=float, default=None)
    parser.add_argument("--climb-throttle", type=float, default=None)
    parser.add_argument("--climb-duration", type=float, default=None)
    parser.add_argument("--settle-good-frames", type=int, default=None)

    parser.add_argument("--base-throttle", type=float, default=None)
    parser.add_argument("--rate-hz", type=float, default=None)

    parser.add_argument("--no-kalman", action="store_true")
    parser.add_argument("--sigma-a", type=float, default=None)
    parser.add_argument("--sigma-v", type=float, default=None)

    parser.add_argument("--min-quality", type=float, default=None)
    parser.add_argument("--max-cmd", type=float, default=None)

    parser.add_argument("--kp-vx", type=float, default=None)
    parser.add_argument("--kp-vy", type=float, default=None)
    parser.add_argument("--ki-vx", type=float, default=None)
    parser.add_argument("--ki-vy", type=float, default=None)
    parser.add_argument("--deadband", type=float, default=None)
    parser.add_argument("--roll-sign", type=float, default=None)
    parser.add_argument("--pitch-sign", type=float, default=None)
    args = parser.parse_args()

    drone = Drone(protocol=args.protocol)
    drone.connect()

    if args.camera is not None:
        drone.switch_camera(args.camera)

    d = StabilizerConfig()
    cfg = StabilizerConfig(
        enable_takeoff=bool(args.enable_takeoff),
        takeoff_throttle=float(d.takeoff_throttle if args.takeoff_throttle is None else args.takeoff_throttle),
        takeoff_duration_sec=float(d.takeoff_duration_sec if args.takeoff_duration is None else args.takeoff_duration),
        climb_throttle=float(d.climb_throttle if args.climb_throttle is None else args.climb_throttle),
        climb_duration_sec=float(d.climb_duration_sec if args.climb_duration is None else args.climb_duration),
        settle_good_frames=int(d.settle_good_frames if args.settle_good_frames is None else args.settle_good_frames),
        base_throttle=float(d.base_throttle if args.base_throttle is None else args.base_throttle),
        cmd_rate_hz=float(d.cmd_rate_hz if args.rate_hz is None else args.rate_hz),
        use_kalman=(False if bool(args.no_kalman) else bool(d.use_kalman)),
        kalman_sigma_a=float(d.kalman_sigma_a if args.sigma_a is None else args.sigma_a),
        kalman_sigma_v=float(d.kalman_sigma_v if args.sigma_v is None else args.sigma_v),
        min_quality=float(d.min_quality if args.min_quality is None else args.min_quality),
        max_cmd=float(d.max_cmd if args.max_cmd is None else args.max_cmd),
        kp_vx=float(d.kp_vx if args.kp_vx is None else args.kp_vx),
        kp_vy=float(d.kp_vy if args.kp_vy is None else args.kp_vy),
        ki_vx=float(d.ki_vx if args.ki_vx is None else args.ki_vx),
        ki_vy=float(d.ki_vy if args.ki_vy is None else args.ki_vy),
        deadband_px_s=float(d.deadband_px_s if args.deadband is None else args.deadband),
        roll_sign=float(d.roll_sign if args.roll_sign is None else args.roll_sign),
        pitch_sign=float(d.pitch_sign if args.pitch_sign is None else args.pitch_sign),
    )

    stabilizer = AutoStabilizer(drone, cfg=cfg)
    stabilizer.activate()

    try:
        if args.duration is not None or args.no_interactive:
            stabilizer.run(duration_sec=args.duration)
        else:
            th = threading.Thread(target=stabilizer.run, kwargs={"duration_sec": None}, daemon=True)
            th.start()
            input("Press Enter to land and exit...\n")
            stabilizer.request_stop()
            t0 = time.monotonic()
            while (time.monotonic() - t0) < 0.6:
                drone.land()
                time.sleep(0.05)
            th.join(timeout=3.0)
    finally:
        drone.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
