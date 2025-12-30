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
    parser.add_argument("--takeoff-throttle", type=float, default=100.0)
    parser.add_argument("--takeoff-duration", type=float, default=0.5)
    parser.add_argument("--climb-throttle", type=float, default=70.0)
    parser.add_argument("--climb-duration", type=float, default=1.5)
    parser.add_argument("--settle-good-frames", type=int, default=5)

    parser.add_argument("--base-throttle", type=float, default=50.0)
    parser.add_argument("--rate-hz", type=float, default=20.0)

    parser.add_argument("--no-kalman", action="store_true")
    parser.add_argument("--sigma-a", type=float, default=25.0)
    parser.add_argument("--sigma-v", type=float, default=60.0)

    parser.add_argument("--min-quality", type=float, default=0.15)
    parser.add_argument("--max-cmd", type=float, default=0.35)

    parser.add_argument("--kp-vx", type=float, default=0.003)
    parser.add_argument("--kp-vy", type=float, default=0.003)
    parser.add_argument("--ki-vx", type=float, default=0.0005)
    parser.add_argument("--ki-vy", type=float, default=0.0005)
    parser.add_argument("--deadband", type=float, default=3.0)
    parser.add_argument("--roll-sign", type=float, default=-1.0)
    parser.add_argument("--pitch-sign", type=float, default=-1.0)
    args = parser.parse_args()

    drone = Drone(protocol=args.protocol)
    drone.connect()

    if args.camera is not None:
        drone.switch_camera(args.camera)

    cfg = StabilizerConfig(
        enable_takeoff=bool(args.enable_takeoff),
        takeoff_throttle=float(args.takeoff_throttle),
        takeoff_duration_sec=float(args.takeoff_duration),
        climb_throttle=float(args.climb_throttle),
        climb_duration_sec=float(args.climb_duration),
        settle_good_frames=int(args.settle_good_frames),
        base_throttle=float(args.base_throttle),
        cmd_rate_hz=float(args.rate_hz),
        use_kalman=not bool(args.no_kalman),
        kalman_sigma_a=float(args.sigma_a),
        kalman_sigma_v=float(args.sigma_v),
        min_quality=float(args.min_quality),
        max_cmd=float(args.max_cmd),
        kp_vx=float(args.kp_vx),
        kp_vy=float(args.kp_vy),
        ki_vx=float(args.ki_vx),
        ki_vy=float(args.ki_vy),
        deadband_px_s=float(args.deadband),
        roll_sign=float(args.roll_sign),
        pitch_sign=float(args.pitch_sign),
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
