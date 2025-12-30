from __future__ import annotations

import argparse
import time

import cv2

from e88_autopilot.kalman import VelocityKalman2D
from e88_autopilot.optical_flow import LucasKanadeDriftEstimator
from turbodrone import Drone


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="E88")
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--print-hz", type=float, default=5.0)
    parser.add_argument("--kalman", action="store_true")
    parser.add_argument("--sigma-a", type=float, default=25.0)
    parser.add_argument("--sigma-v", type=float, default=60.0)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    drone = Drone(protocol=args.protocol)
    drone.connect()

    if args.camera is not None:
        drone.switch_camera(args.camera)

    estimator = LucasKanadeDriftEstimator()
    kf = None if not args.kalman else VelocityKalman2D(sigma_a=args.sigma_a, sigma_v_meas=args.sigma_v)

    last_print = 0.0
    last_ts = None
    fps_ema = None

    try:
        while True:
            item = drone.get_frame_with_timestamp(timeout=args.timeout)
            if item is None:
                raise RuntimeError("No frame received")
            frame, ts = item

            if last_ts is not None:
                dt = ts - last_ts
                if dt > 1e-6:
                    fps = 1.0 / dt
                    fps_ema = fps if fps_ema is None else (0.9 * fps_ema + 0.1 * fps)
            last_ts = ts

            est = estimator.update(frame, timestamp=ts)

            k_est = None
            if est is not None and kf is not None:
                k_est = kf.update_velocity(t=ts, vx_px_s=est.vx_px_s, vy_px_s=est.vy_px_s, quality=est.quality)

            now = time.monotonic()
            if est is not None and (now - last_print) >= (1.0 / max(0.1, args.print_hz)):
                last_print = now
                fps_str = "?" if fps_ema is None else f"{fps_ema:.1f}"
                if k_est is None:
                    print(
                        f"fps={fps_str} dt={est.dt_sec*1000.0:.1f}ms "
                        f"dx={est.dx_px:+.2f}px dy={est.dy_px:+.2f}px "
                        f"vx={est.vx_px_s:+.2f}px/s vy={est.vy_px_s:+.2f}px/s "
                        f"q={est.quality:.2f} tracked={est.n_tracked}/{est.n_features}"
                    )
                else:
                    print(
                        f"fps={fps_str} dt={est.dt_sec*1000.0:.1f}ms "
                        f"vx={est.vx_px_s:+.2f}px/s vy={est.vy_px_s:+.2f}px/s "
                        f"kvx={k_est.vx_px_s:+.2f}px/s kvy={k_est.vy_px_s:+.2f}px/s "
                        f"q={est.quality:.2f} tracked={est.n_tracked}/{est.n_features}"
                    )

            if args.show:
                view = frame.copy()
                if est is not None:
                    if k_est is None:
                        cv2.putText(
                            view,
                            f"dx {est.dx_px:+.2f} dy {est.dy_px:+.2f} q {est.quality:.2f}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 255, 0),
                            2,
                        )
                    else:
                        cv2.putText(
                            view,
                            f"vx {est.vx_px_s:+.1f} vy {est.vy_px_s:+.1f}  kvx {k_est.vx_px_s:+.1f} kvy {k_est.vy_px_s:+.1f} q {est.quality:.2f}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 255, 0),
                            2,
                        )
                cv2.imshow("e88 optical flow probe", view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        try:
            drone.close()
        finally:
            if args.show:
                cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
