from __future__ import annotations

import argparse
import time

import cv2

from e88_autopilot.intrinsics import resolve_camera_geometry
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
    parser.add_argument("--motion-model", type=str, default="affine_translation")
    parser.add_argument(
        "--focal-length-px",
        type=float,
        default=None,
        help="Override the focal length. By default it is loaded from the saved "
        "chessboard calibration for --camera-id at the stream's resolution.",
    )
    parser.add_argument(
        "--camera-id",
        default=None,
        help="Calibration entry to load (default: cam<--camera>, else cam2).",
    )
    args = parser.parse_args()

    drone = Drone(protocol=args.protocol)
    drone.connect()

    if args.camera is not None:
        drone.switch_camera(args.camera)

    camera_id = args.camera_id or (f"cam{int(args.camera)}" if args.camera is not None else "cam2")

    # The estimator needs f and the principal point up front, but calibration entries
    # are keyed by the resolution RTSP actually negotiated -- so build it once the
    # first frame tells us the frame size.
    estimator = None
    kf = None if not args.kalman else VelocityKalman2D(sigma_a=args.sigma_a, sigma_v_meas=args.sigma_v)

    last_print = 0.0
    last_ts = None
    fps_ema = None
    baseline_ts = None

    try:
        while True:
            item = drone.get_frame_with_timestamp(timeout=args.timeout)
            if item is None:
                raise RuntimeError("No frame received")
            frame, ts = item

            if estimator is None:
                # After a camera switch the stream still holds the pre-switch frame,
                # possibly at a different resolution. Skip the first frame seen so the
                # geometry is resolved against the selected camera.
                if args.camera is not None and baseline_ts is None:
                    baseline_ts = float(ts)
                    continue
                if baseline_ts is not None and float(ts) <= baseline_ts:
                    continue

                h, w = frame.shape[:2]
                f_px, principal, note = resolve_camera_geometry(
                    camera_id=camera_id,
                    image_w=int(w),
                    image_h=int(h),
                    focal_length_px_override=args.focal_length_px,
                )
                print(f"[intrinsics] {note}")
                estimator = LucasKanadeDriftEstimator(
                    motion_model=str(args.motion_model),
                    focal_length_px=float(f_px),
                    principal_point_px=principal,
                )

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
                        f"q={est.quality:.2f} tracked={est.n_tracked}/{est.n_features} "
                        f"model={est.motion_model} rmse={est.model_rmse_px:.2f} "
                        f"wx={est.omega_x_rad:+.4f} wy={est.omega_y_rad:+.4f} wz={est.omega_z_rad:+.4f}"
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
