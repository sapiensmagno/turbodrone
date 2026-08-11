"""Measure camera intrinsics (focal length + lens distortion) from a chessboard.

Print a chessboard, measure one square with a ruler, hold it in front of the drone
camera, and move it around the frame. Press SPACE to accept a view, 'c' to calibrate,
'q' to quit.

Aim for 10-20 views spread across the frame -- corners, edges, and centre -- and at
a few different tilts. Pose diversity is what constrains the distortion terms; twenty
views from the same spot are worth about as much as one.

Usage:
    python -m e88_autopilot.run_intrinsics_calibration --camera-id cam2 --square-size-m 0.025
    python -m e88_autopilot.run_intrinsics_calibration --images "calib/*.png"
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

_PKG_PARENT = Path(__file__).resolve().parents[1]
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

from e88_autopilot.intrinsics import IntrinsicsCollector, save_intrinsics


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chessboard intrinsic calibration")
    p.add_argument("--camera-id", default="cam2", help="Identifier to store the result under (e.g. cam1/cam2)")
    p.add_argument(
        "--cam",
        type=int,
        default=None,
        choices=(1, 2),
        help="Physical camera to switch to before capturing (1=forward, 2=downward). "
        "Defaults to the number parsed from --camera-id. Without this the drone stays "
        "on whichever camera is active, and you would save one camera's intrinsics "
        "under the other's id.",
    )
    p.add_argument("--pattern-cols", type=int, default=9, help="Inner corners across (10 squares -> 9)")
    p.add_argument("--pattern-rows", type=int, default=6, help="Inner corners down (7 squares -> 6)")
    p.add_argument("--square-size-m", type=float, default=0.025, help="Measured square edge length in metres")
    p.add_argument("--min-views", type=int, default=6)
    p.add_argument("--images", default=None, help="Glob of still images to use instead of the live camera")
    p.add_argument("--output", default=None, help="Path to intrinsics.json (default: alongside this module)")
    return p.parse_args()


def _calibrate_and_save(collector: IntrinsicsCollector, args: argparse.Namespace) -> int:
    if collector.n_views < int(args.min_views):
        print(f"not enough views: {collector.n_views} < {args.min_views}")
        return 2
    cov = collector.coverage_report()
    intr = collector.calibrate(camera_id=str(args.camera_id), min_views=int(args.min_views))
    out = save_intrinsics(intr, path=None if args.output is None else Path(args.output))

    print()
    print(f"  views              {intr.n_views}")
    print(
        f"  coverage           position {cov['position_spread']:.3f}  "
        f"scale {cov['scale_spread']:.3f}  tilt {cov['tilt_spread']:.3f}"
    )
    print(f"  resolution         {intr.image_w}x{intr.image_h}")
    print(f"  fx, fy             {intr.fx:.2f}, {intr.fy:.2f}")
    print(f"  cx, cy             {intr.cx:.2f}, {intr.cy:.2f}")
    print(f"  focal_length_px    {intr.focal_length_px:.2f}")
    print(f"  k1, k2, k3         {intr.k1:.5f}, {intr.k2:.5f}, {intr.k3:.5f}")
    print(f"  p1, p2             {intr.p1:.5f}, {intr.p2:.5f}")
    print(f"  RMS reproj error   {intr.rms_reproj_error_px:.4f} px")
    print(f"  saved to           {out}")
    if intr.rms_reproj_error_px > 1.0:
        print("  WARNING: RMS > 1 px suggests a bad square size, a mis-specified")
        print("           pattern, or too little pose diversity. Recollect.")
    if float(cov["tilt_spread"]) < 0.05:
        print("  WARNING: almost no out-of-plane tilt across the views. A low RMS here")
        print("           does NOT mean the focal length and distortion are pinned")
        print("           down -- they are only weakly constrained by flat views.")
        print("           Recollect with the board tipped toward and away from the")
        print("           camera and rotated about both axes.")
    return 0


def _run_images(args: argparse.Namespace) -> int:
    paths = sorted(glob.glob(str(args.images)))
    if not paths:
        print(f"no images matched: {args.images}")
        return 2
    collector = IntrinsicsCollector(
        pattern_size=(int(args.pattern_cols), int(args.pattern_rows)),
        square_size_m=float(args.square_size_m),
        min_pose_distance=0.0,  # stills are curated by the operator
    )
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            print(f"  skip (unreadable): {p}")
            continue
        ok, reason = collector.try_add(img)
        print(f"  {'OK  ' if ok else 'skip'} {p}: {reason}")
    return _calibrate_and_save(collector, args)


def _run_live(args: argparse.Namespace) -> int:
    from turbodrone.drone import Drone

    collector = IntrinsicsCollector(
        pattern_size=(int(args.pattern_cols), int(args.pattern_rows)),
        square_size_m=float(args.square_size_m),
    )

    cam = args.cam
    if cam is None:
        # "cam2" -> 2. Only trust the inference when it is unambiguous.
        digits = "".join(ch for ch in str(args.camera_id) if ch.isdigit())
        cam = int(digits) if digits in ("1", "2") else None
    if cam is None:
        print(
            f"cannot infer a physical camera from --camera-id {args.camera_id!r}; "
            "pass --cam 1 or --cam 2 so the intrinsics are saved for the camera "
            "actually being viewed"
        )
        return 2

    drone = Drone()
    drone.connect()
    print(f"switching to camera {cam} ...")
    drone.switch_camera(int(cam))

    print("SPACE = accept view, c = calibrate, q = quit")
    try:
        while True:
            frame = drone.get_frame(timeout=1.0)
            if frame is None:
                continue

            view = frame.copy()
            cv2.putText(
                view,
                f"views: {collector.n_views}  (SPACE accept, c calibrate, q quit)",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.imshow("intrinsics", view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return 1
            if key == ord(" "):
                ok, reason = collector.try_add(frame)
                print(f"  {'OK  ' if ok else 'skip'}: {reason}")
            elif key == ord("c"):
                return _calibrate_and_save(collector, args)
    finally:
        cv2.destroyAllWindows()
        try:
            drone.close()
        except Exception:
            pass


def main() -> int:
    args = _parse_args()
    if args.images:
        return _run_images(args)
    return _run_live(args)


if __name__ == "__main__":
    raise SystemExit(main())
