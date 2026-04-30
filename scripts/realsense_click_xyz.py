import argparse
import sys
from typing import Optional

import cv2
import numpy as np
import pyrealsense2 as rs


def mean_valid_depth(depth_image: np.ndarray, x: int, y: int, radius: int, depth_scale: float) -> float:
    x0 = max(0, x - radius)
    x1 = min(depth_image.shape[1], x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(depth_image.shape[0], y + radius + 1)

    patch = depth_image[y0:y1, x0:x1].astype(np.float32) * depth_scale
    valid = patch[patch > 0]
    if valid.size == 0:
        return 0.0
    return float(valid.mean())


def main() -> int:
    parser = argparse.ArgumentParser(description="Click a pixel in the RealSense color image to get its 3D camera-frame coordinates.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--avg-radius", type=int, default=2, help="Average depth over a square neighborhood of this radius.")
    args = parser.parse_args()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    latest_color: Optional[np.ndarray] = None
    latest_depth: Optional[np.ndarray] = None
    intrinsics: Optional[rs.intrinsics] = None
    depth_scale = 0.001
    click_message = "left click to query 3D point, press 'q' to quit"

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        nonlocal click_message
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if latest_depth is None or intrinsics is None:
            click_message = "frame not ready yet"
            return

        depth_m = mean_valid_depth(latest_depth, x, y, args.avg_radius, depth_scale)
        if depth_m <= 0:
            click_message = f"pixel=({x}, {y}) depth invalid"
            print(click_message)
            return

        point = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth_m)
        click_message = (
            f"pixel=({x}, {y}) depth={depth_m:.4f}m "
            f"xyz=({point[0]:.4f}, {point[1]:.4f}, {point[2]:.4f})m"
        )
        print(click_message)

    try:
        profile = pipeline.start(config)
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        align = rs.align(rs.stream.color)

        cv2.namedWindow("RealSense Click XYZ", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("RealSense Click XYZ", on_mouse)
        print(click_message)

        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
            latest_color = np.asanyarray(color_frame.get_data())
            latest_depth = np.asanyarray(depth_frame.get_data())

            display = latest_color.copy()
            cv2.putText(
                display,
                click_message,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("RealSense Click XYZ", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    except Exception as exc:
        print(f"click xyz failed: {exc}", file=sys.stderr)
        return 1
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
