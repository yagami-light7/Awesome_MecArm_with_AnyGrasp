import argparse
import sys

import cv2
import numpy as np
import pyrealsense2 as rs


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview RealSense color and depth streams.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    try:
        profile = pipeline.start(config)
        device = profile.get_device()
        print("device:", device.get_info(rs.camera_info.name))
        print("serial:", device.get_info(rs.camera_info.serial_number))
        print("press 'q' to quit")

        align = rs.align(rs.stream.color)
        colorizer = rs.colorizer()

        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_colorized = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            preview = np.hstack((color_image, depth_colorized))

            cv2.imshow("RealSense Preview", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    except Exception as exc:
        print(f"preview failed: {exc}", file=sys.stderr)
        return 1
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
