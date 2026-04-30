import argparse
import json
from pathlib import Path

import cv2
import numpy as np


# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare point-cloud input for grasp models from a saved RGB-D capture."
    )
    parser.add_argument("--capture-dir", type=Path, required=True)  # 保存的RGB-D数据目录
    parser.add_argument("--output-dir", type=Path, required=True)   # 处理后点云输出目录
    parser.add_argument("--z-min", type=float, default=0.15)        # 深度过滤的最小值，单位为米
    parser.add_argument("--z-max", type=float, default=1.2)         # 深度过滤的最大值，单位为米
    parser.add_argument("--num-points", type=int, default=20000)    # 输出点云的点数
    
    return parser.parse_args()

# 主程序入口
def main():
    args = parse_args()
    
    # 读取RGB-D数据
    color_path = args.capture_dir / "color.png"
    depth_path = args.capture_dir / "depth.png"
    intrinsics_path = args.capture_dir / "intrinsics.json"

    color_image = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    depth_image = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

    if color_image is None:
        print(f"Failed to read color image from {color_path}")
        return 1
    if depth_image is None:
        print(f"Failed to read depth image from {depth_path}")
        return 1
    try:
        intrinsics = json.loads(intrinsics_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"Intrinsics file not found: {intrinsics_path}")
        return 1
    except json.JSONDecodeError:
        print(f"Invalid JSON in: {intrinsics_path}")
        return 1

    
    # 将BGR转化为RGB Open3D使用RGB格式 而OpenCV默认使用BGR
    color_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    # 将深度图转换为米
    depth_m = depth_image.astype(np.float32) * intrinsics["depth_scale"]

    # 构建点云
    height, width = depth_m.shape

    u,v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))

    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]

    z = depth_m
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.stack((x,y,z), axis=-1).reshape(-1, 3)

    # 处理颜色
    colors = color_rgb.reshape(-1, 3).astype(np.float32) / 255.0

    # 深度过滤
    z_flat = z.reshape(-1)

    valid_mask = (z_flat > args.z_min) & (z_flat < args.z_max) & (z_flat > 0)

    points = points[valid_mask]
    colors = colors[valid_mask]

    # 固定采样点
    if len(points) == 0:
        print("No valid points found after depth filtering.")
        return 1
    if len(points) > args.num_points:
        indices = np.random.choice(len(points), args.num_points, replace=False)
    else:
        indices = np.random.choice(len(points), args.num_points, replace=True)
    
    points = points[indices]
    colors = colors[indices]

    # 保存点云数据
    output_dir = args.output_dir
    capture_dir = args.capture_dir
    output_dir.mkdir(parents=True, exist_ok=False)

    np.save(output_dir / "points.npy", points)
    np.save(output_dir / "colors.npy", colors)
    meta = {
    "capture_dir": str(capture_dir),
    "num_points": int(len(points)),
    "z_min": args.z_min,
    "z_max": args.z_max,
    "fx": intrinsics["fx"],
    "fy": intrinsics["fy"],
    "cx": intrinsics["cx"],
    "cy": intrinsics["cy"],
    "depth_scale": intrinsics["depth_scale"],
}
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("raw depth shape:", depth_image.shape, depth_image.dtype)
    print("points after filtering:", points.shape)
    print("colors after filtering:", colors.shape)
    print("saved to:", output_dir)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())