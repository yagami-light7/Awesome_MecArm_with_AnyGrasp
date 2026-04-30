import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

# 解析命令行参数 仅保留输入目录参数
def parse_args():
    parser = argparse.ArgumentParser(
        description="Rebuild a point cloud from a saved RGB-D capture."
    )
    parser.add_argument(
        "--capture-dir",
        type=Path,
        required=True,
        help="Path to a capture directory containing color.png, depth.png, and intrinsics.json",
    )
    return parser.parse_args()


# 主程序入口
def main():
    # 读取命令行参数
    args = parse_args()
    capture_dir = args.capture_dir
    
    # 读取文件
    color_path = capture_dir / "color.png"
    depth_path = capture_dir / "depth.png"
    intrinsics_path = capture_dir / "intrinsics.json"

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

    # 构建Open3D的内参对象 让open3d理解相机内参
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=intrinsics["width"],
        height=intrinsics["height"],
        fx=intrinsics["fx"],
        fy=intrinsics["fy"],
        cx=intrinsics["cx"],
        cy=intrinsics["cy"],
    )

    # 创建Open3D的RGBD图像对象
    o3d_color = o3d.geometry.Image(color_rgb)
    o3d_depth = o3d.geometry.Image(depth_image)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color,
        o3d_depth,
        depth_scale= 1.0 / intrinsics["depth_scale"],
        depth_trunc= 1.5,  # 深度截断距离 1.5米以外的深度值将被丢弃
        convert_rgb_to_intensity=False,  # 保持彩色信息 不转换为灰度
    )


    # 重建点云
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd,
        intrinsic,
    )
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])  # 调整点云朝向，便于按 Open3D 常见视角显示

    # 可视化点云
    o3d.visualization.draw_geometries([pcd], window_name="Reconstructed Point Cloud")

    print("color shape:", color_image.shape, color_image.dtype)
    print("depth shape:", depth_image.shape, depth_image.dtype)
    print("points:", len(pcd.points))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
