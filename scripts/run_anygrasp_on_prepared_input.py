import argparse 
import json
from pathlib import Path

import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run AnyGrasp on prepared point-cloud input."
    )

    # prepare_grasp_input.py 生成的输出目录
    parser.add_argument("--input-dir", type=Path, required=True)

    # 官方 demo 里需要 checkpoint 路径
    parser.add_argument("--checkpoint-path", type=str, required=True)

    # 官方 demo 里有这两个参数
    parser.add_argument("--max-gripper-width", type=float, default=0.1)
    parser.add_argument("--gripper-height", type=float, default=0.03)

    # 官方 demo 里的开关
    parser.add_argument("--top-down-grasp", action="store_true")
    parser.add_argument("--debug", action="store_true")

    # 这些是工作空间裁剪范围，AnyGrasp 官方 demo 也要求 lims
    parser.add_argument("--xmin", type=float, default=-0.3)
    parser.add_argument("--xmax", type=float, default=0.3)
    parser.add_argument("--ymin", type=float, default=-0.3)
    parser.add_argument("--ymax", type=float, default=0.3)
    parser.add_argument("--zmin", type=float, default=0.0)
    parser.add_argument("--zmax", type=float, default=1.2)

    return parser.parse_args()

# 加载预处理数据
def load_prepared_input(input_dir:Path):
    points_path = input_dir / "points.npy"
    colors_path = input_dir / "colors.npy"
    meta_path = input_dir / "meta.json"

    points = np.load(points_path)
    colors = np.load(colors_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points to be of shape (N, 3), but got {points.shape}")
    if colors.ndim != 2 or colors.shape[1] != 3:
        raise ValueError(f"Expected colors to be of shape (N, 3), but got {colors.shape}")
    if len(points) != len(colors):
        raise ValueError(f"Number of points ({len(points)}) does not match number of colors ({len(colors)})")

    points = points.astype(np.float32)
    colors = colors.astype(np.float32)

    return points, colors, meta


# 主程序入口
def main():
    args = parse_args()

    points, colors, meta = load_prepared_input(args.input_dir)

    print("points shape:", points.shape, points.dtype)
    print("colors shape:", colors.shape, colors.dtype)
    print("meta keys:", list(meta.keys()))

    # 官方 demo 里要求最大夹爪宽度不要超过 0.1 米
    args.max_gripper_width = max(0.0, min(0.1, args.max_gripper_width))

    try:
        from gsnet import AnyGrasp
    except ImportError:
        print("Failed to import AnyGrasp SDK. Please install gsnet/AnyGrasp first.")
        return 1
    
    try:
        import open3d as o3d
    except ImportError:
        o3d = None

    # 工作空间
    lims = [
        args.xmin, args.xmax,
        args.ymin, args.ymax,
        args.zmin, args.zmax,
    ]

    print("workspace limits:", lims)

    # TODO: after AnyGrasp SDK is installed
    # anygrasp = AnyGrasp(args)
    # anygrasp.load_net()
    # gg, cloud = anygrasp.get_grasp(
    #     points,
    #     colors,
    #     lims=lims,
    #     apply_object_mask=True,
    #     dense_grasp=False,
    #     collision_detection=True,
    # )


