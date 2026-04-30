import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

# 导入和参数解析  
def parse_args():
    parser = argparse.ArgumentParser(description="Capture one aligned RGB-D frame from the RealSense camera.")
    
    # 图像宽度
    parser.add_argument("--width", type=int, default=640)
    
    # 图像高度
    parser.add_argument("--height", type=int, default=480)
    
    # 帧率
    parser.add_argument("--fps", type=int, default=30)
    
    # 预热帧数 前几帧常常曝光和深度不稳定
    parser.add_argument("--warmup" , type=int, default=30)

    # 输出根目录
    parser.add_argument("--output-root", type=Path, default=Path("captures"))

    # 可选的采集名称，默认使用时间戳
    parser.add_argument("--name", type=str, default="")

    return parser.parse_args()

# 输出目录函数
def make_output_dir(output_root:Path, name:str) -> Path:
    # 如果没有传入name，就使用当前时间戳作为目录名
    output_name = name or time.strftime("%Y%m%d_%H%M%S")

    # 最终目录
    output_dir = output_root / output_name
    
    # 避免覆盖原数据
    output_dir.mkdir(parents=True, exist_ok=False)
    
    return output_dir


# RealSense初始化
def initialize_realsense(width: int, height: int, fps: int) -> tuple[rs.pipeline, rs.align, float]:
    # 创建管线对象
    pipeline = rs.pipeline()

    # 创建配置对象
    config = rs.config()

    # 开启彩色流 BGR8与OpenCV兼容 
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    
    # 开启深度流 Z16格式表示16位深度图
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    
    # start() 返回当前激活配置对应的 profile
    profile = pipeline.start(config)

    # 取深度比例尺 用于后续单位转换为m
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    # 将深度对齐到彩色图坐标系
    align = rs.align(rs.stream.color)


    return pipeline, align, depth_scale


# 主函数
def main():
    # 1.读取命令行参数
    args = parse_args()

    # 2.创建输出目录
    output_dir = make_output_dir(args.output_root, args.name)

    # 3.初始化RealSense相机
    pipeline, align, depth_scale = initialize_realsense(args.width, args.height, args.fps)

    try:
        # 4.预热若干帧 稳定相机
        for _ in range(args.warmup):
            pipeline.wait_for_frames()

        # 5.取一帧原始数据
        frames = pipeline.wait_for_frames()

        # 6.对齐深度和彩色图
        aligned_frames = align.process(frames)

        # 7.获取彩色帧和深度帧
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        # 8.检查帧是否有效
        if not color_frame or not depth_frame:
            raise RuntimeError("Failed to capture frames from RealSense camera.")
        
        # 9.转换为numpy数组
        # color_image 通常是 (H, W, 3)，dtype=uint8
        # depth_image 通常是 (H, W)，dtype=uint16
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        print("color shape:", color_image.shape, "dtype:", color_image.dtype)
        print("depth shape:", depth_image.shape, "dtype:", depth_image.dtype)

        # 10.保存彩色图和深度图
        color_path = output_dir / "color.png"
        depth_path = output_dir / "depth.png"
        depth_vis_path = output_dir / "depth_vis.png"

        cv2.imwrite(str(color_path), color_image)
        cv2.imwrite(str(depth_path), depth_image)
        # 再额外保存一张“给人看”的深度可视化图
        # alpha=0.03 只是为了把数值缩放到 8-bit 显示范围
        depth_vis = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET,
        )
        cv2.imwrite(str(depth_vis_path), depth_vis)

        # 11. 从 color_frame 里读取相机内参
        intr = color_frame.profile.as_video_stream_profile().intrinsics

        intrinsics = {
            "width": intr.width,
            "height": intr.height,
            "fx": intr.fx,
            "fy": intr.fy,
            "cx": intr.ppx,
            "cy": intr.ppy,
            "depth_scale": depth_scale,
        }

        intrinsics_path = output_dir / "intrinsics.json"
        intrinsics_path.write_text(
            json.dumps(intrinsics, indent=2),
            encoding="utf-8",
        )


        print("Capture succeeded")
        print("Output dir:", output_dir)
        print("Depth Scale:", depth_scale)
        print("Saved color to:", color_path)
        print("Saved depth to:", depth_path)
        print("Saved depth visualization to:", depth_vis_path)
        print("Saved intrinsics to:", intrinsics_path)

    
    finally:
        # 停止相机释放资源
        pipeline.stop()

    return 0

# 主函数入口
if __name__ ==  "__main__":
    raise SystemExit(main())
    




