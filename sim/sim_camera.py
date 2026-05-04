from dataclasses import dataclass
import math
from turtle import color, width

import numpy as np
import pybullet as p

# 定义相机规格
@dataclass(frozen=True)
class SimCameraSpec:
    width: int = 640
    height: int = 480

    # 垂直视场角
    fov_y_deg : float = 60.0

    near:float = 0.02
    far:float = 2.0

    # 相机在世界坐标系的位置
    eye_xyz : tuple[float, float, float] = (0.45, - 0.55, 0.55)

    # 相机看向的目标点
    target_xyz : tuple[float, float, float] = (0.25, 0.0, 0.08)

    # 世界系上方向
    up_xyz: tuple[float, float, float] = (0.0, 0.0, 1.0)

# 定义相机返回的图像数据结构
@dataclass
class SimCameraFrame:
    color_rgb:np.ndarray
    depth_m:np.ndarray

    fx:float
    fy:float
    cx:float
    cy:float

    width:int
    height:int

# 从规格生成相机矩阵
def build_camera_matrices(spec: SimCameraSpec) -> tuple[list[float], list[float]]:
    aspect = spec.width / spec.height

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=spec.eye_xyz,
        cameraTargetPosition=spec.target_xyz,
        cameraUpVector=spec.up_xyz,
    )

    
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=spec.fov_y_deg,
        aspect=aspect,
        nearVal=spec.near,
        farVal=spec.far,
    )

    return view_matrix, projection_matrix

# 将深度单位转化为m
def depth_buffer_to_meters(
        depth_buffer: np.ndarray,
        near:float,
        far:float
) -> np.ndarray:
    depth_m = near * far / (far - (far - near) * depth_buffer)
    return depth_m.astype(np.float32)

# intrinsic参数计算
def intrinsics_from_fovy(
        width:int,
        height:int,
        fov_y_deg:float
) -> tuple[float, float, float, float]:

    fov_y_rad = math.radians(fov_y_deg)
    fy = 0.5 * height / math.tan(0.5 * fov_y_rad)
    fx = fy

    cx = 0.5 * (width - 1)
    cy = 0.5 * (height - 1)

    return fx, fy, cx, cy

# 渲染函数
def render_camera(
        client_id: int,
        spec: SimCameraSpec
) -> SimCameraFrame:
    
    view_matrix, projection_matrix = build_camera_matrices(spec)

    _, _, rgba, depth_buffer, _ = p.getCameraImage(
        width = spec.width,
        height = spec.height,
        viewMatrix = view_matrix,
        projectionMatrix = projection_matrix,
        renderer = p.ER_BULLET_HARDWARE_OPENGL,
        physicsClientId=client_id,
    )

    # 处理颜色
    rgba = np.asarray(rgba, dtype=np.uint8).reshape(spec.height, spec.width, 4)
    color_rgb = rgba[:, :, :3].copy()

    # 处理深度
    depth_buffer = np.asarray(depth_buffer, dtype=np.float32).reshape(spec.height, spec.width)
    depth_m = depth_buffer_to_meters(depth_buffer, spec.near, spec.far)

    # 计算intrinsic
    fx, fy, cx, cy = intrinsics_from_fovy(spec.width, spec.height, spec.fov_y_deg)

    return SimCameraFrame(
        color_rgb=color_rgb,
        depth_m=depth_m,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        width=spec.width,
        height=spec.height
    )

# RGB-D 转 点云
def rgbd_to_point_cloud(
        frame: SimCameraFrame,
        z_min: float = 0.02,
        z_max: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = frame.height, frame.width

    # 生成像素坐标网格
    u = np.meshgrid(np.arange(width), np.arange(height))[0]
    v = np.meshgrid(np.arange(width), np.arange(height))[1]

    z = frame.depth_m
    x = (u - frame.cx) * z / frame.fx
    y = (v - frame.cy) * z / frame.fy

    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = frame.color_rgb.reshape(-1, 3).astype(np.float32) / 255.0

    z_flat = z.reshape(-1)
    valid_mask = (z_flat > z_min) & (z_flat < z_max)

    points = points[valid_mask]
    colors = colors[valid_mask]

    return points.astype(np.float32), colors.astype(np.float32)