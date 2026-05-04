import re
import time

from flask import render_template
import pybullet as p

from sim.robot_loader import connect_pybullet
from sim.scene_builder import build_minimal_scene
from sim.sim_camera import SimCameraSpec, render_camera, rgbd_to_point_cloud

def main():
    
    client_id = connect_pybullet()
    try:
        # 建立场景
        build_minimal_scene(client_id)

        p.resetDebugVisualizerCamera(
            cameraDistance=1.1,
            cameraYaw=45.0,
            cameraPitch=-35.0,
            cameraTargetPosition=[0.25, 0.0, 0.08],
            physicsClientId=client_id,
        )

        spec = SimCameraSpec()
        # 渲染一帧
        frame = render_camera(client_id, spec)
        # 转化为点云
        points, colors = rgbd_to_point_cloud(frame)
        # 打印shape和范围
        print("color shape:", frame.color_rgb.shape, frame.color_rgb.dtype)
        print("depth shape:", frame.depth_m.shape, frame.depth_m.dtype)
        print("points shape:", points.shape, points.dtype)
        print("colors shape:", colors.shape, colors.dtype)
        print("points min:", points.min(axis=0))
        print("points max:", points.max(axis=0))

        while p.isConnected(client_id):
            p.stepSimulation(physicsClientId=client_id)
            time.sleep(1.0 / 240.0)

    finally:
        if p.isConnected(client_id):
            p.disconnect(client_id)

    
if __name__ == "__main__":
    main()
    