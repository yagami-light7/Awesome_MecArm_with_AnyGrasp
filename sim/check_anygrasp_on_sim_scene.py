from pathlib import Path
import time

import pybullet as p

from sim.anygrasp_interface import AnyGraspConfig, run_anygrasp
from sim.robot_loader import connect_pybullet
from sim.scene_builder import build_minimal_scene
from sim.sim_camera import SimCameraSpec, render_camera,rgbd_to_point_cloud

# 点云自动生成lims
def make_lims_from_points(points, margin: float = 0.02) -> list[float]:
    mins = points.min(axis=0) 
    maxs = points.max(axis=0)

    return [
        float(mins[0] - margin),
        float(maxs[0] + margin),
        float(mins[1] - margin),
        float(maxs[1] + margin),
        float(max(mins[2] - margin, 0.0)),
        float(maxs[2] + margin),
    ]


def main():
    client_id = connect_pybullet(gui=True)

    try:
        # 建立场景
        scene = build_minimal_scene(client_id)

        spec = SimCameraSpec()
        
        # 渲染一帧
        frame = render_camera(client_id, spec)
        
        # 转化为点云
        points, colors = rgbd_to_point_cloud(frame)
        
        print("points shape:", points.shape)
        print("colors shape:", colors.shape)
        print("points min:", points.min(axis=0))
        print("points max:", points.max(axis=0))

        # checkpoint路径
        project_root = Path(__file__).resolve().parents[1]
        checkpoint_path = (
            project_root
            / "third_party"
            / "anygrasp_sdk"
            / "grasp_detection"
            / "log"
            / "checkpoint_detection.tar"
        )

        print("checkpoint path:", checkpoint_path)

        # config
        cfg = AnyGraspConfig(
            checkpoint_path=checkpoint_path,
            top_down_grasp=True,
            debug=False,
        )

        # 生成lims并跑推理
        lims = make_lims_from_points(points)
        print("lims:", lims)

        gg, cloud = run_anygrasp(points, colors, lims, cfg)

        # 打印结果
        print("num grasps:", len(gg))
        
        if len(gg) > 0:
            top_k = gg[:5]
            print("top scores:", top_k.scores)

            best = top_k[0]
            print("best score:", best.score)
            print("best translation:", best.translation)
            print("best width:", best.width)
            print("best depth:", best.depth)

        while p.isConnected(client_id):
            p.stepSimulation(physicsClientId=client_id)
            time.sleep(1.0 / 240.0)

    finally:
        if p.isConnected(client_id):
            p.disconnect(client_id)


if __name__ == "__main__":
    main()
