import time
import pybullet as p

from sim.robot_loader import connect_pybullet
from sim.scene_builder import build_minimal_scene
from sim.grasp_targets import build_all_topdown_grasp_targets

def main():
    # 连接PyBullet
    client_id = connect_pybullet(gui=True)

    try:
        scene = build_minimal_scene(client_id)
        targets = build_all_topdown_grasp_targets(scene)

        for target in targets:
            print("box_id:", target.box_id)
            print("center_xyz:", target.center_xyz)
            print("top_center_xyz:", target.top_center_xyz)
            print("pregrasp_xyz:", target.pregrasp_xyz)
            print("grasp_yaw:", target.grasp_yaw)

        p.resetDebugVisualizerCamera(
            cameraDistance=1.1,
            cameraYaw=45.0,
            cameraPitch=-35.0,
            cameraTargetPosition=[0.25, 0.0, 0.15],
            physicsClientId=client_id,
        )

        while p.isConnected(client_id):
            p.stepSimulation(physicsClientId=client_id)
            time.sleep(1.0 / 240.0)

    finally:
        if p.isConnected(client_id):
            p.disconnect(client_id)


if __name__ == "__main__":
    main()