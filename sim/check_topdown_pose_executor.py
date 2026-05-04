from asyncio import sleep
import time
import math
import pybullet as p

from sim.grasp_targets import build_all_topdown_grasp_targets
from sim.pregrasp_executor import PregraspExecutor, PregraspConfig
from sim.robot_adapter import RobotAdapter, JointControlConfig
from sim.robot_loader import connect_pybullet
from sim.scene_builder import build_minimal_scene

TOPDOWN_QUAT = (-1.0, 0.0, 0.01, 0)

def pause_stage(seconds: float = 0.5) -> None:
    time.sleep(seconds)


def main():
    client_id = connect_pybullet(gui=True)

    try:
        # 设置摄像头
        p.resetDebugVisualizerCamera(
            cameraDistance=1.1,
            cameraYaw=45.0,
            cameraPitch=-35.0,
            cameraTargetPosition=[0.25, 0.0, 0.12],
            physicsClientId=client_id,
        )

        # 创建场景
        scene = build_minimal_scene(client_id)
        
        # 创建机器人适配器
        adapter = RobotAdapter(
            scene.robot,
            JointControlConfig(
                force_scale=10.0,
                velocity_scale=0.25,
                position_tolerance=0.01,
                timeout_s=5.0,
                control_hz=240,
            )
        )
        # 创建执行器
        executor = PregraspExecutor(
            adapter,
            PregraspConfig(
                sleep = True,
                topdown_quat = TOPDOWN_QUAT,
            )
        )

        # 先回到初始位置
        adapter.go_home()

        # 获取抓取目标
        targets = build_all_topdown_grasp_targets(scene)
        target = targets[0]

        print("selected box_id:", target.box_id)
        print("pregrasp_xyz:", target.pregrasp_xyz)

        # 执行预抓取
        pregrasp_reached = executor.execute_pregrasp(target)
        print("pregrasp_reached:", pregrasp_reached)
        pregrasp_position_error = executor.get_pregrasp_position_error(target)
        print("pregrasp_position_error:", pregrasp_position_error)
        pregrasp_orientation_error = executor.get_orientation_error_rad(TOPDOWN_QUAT)
        print("pregrasp_orientation_error (rad):", pregrasp_orientation_error)
        pause_stage()


        while p.isConnected(client_id):
            p.stepSimulation(physicsClientId=client_id)
            time.sleep(1.0 / 240.0)


    finally:
        if p.isConnected(client_id):
            p.disconnect(client_id)




if __name__ == "__main__":
    main()