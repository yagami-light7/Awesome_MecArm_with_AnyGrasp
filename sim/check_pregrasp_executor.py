from http import client
import time 
import pybullet as p
from sympy import im

from sim.grasp_targets import build_all_topdown_grasp_targets
from sim.pregrasp_executor import PregraspExecutor
from sim.robot_adapter import RobotAdapter
from sim.robot_loader import connect_pybullet
from sim.scene_builder import build_minimal_scene


# 预抓取执行器测试
def main():
    client_id = connect_pybullet(gui=True)

    try:
        # 创建场景
        scene = build_minimal_scene(client_id)
        # 创建机器人适配器
        adapter = RobotAdapter(scene.robot)
        # 创建执行器
        executor = PregraspExecutor(adapter)

        # 先回到初始位置
        adapter.go_home()
        # 获取抓取目标
        targets = build_all_topdown_grasp_targets(scene)
        target = targets[0]

        print("selected box_id:", target.box_id)
        print("pregrasp_xyz:", target.pregrasp_xyz)

        reached = executor.execute_pregrasp(target)
        print("pregrasp_reached:", reached)

        error = executor.get_pregrasp_position_error(target)
        print("pregrasp_position_error:", error)

        p.resetDebugVisualizerCamera(
            cameraDistance=1.1,
            cameraYaw=45.0,
            cameraPitch=-35.0,
            cameraTargetPosition=[0.25, 0.0, 0.12],
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