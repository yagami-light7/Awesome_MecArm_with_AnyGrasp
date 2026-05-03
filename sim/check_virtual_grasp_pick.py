import time
import math
import pybullet as p

from sim.grasp_targets import build_all_topdown_grasp_targets
from sim.pregrasp_executor import PregraspExecutor
from sim.robot_adapter import RobotAdapter
from sim.robot_loader import connect_pybullet
from sim.scene_builder import build_minimal_scene
from sim.virtual_grasp import attach_box_to_tcp


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

        # 执行预抓取
        pregrasp_reached = executor.execute_pregrasp(target)
        print("pregrasp_reached:", pregrasp_reached)
        pregrasp_error = executor.get_pregrasp_position_error(target)
        print("pregrasp_position_error:", pregrasp_error)

        # 执行approach
        attach_xyz = (
            target.top_center_xyz[0],
            target.top_center_xyz[1],
            target.top_center_xyz[2] + 0.01,
        )
        print("attach_xyz:", attach_xyz)

        approach_reached = executor.execute_xyz(attach_xyz)
        print("approach_reached:", approach_reached)
        approach_error = executor.get_position_error(attach_xyz)
        print("approach_position_error:", approach_error)

        # 计算误差范数并根据误差判断是否attach
        dx, dy, dz = approach_error
        approach_error_norm = math.sqrt(dx**2 + dy**2 + dz**2)
        print("approach_error_norm:", approach_error_norm)
        
        grasp_handle = None

        if approach_error_norm > 0.03:
            print("attah skipped:TCP is too far from target")
        else:
            grasp_handle = attach_box_to_tcp(adapter.robot, target.box_id)
            print("virtual grasp attached:", grasp_handle.constraint_id)

        # attach成功 执行lift
        if grasp_handle is not None:
            lift_reached = executor.execute_xyz(target.pregrasp_xyz)
            print("lift_reached:", lift_reached)
            lift_error = executor.get_position_error(target.pregrasp_xyz)
            print("lift_position_error:", lift_error)

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