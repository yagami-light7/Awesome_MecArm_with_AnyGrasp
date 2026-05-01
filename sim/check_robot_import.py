import time
import pybullet as p
from sim.robot_loader import (
    connect_pybullet, 
    default_robot_spec, 
    get_end_effector_pose,
    reset_arm_joints,
    load_robot,
    reset_arm_joints
)

def main():
    # 连接PyBullet
    client_id = connect_pybullet(gui=True)

    try:
        robot = load_robot(client_id, default_robot_spec())

        print("body_id:", robot.body_id)
        print("num_all_joints:", len(robot.joints))
        print("num_movable_joints:", len(robot.movable_joints))
        print("movable_joint_indices:", robot.movable_joint_indices)
        print("end_effector_link_name:", robot.spec.end_effector_link_name)
        print("end_effector_link_index:", robot.end_effector_link_index)

        reset_arm_joints(robot)

        ee_pos, ee_quat = get_end_effector_pose(robot)
        print("end_effector_position:", ee_pos)
        print("end_effector_orientation:", ee_quat)

        for joint in robot.joints:
            print(
                f"{joint.index}:{joint.name}"
                f"limits=({joint.lower_limit:.2f}, {joint.upper_limit:.2f})"
                f"link={joint.child_link_name}"
            )

        # 调整相机视角
        p.resetDebugVisualizerCamera(
            cameraDistance=0.8,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.2],
            physicsClientId=client_id
        )

        while p.isConnected(client_id):
            p.stepSimulation(physicsClientId = client_id)
            time.sleep(1.0 / 240.0)

    finally:
        if p.isConnected(client_id):
            p.disconnect(client_id)     


if __name__ == "__main__":
    main()