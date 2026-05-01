from json import load
import time

import pybullet as p

from sim.robot_adapter import RobotAdapter
from sim.robot_loader import (
    connect_pybullet,
    default_robot_spec,
    load_robot
)


def main():
    # 连接PyBullet
    client_id = connect_pybullet(gui=True)

    try:
        robot = load_robot(client_id, default_robot_spec())
        adapter = RobotAdapter(robot)

        adapter.go_home()

        print("home:", adapter.get_joint_position_map())

        target_a = {
                "Joint2": -0.6,
                "Joint3": 0.8,
                "Joint5": 0.5,
            }

        reached = adapter.move_joints(target_a, sleep=True)
        print("target_a reached:", reached)
        print("after target_a:", adapter.get_joint_position_map())

        time.sleep(1.0)

        target_b = {
            "Joint1": 0.4,
            "Joint4": -0.5,
            "Joint6": 0.3,
        }

        reached = adapter.move_joints(target_b, sleep=True)
        print("target_b reached:", reached)
        print("after target_b:", adapter.get_joint_position_map())

        while p.isConnected(client_id):
            adapter.step_simulation(steps=1, sleep=True)

    finally:
        if p.isConnected(client_id):
            p.disconnect(client_id)


if __name__ == "__main__":
    main()