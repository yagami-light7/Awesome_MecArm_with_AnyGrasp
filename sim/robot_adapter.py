'''
提供机械臂运动接口
'''

from dataclasses import dataclass
from decimal import Clamped
import time

import pybullet as p
from torch import clamp

from sim.robot_loader import (
    DEFAULT_HOME_JOINT_POSITIONS,
    LoadedRobot,
    reset_arm_joints
)

# 控制参数
@dataclass(frozen=True)
class JointControlConfig:
    # 力矩缩放
    force_scale : float = 10.0

    # 速度缩放
    velocity_scale : float = 0.1

    # 位置误差容忍
    position_tolerance: float = 0.01

    # 到位超时
    timeout_s: float = 5.0

    # 控制周期
    control_hz: float = 240


# 加载机器人及其控制接口
@dataclass
class RobotAdapter:
    robot: LoadedRobot
    control: JointControlConfig = JointControlConfig()


    # 仿真步进
    def step_simulation(self, steps: int = 1, sleep: bool = False) -> None:
        dt = 1.0 / self.control.control_hz
        for _ in range(steps):
            p.stepSimulation(physicsClientId=self.robot.client_id)
            if sleep:
                time.sleep(dt)


    # 读取关节状态
    def get_joint_positions(self) -> list[float]:
        positions: list[float] = []

        for joint in self.robot.movable_joints:
            state = p.getJointState(
                self.robot.body_id,
                joint.index,
                physicsClientId=self.robot.client_id
            )
            positions.append(state[0])
        return positions


    # 读取关节对应map
    def get_joint_position_map(self) -> dict[str, float]:
        positions_map: dict[str, float] = {}

        for joint in self.robot.movable_joints:
            state = p.getJointState(
                self.robot.body_id,
                joint.index,
                physicsClientId=self.robot.client_id
            )
            positions_map[joint.name] = state[0]
        return positions_map


    # 关节复位
    def go_home(self) -> None:
        reset_arm_joints(self.robot)


    #关节目标限幅函数 
    def clamp_joint_targets(
            self,
            target_positions: dict[str, float]
    )-> dict[str, float]:
        clamped_targets: dict[str, float] = {}

        for joint in self.robot.movable_joints:
            if joint.name not in target_positions:
                continue
            
            target_pos = target_positions[joint.name]
            clamped_pos = max(joint.lower_limit, min(joint.upper_limit, target_pos))
            clamped_targets[joint.name] = clamped_pos
        
        return clamped_targets
    


    # 关节位置控制
    def command_joint_positions(
            self,
            target_positions: dict[str, float],
    )-> None:
        for joint in self.robot.movable_joints:
            if joint.name not in target_positions:
                continue
            
            clamped_targets = self.clamp_joint_targets(target_positions)

            target_pos = clamped_targets[joint.name]
            p.setJointMotorControl2(
                bodyUniqueId=self.robot.body_id,
                jointIndex=joint.index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_pos,
                force=joint.max_force * self.control.force_scale,
                maxVelocity=joint.max_velocity * self.control.velocity_scale,
                physicsClientId=self.robot.client_id
            )


    # 误差检查接口
    def is_at_joint_positions(self, target_positions: dict[str, float]) -> bool:
        clamped_targets = self.clamp_joint_targets(target_positions)
        current_positions = self.get_joint_position_map()

        # 检查每个关节位置误差是否在容忍范围内
        for joint_name, target_pos in clamped_targets.items():
            if joint_name not in current_positions:
                continue

            current_pos = current_positions[joint_name]
            error = abs(current_pos - target_pos)

            if error > self.control.position_tolerance:
                return False
        
        return True


    # 关节等待控制收敛
    def wait_until_joint_positions_reached(
            self,
            target_positions: dict[str, float],
            sleep: bool = False
    ) -> bool:
        
        # 最大步数
        max_steps = int(self.control.timeout_s * self.control.control_hz)

        # 循环步进直到收敛或超时
        for _ in range(max_steps):

            # 步进
            self.step_simulation(steps=1, sleep=sleep)
            
            # 检查是否收敛
            if self.is_at_joint_positions(target_positions):
                return True
            
        return False
    

    # 控制关节 
    def move_joints(
            self,
            target_positions: dict[str, float],
            sleep: bool = False
    )-> bool:
        clamped_targets = self.clamp_joint_targets(target_positions)
        self.command_joint_positions(clamped_targets)
        return self.wait_until_joint_positions_reached(clamped_targets, sleep=sleep)
    
    # 停止关节控制
    def stop_joints(self) -> None:
        current = self.get_joint_position_map()
        self.command_joint_positions(current)
