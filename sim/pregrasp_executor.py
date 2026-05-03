from dataclasses import dataclass
import pybullet as p

from sim.grasp_targets import TopDownGraspTarget
from sim.robot_adapter import RobotAdapter
from sim.robot_loader import get_end_effector_pose


# 执行参数
@dataclass
class PregraspConfig:
    # IK位置容差
    position_tolerance: float = 0.01

    # 是否在运动时按真实时间
    sleep:bool = True


# 执行器
@dataclass
class PregraspExecutor:
    adapter: RobotAdapter
    config: PregraspConfig = PregraspConfig()

    # 将IK结果转换为Joint map
    def _ik_solution_to_joint_map(
            self, 
            ik_solution: tuple[float,...] | list[float],
    )-> dict[str, float]:
        
        movable_joints = self.adapter.robot.movable_joints

        if len(ik_solution) < len(movable_joints):
            raise ValueError(f"IK solution length {len(ik_solution)} is less than the number of movable joints {len(movable_joints)}")
        
        joint_map : dict[str, float] = {}

        for joint, angle in zip(movable_joints, ik_solution):
            joint_map[joint.name] = angle

        return joint_map
    
    # PreGrasp IK求解并转为joint map
    def solve_pregrasp_joint_positions(
            self,
            target:TopDownGraspTarget
    )->dict[str, float]:
        
        return self.solve_joint_positions_for_xyz(target.pregrasp_xyz)
    

    # 通用xyz IK接口
    def solve_joint_positions_for_xyz(
            self,
            target_xyz: tuple[float, float, float]
    )->dict[str, float]:

        robot = self.adapter.robot

        ik_solution = p.calculateInverseKinematics(
            robot.body_id,
            robot.end_effector_link_index,
            target_xyz,
            physicsClientId=robot.client_id
        )

        joint_posotions = self._ik_solution_to_joint_map(ik_solution)
        return joint_posotions

    # 执行预抓取位姿
    def execute_pregrasp(
            self,
            target:TopDownGraspTarget
    )->bool:
        return self.execute_xyz(target.pregrasp_xyz)
    
    # 通用xyz执行接口
    def execute_xyz(
            self,
            target_xyz: tuple[float, float, float]
    )->bool:
        joint_positions = self.solve_joint_positions_for_xyz(target_xyz)
        
        reached = self.adapter.move_joints(joint_positions, sleep=self.config.sleep)

        return reached
    
    # PreGrasp 误差检查
    def get_pregrasp_position_error(
            self,
            target:TopDownGraspTarget
    )->tuple[float, float, float]:

        return self.get_position_error(target.pregrasp_xyz)
    
    # 通用xyz误差检查
    def get_position_error(
            self,
            target_xyz: tuple[float, float, float]
    )->tuple[float, float, float]:

        current_pos, _ = get_end_effector_pose(self.adapter.robot)

        dx = current_pos[0] - target_xyz[0]
        dy = current_pos[1] - target_xyz[1]
        dz = current_pos[2] - target_xyz[2]

        return dx, dy, dz