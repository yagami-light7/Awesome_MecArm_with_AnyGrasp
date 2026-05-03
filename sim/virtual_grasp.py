from dataclasses import dataclass
import dataclasses
from gc import enable
from networkx import constraint, join_trees
import pybullet as p
from sim.robot_loader import LoadedRobot, get_end_effector_pose

# 定义返回句柄
@dataclass(frozen=True)
class VirtualGraspHandle:
    box_id : int
    constraint_id : int
    disable_robot_link_indices : list[int]

# 列出robot link index
def _all_robot_link_indices(robot: LoadedRobot) -> list[int]:
    
    return [-1] + [joint.index for joint in robot.joints]

# 将box附着到TCP上
def attach_box_to_tcp(
        robot: LoadedRobot,
        box_id: int,
) -> VirtualGraspHandle:
    
    # 读取当前TCP世界位姿
    tcp_pos, tcp_quat = get_end_effector_pose(robot)

    # 读取box 世界位姿
    box_pos, box_quat = p.getBasePositionAndOrientation(box_id, physicsClientId=robot.client_id)

    #计算TCP与box坐标系的相对变换
    tcp_inv_pos, tcp_inv_quat = p.invertTransform(tcp_pos, tcp_quat)

    parent_to_child_pos, parent_to_child_quat = p.multiplyTransforms(
        tcp_inv_pos, tcp_inv_quat,
        box_pos, box_quat
    )

    # 禁用碰撞
    disable_links = _all_robot_link_indices(robot)
    for link_index in disable_links:
        p.setCollisionFilterPair(
            robot.body_id,
            box_id,
            linkIndexA=link_index,
            linkIndexB=-1,
            enableCollision=0,
            physicsClientId=robot.client_id
        )

    # 创建约束
    constraint_id = p.createConstraint(
        parentBodyUniqueId=robot.body_id,
        parentLinkIndex=robot.end_effector_link_index,
        childBodyUniqueId=box_id,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=(0, 0, 0),
        parentFramePosition=parent_to_child_pos,
        childFramePosition=(0, 0, 0),
        parentFrameOrientation=parent_to_child_quat,
        childFrameOrientation=(0, 0, 0, 1),
        physicsClientId=robot.client_id
    )

    return VirtualGraspHandle(
        box_id=box_id,
        constraint_id=constraint_id,
        disable_robot_link_indices=disable_links
    )


# 解除box与TCP的附着
def detach_box_from_tcp(
        robot: LoadedRobot,
        handle: VirtualGraspHandle
) -> None:
    # 移除约束
    p.removeConstraint(handle.constraint_id, physicsClientId=robot.client_id)

    # 恢复碰撞
    for link_index in handle.disable_robot_link_indices:
        p.setCollisionFilterPair(
            robot.body_id,
            handle.box_id,
            linkIndexA=link_index,
            linkIndexB=-1,
            enableCollision=1,
            physicsClientId=robot.client_id
        )
