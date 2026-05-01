from dataclasses import dataclass

import pybullet as p

from sim.scene_builder import SceneHandles

# 定义抓取目标结构
@dataclass(frozen=True)
class TopDownGraspTarget:
    box_id: int

    # 箱子中心点
    center_xyz: tuple[float, float, float]

    # 箱子顶面中心点
    top_center_xyz: tuple[float, float, float]

    # 预抓取点
    pregrasp_xyz: tuple[float, float, float] 

    # 固定yaw
    grasp_yaw : float = 0.0


# 单个箱子AABB抓取
def get_box_aabb(client_id: int, box_id: int) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    aabb_min, aabb_max = p.getAABB(box_id, physicsClientId=client_id)
    
    return aabb_min, aabb_max

# 从AABB生成抓取目标
def build_topdown_grasp_target(
        scene: SceneHandles,
        box_id: int,
        pregrasp_height: float = 0.12,
) -> TopDownGraspTarget:
    aabb_min, aabb_max = get_box_aabb(scene.robot.client_id, box_id)

    center_x = 0.5 * (aabb_min[0] + aabb_max[0])
    center_y = 0.5 * (aabb_min[1] + aabb_max[1])
    center_z = 0.5 * (aabb_min[2] + aabb_max[2])

    center_xyz = (center_x, center_y, center_z)

    top_center_xyz = (center_xyz[0], center_xyz[1], aabb_max[2])
    pregrasp_xyz = (center_xyz[0], center_xyz[1], aabb_max[2] + pregrasp_height)

    return TopDownGraspTarget(
        box_id=box_id,
        center_xyz=center_xyz,
        top_center_xyz=top_center_xyz,
        pregrasp_xyz=pregrasp_xyz,
)
    

# 给整组箱子生成目标
def build_all_topdown_grasp_targets(scene: SceneHandles, pregrasp_height: float = 0.12) -> list[TopDownGraspTarget]:
    targets: list[TopDownGraspTarget] = []
    for box_id in scene.box_ids:
        target = build_topdown_grasp_target(scene, box_id, pregrasp_height)
        targets.append(target)
    return targets

