from dataclasses import dataclass

import pybullet as p
import pybullet_data

from sim.robot_loader import LoadedRobot, default_robot_spec, load_robot


# 桌子构建
@dataclass(frozen=True)
class TableSpec:
    size_xyz: tuple[float, float, float] = (0.8, 1.2, 0.05)
    position_xyz: tuple[float, float, float] = (0.35, 0.0, -0.025)


# 盒子构建
@dataclass(frozen=True)
class BoxSpec:
    half_extents_xyz: tuple[float, float, float] = (0.02, 0.02, 0.04)
    position_xyz: tuple[float, float, float] = (0.25, 0.0, 0.04)
    rgba: tuple[float, float, float, float] = (0.9, 0.5, 0.2, 1.0)
    mass: float = 0.05


# 定义返回句柄
@dataclass
class SceneHandles:
    robot: LoadedRobot
    plane_id: int
    table_id: int
    box_ids: list[int]


# 加载一个简单场景 包含一个机器人 一张桌子 和几个盒子
def load_plane(client_id: int) -> int:
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
    plane_id = p.loadURDF("plane.urdf", physicsClientId=client_id)
    
    return plane_id

def create_table(client_id: int, spec: TableSpec) -> int:
    collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[s / 2.0 for s in spec.size_xyz],
        physicsClientId=client_id,
    )
    visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[s / 2.0 for s in spec.size_xyz],
        rgbaColor=[0.4, 0.3, 0.2, 1.0],
        physicsClientId=client_id,
    )
    return p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=spec.position_xyz,
        physicsClientId=client_id,
    )

def create_box(client_id: int, spec: BoxSpec) -> int:
    collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=list(spec.half_extents_xyz),
        physicsClientId=client_id,
    )

    visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=list(spec.half_extents_xyz),
        rgbaColor=list(spec.rgba),
        physicsClientId=client_id,
    )

    return p.createMultiBody(
        baseMass=spec.mass,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=spec.position_xyz,
        physicsClientId=client_id,
    )


# 构建主场景
def build_minimal_scene(client_id: int) -> SceneHandles:
    robot = load_robot(client_id, default_robot_spec())

    plane_id = load_plane(client_id)

    table_spec = TableSpec()
    table_id = create_table(client_id, table_spec)

    box_specs = [
        BoxSpec(position_xyz=(0.22, -0.08, 0.04), rgba=(0.9, 0.4, 0.2, 1.0)),
        BoxSpec(position_xyz=(0.28, 0.00, 0.04), rgba=(0.2, 0.7, 0.3, 1.0)),
        BoxSpec(position_xyz=(0.34, 0.08, 0.04), rgba=(0.2, 0.4, 0.9, 1.0)),
    ]
    box_ids: list[int] = []

    for spec in box_specs:
        box_id = create_box(client_id, spec)
        box_ids.append(box_id)

    return SceneHandles(robot=robot, plane_id=plane_id, table_id=table_id, box_ids=box_ids)