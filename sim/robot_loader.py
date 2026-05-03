'''
URDF加载机械臂并提供关节和末端位姿接口
'''

from dataclasses import dataclass
from pathlib import Path

import pybullet as p

# 区分可控关节 urdf中一些固定关节用于关节零点偏置修正
MOVABLE_JOINT_TYPES = {
    p.JOINT_REVOLUTE,
    p.JOINT_PRISMATIC,
}

# Home Positions
DEFAULT_HOME_JOINT_POSITIONS = {"Joint1": 0.0, 
                            "Joint2": 0.0, 
                            "Joint3": 0.0, 
                            "Joint4": 0.0, 
                            "Joint5": 0.0, 
                            "Joint6": 0.0}

# 加载配置
@dataclass(frozen=True)
class RobotSpec:
    # urdf 文件路径
    urdf_path: Path

    # 解析mesh路径用到的搜索根目录
    search_root: Path

    # 机械臂基座在世界坐标系的位置
    base_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # 机械臂基座在世界坐标系的欧拉角（弧度） 后转为四元数
    base_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # 默认固定底座
    use_fixed_base: bool = True

    # 关闭自碰撞
    use_self_collision: bool = False

    # 末端执行器
    end_effector_link_name: str = "TCP_link"

# 关节结构化信息
@dataclass(frozen=True)
class JointModel:
    # joint index in pybullet关节索引
    index: int

    # joint name from urdf关节名称
    name: str

    # joint type关节类型
    joint_type: int

    # joint limits关节限位
    lower_limit: float
    upper_limit: float

    # URDF中力矩和速度限制
    max_force: float
    max_velocity: float

    # joint指向child link的名称
    child_link_name: str


# 加载完成之后的机器人对象描述
@dataclass(frozen=True)
class LoadedRobot:
    # 当前PyBullet client ID
    client_id: int

    # 当前机器人body ID
    body_id: int

    # 加载时使用的规格定义
    spec: RobotSpec

    # 关节列表 包括所有关节
    joints: list[JointModel]

    # 可控关节列表 只保留revolute和prismatic关节
    movable_joints: list[JointModel]

    @property
    def movable_joint_indices(self) -> list[int]:
        return [joint.index for joint in self.movable_joints]
    
    @property
    def joint_name_to_index(self) -> dict[str, int]:
        return {joint.name: joint.index for joint in self.joints}
    
    @property
    def link_name_to_joint_index(self) -> dict[str, int]:
        return{
            joint.child_link_name:joint.index
            for joint in self.joints
        }
    
    @property
    def end_effector_link_index(self) -> int:
        try:
            return self.link_name_to_joint_index[self.spec.end_effector_link_name]
        except KeyError as exc:
            raise ValueError(f"End effector link not found: {self.spec.end_effector_link_name}") from exc


# 默认机器人路径解析
def default_robot_spec() -> RobotSpec:
    # 当前文件是 sim/robot_loader.py
    # parents[1] 刚好回到项目根目录 /home/light/workspace/mecarm
    project_root = Path(__file__).resolve().parents[1]

    return RobotSpec(
        urdf_path=project_root / "mec_arm_model" / "urdf" / "mec_arm.urdf",
        search_root=project_root,
        base_xyz=(0.0, 0.0, 0.0),
        base_rpy=(0.0, 0.0, 0.0),
        use_fixed_base=True,
        use_self_collision=False,
    )


# 连接PyBullet
def connect_pybullet(gui: bool = True) -> int:
    # GUI模式用于观察 Direct模式用于批处理和单元测试
    connection_mode = p.GUI if gui else p.DIRECT

    # 建立仿真连接
    client_id = p.connect(connection_mode)

    if client_id < 0:
        raise RuntimeError("Failed to connect to PyBullet")
    
    # 预设重力
    p.setGravity(0, 0, -9.81, physicsClientId=client_id)

    return client_id


# 读取Joint信息
def _read_joints_models(client_id: int, body_id: int) -> list[JointModel]:
    joints: list[JointModel] = []

    num_joints = p.getNumJoints(body_id, physicsClientId=client_id)

    for joint_index in range(num_joints):
        info = p.getJointInfo(body_id, joint_index, physicsClientId=client_id)

        '''  
        - info[1]：joint name
        - info[2]：joint type
        - info[8]：lower limit
        - info[9]：upper limit
        - info[12]：child link name
        '''
        joints.append(
            JointModel(
                index = info[0],
                name = info[1].decode("utf-8"),
                joint_type = info[2],
                lower_limit = float(info[8]),
                upper_limit = float(info[9]),
                max_force = float(info[10]),
                max_velocity = float(info[11]),
                child_link_name = info[12].decode("utf-8"),
            )
        )
    return joints

# 导入机器人urdf
def load_robot(client_id: int, spec: RobotSpec) -> LoadedRobot:
    # 检查URDF文件和搜索路径
    if not spec.urdf_path.is_file():
        raise FileNotFoundError(f"URDF file not found: {spec.urdf_path}")
    
    if not spec.search_root.is_dir():
        raise NotADirectoryError(f"Search root is not a directory: {spec.search_root}")

    # 添加搜索路径 以便pybullet能找到urdf里引用的mesh资源    
    p.setAdditionalSearchPath(str(spec.search_root), physicsClientId=client_id)

    flags = 0
    if spec.use_self_collision:
        flags |= p.URDF_USE_SELF_COLLISION

    base_quat = p.getQuaternionFromEuler(spec.base_rpy)

    body_id = p.loadURDF(
        str(spec.urdf_path),
        basePosition=spec.base_xyz,
        baseOrientation=base_quat,
        useFixedBase=spec.use_fixed_base,
        flags=flags,
        physicsClientId=client_id,
    )

    # 读取关节信息
    joints = _read_joints_models(client_id, body_id)

    # 筛选出可动关节
    movable_joints = [joint for joint in joints if joint.joint_type in MOVABLE_JOINT_TYPES]

    return LoadedRobot(
        client_id=client_id,
        body_id=body_id,
        spec=spec,
        joints=joints,
        movable_joints=movable_joints,
    )


# 读取link位姿
def get_link_pose(robot: LoadedRobot, link_index: int) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    
    state = p.getLinkState(robot.body_id, link_index, physicsClientId=robot.client_id)

    world_position = state[4]  # 世界坐标系下的位置
    world_orientation = state[5]  # 世界坐标系下的四元数

    return world_position, world_orientation


def get_link_com_pose(
    robot: LoadedRobot,
    link_index: int,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    state = p.getLinkState(robot.body_id, link_index, physicsClientId=robot.client_id)

    world_position = state[0]
    world_orientation = state[1]

    return world_position, world_orientation


# 读取末端位姿
def get_end_effector_pose(robot: LoadedRobot) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    end_effector_index = robot.end_effector_link_index
    return get_link_pose(robot, end_effector_index)


# 重置关节
def reset_arm_joints(
        robot:LoadedRobot,
        joint_positions:dict[str, float] | None = None,
) -> None:
    target_angles = dict(DEFAULT_HOME_JOINT_POSITIONS)
    if joint_positions is not None:
        target_angles.update(joint_positions)
    
    for joint in robot.movable_joints:
        if joint.name not in target_angles:
            continue
        
        target = target_angles[joint.name]
        clamped_target = min(max(target, joint.lower_limit), joint.upper_limit)

        p.resetJointState(
            robot.body_id,
            joint.index,
            targetValue=clamped_target,
            physicsClientId=robot.client_id,
        )
