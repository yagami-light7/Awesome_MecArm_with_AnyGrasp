from dataclasses import dataclass
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import sys
import ctypes

# 导入AnyGrasp SDK
def import_anygrasp_sdk():
    project_root = Path(__file__).resolve().parents[1]
    grasp_detection_dir = (
        project_root / "third_party" / "anygrasp_sdk" / "grasp_detection"
    )
    openssl_dir = (
        project_root
        / "third_party"
        / "anygrasp_sdk"
        / "license_registration"
        / "openssl_1_1_local"
        / "usr"
        / "lib"
        / "x86_64-linux-gnu"
    )

    if str(grasp_detection_dir) not in sys.path:
        sys.path.insert(0, str(grasp_detection_dir))

    ctypes.CDLL(str(openssl_dir / "libcrypto.so.1.1"),
    mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL(str(openssl_dir / "libssl.so.1.1"),
    mode=ctypes.RTLD_GLOBAL)

    from gsnet import AnyGrasp
    return AnyGrasp


# 定义AnyGrasp配置数据结构
@dataclass(frozen=True)
class AnyGraspConfig:
    checkpoint_path:Path
    max_gripper_width:float = 0.1
    gripper_height:float = 0.03
    top_down_grasp:bool = True
    debug:bool = False

# 将dataclass转成SDK配置对象
def to_sdk_namespace(cfg: AnyGraspConfig) -> SimpleNamespace:
    max_width = max(0.0, min(0.1, cfg.max_gripper_width))

    return SimpleNamespace(
        checkpoint_path=str(cfg.checkpoint_path),
        max_gripper_width=max_width,
        gripper_height=cfg.gripper_height,
        top_down_grasp=cfg.top_down_grasp,
        debug=cfg.debug,
    )

# AnyGrasp 推理函数
def run_anygrasp(
        points:np.ndarray,
        colors:np.ndarray,
        lims:list[float],
        cfg:AnyGraspConfig,
):
    # 输入检查
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points should be (N, 3), got {points.shape}")
    if colors.ndim != 2 or colors.shape[1] != 3:
        raise ValueError(f"colors should be (N, 3), got {colors.shape}")
    if len(points) != len(colors):
        raise ValueError("points and colors must have the same length")


    # 统一dtype
    points = points.astype(np.float32, copy=False)
    colors = colors.astype(np.float32, copy=False)

    # 延迟导入SDK
    AnyGrasp = import_anygrasp_sdk()
    
    # 初始化并加载模型
    sdk_cfg = to_sdk_namespace(cfg)
    anygrasp = AnyGrasp(sdk_cfg)
    anygrasp.load_net()
    
    # 推理
    gg, cloud = anygrasp.get_grasp(
        points,
        colors,
        lims=lims,
        apply_object_mask=True,
        dense_grasp=False,
        collision_detection=True,
    )

    # 后处理
    if len(gg) > 0:
        gg = gg.nms().sort_by_score()
    
    return gg, cloud
