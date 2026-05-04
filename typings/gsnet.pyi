from typing import Any

import numpy as np


class AnyGrasp:
    def __init__(self, cfgs: Any) -> None: ...

    def load_net(self) -> None: ...

    def get_grasp(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        lims: list[float],
        apply_object_mask: bool = ...,
        dense_grasp: bool = ...,
        collision_detection: bool = ...,
    ) -> tuple[Any, Any]: ...
