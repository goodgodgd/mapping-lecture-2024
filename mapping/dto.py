from typing import List, Tuple
import numpy as np
from enum import Enum
import dataclasses as dc
import cv2


class CameraMove(Enum):
    Forward = 'd'
    Backward = 'g'
    Left = 'c'
    Right = 'b'
    Upward = 'f'
    Downward = 'v'


@dc.dataclass(unsafe_hash=True)
class Pose:
    posi: np.ndarray  # 카메라 3차원 좌표 (x, y, z)
    quat: np.ndarray  # 카메라 3차원 방향 (quaternion)


@dc.dataclass(unsafe_hash=True)
class Intrinsic:
    fx: float  # focal length
    fy: float
    cx: float  # principal point
    cy: float


@dc.dataclass(unsafe_hash=True)
class FrameData:
    index: int
    timestamp: float
    pose: Pose
    image: np.ndarray
    depth: np.ndarray
    intrinsic: Intrinsic

@dc.dataclass(unsafe_hash=True)
class Feature:
    keypts: List[cv2.KeyPoint]  # key points
    descrs: np.ndarray  # descriptor
