import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from typing import List

import mapping.dto as dto
from mapping.geometry_visualizer import get_geometry_visualizer


class PointMapping:
    def __init__(self, camera_intrinsic):
        self._map_points: List[dto.MapPoint] = []  # 점 지도 데이터
        self._intrinsic = dto.Intrinsic(*camera_intrinsic)
        self._visualizer = get_geometry_visualizer()
        self.frame_id = -1
        self._im_shape = ()
        self._prev_frame = None

    def update_map(self, frame: dto.FrameData):
        self.frame_id = frame.index
        self._im_shape = frame.image.shape
        if not self._check_pose_change(frame):
            return

        matches, prv_features, cur_features = self._match_features(self._prev_frame.image, frame.image)
        points3d = self._triangulate_points(matches, prv_features, cur_features)
        self._visualize_points(points3d, self._prev_frame.pose, frame.pose)
        self._prev_frame = frame
        print('update frame')

    def _check_pose_change(self, frame: dto.FrameData):
        if self._prev_frame is None:
            self._prev_frame = frame
            return False
        dist = np.linalg.norm(frame.pose.posi - self._prev_frame.pose.posi)
        print(f'pose dist: {dist}')
        if dist > 0.1:
            return True
        return False
    
    def _match_features(self, prv_image: np.ndarray, cur_image: np.ndarray):
        return [], [], []
    
    def _triangulate_points(self, matches: List[cv2.DMatch], prv_features: List[dto.Feature], cur_features: List[dto.Feature]):
        return []

    def _visualize_points(self, points3d: np.ndarray, prv_pose: dto.Pose, cur_pose: dto.Pose):
        points3d = np.random.rand(10, 3)
        self._visualizer.clear_points()
        self._visualizer.clear_lines()
        self._visualizer.draw_points(points3d, color=np.array((0, 0, 255), dtype=np.uint8), size=5)
        self._visualizer.draw_lines(np.tile(prv_pose.posi, (len(points3d), 1)), points3d, color=np.array((0, 0, 255), dtype=np.uint8), thickness=2)
        self._visualizer.draw_lines(np.tile(cur_pose.posi, (len(points3d), 1)), points3d, color=np.array((0, 255, 0), dtype=np.uint8), thickness=2)
        self._visualizer.draw_pose(prv_pose)
        self._visualizer.draw_pose(cur_pose)
        self._visualizer.show()

