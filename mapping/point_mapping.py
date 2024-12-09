import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from typing import List

import mapping.dto as dto
from mapping.geometry_visualizer import get_geometry_visualizer


class PointMapping:
    def __init__(self, camera_intrinsic):
        self._intrinsic = dto.Intrinsic(*camera_intrinsic)
        self._visualizer = get_geometry_visualizer()
        self._prev_frame = None

    def update_map(self, frame: dto.FrameData):
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
        if dist > 0.2:
            rel_pose = tf_pose_to_local(self._prev_frame.pose, frame.pose)
            print(f'rel_pose: {rel_pose}')
            return True
        return False
    
    def _match_features(self, prv_image: np.ndarray, cur_image: np.ndarray):
        """
        param:
            prv_image: previous image (H, W, 3 | uint8)
            cur_image: current image (H, W, 3 | uint8)
        return:
            matches: List[cv2.DMatch]
            prv_features: List[dto.Feature]
            cur_features: List[dto.Feature]
        """
        # TODO: extract keypoints, descriptors -> prv_features, cur_features: List[dto.Feature]
        # TODO: match descriptors -> matches: List[cv2.DMatch]
        # TODO: verify matches -> matches: List[cv2.DMatch]
        # TODO: visualize matches
        return [], [], []
    
    def _triangulate_points(self, matches: List[cv2.DMatch], prv_features: List[dto.Feature], cur_features: List[dto.Feature]):
        """
        param:
            matches: list of cv2.DMatch
            prv_features: list of dto.Feature
            cur_features: list of dto.Feature
        return:
            points3d: 3D points (N, 3 | float)
        """
        # TODO: compute a pair of 3D line parameters for each match ( ax + by + cz + d = 0 )
        # TODO: triangulate points -> points3d: np.ndarray
        return []

    def _visualize_points(self, points3d: np.ndarray, prv_pose: dto.Pose, cur_pose: dto.Pose):
        points3d = np.random.rand(10, 3)  # TODO: this is for test, should be replaced with points3d from triangulation
        self._visualizer.clear_points()
        self._visualizer.clear_lines()
        self._visualizer.draw_points(points3d, color=np.array((0, 0, 255), dtype=np.uint8), size=5)
        self._visualizer.draw_lines(np.tile(prv_pose.posi, (len(points3d), 1)), points3d, color=np.array((0, 0, 255), dtype=np.uint8), thickness=2)
        self._visualizer.draw_lines(np.tile(cur_pose.posi, (len(points3d), 1)), points3d, color=np.array((0, 255, 0), dtype=np.uint8), thickness=2)
        self._visualizer.draw_pose(prv_pose)
        self._visualizer.draw_pose(cur_pose)
        self._visualizer.show()


def tf_pose_to_local(pose1: dto.Pose, pose2: dto.Pose) -> dto.Pose:
    r1 = R.from_quat(pose1.quat)
    r2 = R.from_quat(pose2.quat)
    rel_rotation = r1.inv() * r2
    rotmat = rel_rotation.as_matrix()
    rel_quat = rel_rotation.as_quat()
    rel_posi = r1.inv().apply(pose2.posi - pose1.posi)
    return dto.Pose(posi=rel_posi, quat=rel_quat)