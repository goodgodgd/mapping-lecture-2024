import sys
import os
import pyvista as pv
import numpy as np
import time
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mapping.dto as dto
import config as cfg


class GeometryVisualizer:
    def __init__(self):
        self.plotter = pv.Plotter()
        self.point_actors = []
        self.line_actors = []
        self.pose_actors = []
        self.draw_axes()

    def draw_axes(self):
        self.plotter.add_axes()
        origin = np.array([0, 0, 0])
        x_arrow = np.array([1, 0, 0])
        y_arrow = np.array([0, 1, 0])
        z_arrow = np.array([0, 0, 1])
        self.plotter.add_arrows(origin, x_arrow, color='red')  # X축: 빨간색
        self.plotter.add_arrows(origin, y_arrow, color='green')  # Y축: 녹색
        self.plotter.add_arrows(origin, z_arrow, color='blue')  # Z축: 파란색

    def draw_points(self, point: np.ndarray, color: np.ndarray, size: float):
        """
        point: shape=(N, 3) or (3,), 점의 3차원 좌표 (X,Y,Z)
        color: shape=(3,) 점의 색상 (R,G,B), (0~1)사이의 값을 가진 float 타입 이거나 0~255 값을 가진 uint8 타입
        size: 점의 크기
        """
        if point.size == 0:
            return
        if color.dtype == np.uint8:
            color = color / 255.0
        actor = self.plotter.add_points(point, color=color, point_size=size, render_points_as_spheres=True)
        self.point_actors.append(actor)
        self.plotter.render()

    def clear_points(self):
        for actor in self.point_actors:
            self.plotter.remove_actor(actor)
        self.point_actors.clear()

    def draw_lines(self, pt1: np.ndarray, pt2: np.ndarray, color: np.ndarray, thickness: float):
        """
        pt1: shape=(N,3) or (3,), 시작점의 3차원 좌표
        pt2: shape=(N,3) or (3,), 끝점의 3차원 좌표
        color: shape=(3,), 선분의 색상 (R,G,B), (0~1)사이의 값을 가진 float 타입 이거나 0~255 값을 가진 uint8 타입
        thickness: 선의 두께
        """
        if color.dtype == np.uint8:
            color = color / 255.0
        if pt1.ndim == 1:
            line = np.concat([pt1, pt2], axis=0).reshape((-1, 3))
        elif pt1.ndim == 2:
            line = np.concat([pt1, pt2], axis=1).reshape((-1, 3))
        else:
            raise ValueError('pt1.ndim > 2')
        actors = self.plotter.add_lines(line, color=color, width=thickness)
        self.line_actors.append(actors)
        self.plotter.render()
        return actors
    
    def clear_lines(self):
        for actor in self.line_actors:
            self.plotter.remove_actor(actor)
        self.line_actors.clear()

    def draw_pose(self, pose: dto.Pose):
        axis_length = 0.2
        rotation = R.from_quat(pose.quat).as_matrix()
        x_axis = pose.posi + rotation @ np.array([axis_length, 0, 0])
        y_axis = pose.posi + rotation @ np.array([0, axis_length, 0])
        z_axis = pose.posi + rotation @ np.array([0, 0, axis_length])
        actor1 = self.draw_lines(pose.posi, x_axis, color=np.array([255, 0, 0], dtype=np.uint8), thickness=2)
        actor2 = self.draw_lines(pose.posi, y_axis, color=np.array([0, 255, 0], dtype=np.uint8), thickness=2)
        actor3 = self.draw_lines(pose.posi, z_axis, color=np.array([0, 0, 255], dtype=np.uint8), thickness=2)
        self.pose_actors.append((actor1, actor2, actor3))

    def show(self):
        self.plotter.show(interactive_update=True)

    def update(self):
        self.plotter.update()

    def move_camera(self, move: dto.CameraMove):
        step = 0.1
        focal_dist = 8
        position, focal_point, view_up = self.plotter.camera_position
        position = np.array(position)
        focal_point = np.array(focal_point)
        view_up = np.array(view_up)
        print('position:', position)
        print('focal_point:', focal_point)
        print('view_up:', view_up)

        view_front = focal_point - position
        view_front /= np.linalg.norm(view_front)
        view_left = np.cross(view_up, view_front)
        view_left /= np.linalg.norm(view_left)
        view_upwd = np.cross(view_front, view_left)
        view_upwd /= np.linalg.norm(view_upwd)

        if move == dto.CameraMove.Forward:
            position += step * view_front
        elif move == dto.CameraMove.Backward:
            position -= step * view_front
        elif move == dto.CameraMove.Left:
            position += step * view_left
        elif move == dto.CameraMove.Right:
            position -= step * view_left
        elif move == dto.CameraMove.Upward:
            position += step * view_upwd
        elif move == dto.CameraMove.Downward:
            position -= step * view_upwd
        focal_point = position + focal_dist * view_front

        position = tuple(position.tolist())
        focal_point = tuple(focal_point.tolist())
        view_up = tuple(view_up.tolist())
        self.plotter.camera_position = (position, focal_point, view_up)


class Dummy(GeometryVisualizer):
    def __init__(self):
        pass

    def draw_axes(self):
        pass

    def draw_point(self, point: np.ndarray, color: np.ndarray, size: float):
        pass

    def draw_line(self, pt1: np.ndarray, pt2: np.ndarray, color: np.ndarray, thickness: float):
        return 1

    def draw_pose(self, pose: dto.Pose):
        pass

    def trim_pose(self):
        pass

    def show(self):
        pass

    def update(self):
        pass

    def clear_points(self):
        pass

    def move_camera(self, move: dto.CameraMove):
        pass


visualizer = GeometryVisualizer()


def get_geometry_visualizer():
    global visualizer
    return visualizer


if __name__ == "__main__":
    vis = GeometryVisualizer()

    # 여러 점 그리기
    points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    color = np.array([255, 0, 0], dtype=np.uint8)  # 빨간색
    size = 10
    vis.draw_point(points, color, size)

    # 여러 선 그리기
    pt1 = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32)
    pt2 = np.array([[1, 0, 0], [2, 1, 1], [3, 2, 2]], dtype=np.float32)
    color = np.array([0, 255, 0], dtype=np.uint8)  # 빨간색
    thickness = 5
    vis.draw_line(pt1, pt2, color, thickness)

    # 시각화 표시
    vis.show()

    cur_time = time.time()
    updated = False
    while True:
        vis.update()
        time.sleep(0.1)
        if updated is False and time.time() - cur_time >= 3:
            vis.draw_point(points+1, color, size)
