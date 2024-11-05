import sys
import os
import pandas as pd
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from mapping.dto import Pose, FrameData

np.set_printoptions(linewidth=200,  edgeitems=5)
np.set_printoptions(suppress=True, formatter={'float': '{:.5f}'.format})
pd.set_option('display.float_format', '{:.5f}'.format)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 250)
pd.set_option('display.max_colwidth', 20)


class Dataset:
    def __init__(self, data_path):
        self._data_path = data_path
        self._image_list = self._list_images(data_path)
        self._gt_pose = self._read_poses(data_path)

    def __len__(self):
        return len(self._image_list)

    def _list_images(self, data_path):
        img_path = os.path.join(data_path, 'cam0', 'data')
        timestamps = [f[:-4] for f in os.listdir(img_path) if f.endswith('.png')]
        timestamps.sort()
        timestamps = timestamps[30:]
        print(f'png files: len={len(timestamps)}, first_file={timestamps[0]}')
        return timestamps

    def _read_poses(self, data_path):
        pose_path = os.path.join(data_path, 'poses.gt')
        pose = pd.read_csv(pose_path, dtype={'#timestamp [ns]': str})
        renamer = dict(zip(pose.columns, ['timestamp', 'x', 'y', 'z', 'qw', 'qx', 'qy', 'qz']))
        pose = pose.rename(columns=renamer)
        print(f'pose data: shape={pose.shape}\n', pose.head())
        return pose
    
    def get_frame(self, index: int) -> FrameData:
        timestamp = self._image_list[index]
        image = self._get_image_at(timestamp)
        pose = self._get_pose_at(timestamp)
        frame = FrameData(index=index, timestamp=timestamp, image=image, pose=pose)
        cv2.imshow('image', image)
        cv2.waitKey(1)
        return frame

    def _get_image_at(self, timestamp):
        filename = os.path.join(self._data_path, 'cam0', 'data', f'{timestamp}.png')
        print(f'get_image_at: {filename}')
        image = cv2.imread(filename)
        return image

    def _get_pose_at(self, timestamp):
        pose = self._gt_pose.loc[self._gt_pose.timestamp == timestamp, :]
        if pose.empty:
            raise ValueError(f'pose not found for timestamp: {timestamp}')
        else:
            pose = pose.iloc[0]
        posi = np.array([pose.x, pose.y, pose.z], dtype=np.float32)
        quat = np.array([pose.qw, pose.qx, pose.qy, pose.qz], dtype=np.float32)
        pose = Pose(posi=posi, quat=quat)
        print(f'get_pose_at: {pose}')
        return pose


if __name__ == '__main__':
    dataset = Dataset(cfg.DATA_PATH)
    for i in range(len(dataset)):
        frame = dataset.get_frame(i)
        cv2.imshow('image', frame.image)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
