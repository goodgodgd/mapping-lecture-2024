import os
import sys
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

import_path = os.path.dirname(os.path.dirname(__file__))
if import_path not in sys.path:
    sys.path.append(import_path)

import config as cfg
import mapping.dto as dto
from mapping.dataset import Dataset
from mapping.point_mapping import PointMapping
from mapping.geometry_visualizer import get_geometry_visualizer

auto_frame = False


def point_mapping_main():
    dataset = Dataset(cfg.DATA_PATH)
    mapper = PointMapping(cfg.CAMERA_INTRINSIC)

    for index in range(len(dataset)):
        print(f'\n========== frame: {index} ==========')
        frame = dataset.get_frame(index)
        mapper.update_map(frame)
        show_frame_result(frame)

    print('mapping DONE')
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_frame_result(frame: dto.FrameData):
    global auto_frame
    vis = get_geometry_visualizer()
    vis.show()

    while True:
        vis.update()
        key = cv2.waitKey(1)
        if key == -1:
            if auto_frame:
                break
            continue

        key = chr(key)
        if key in dto.CameraMove._value2member_map_:
            move = dto.CameraMove(key)
            print(f'key: {key}, move: {move}')
            vis.move_camera(move)
        elif key == 'n':
            break
        elif key == 'q':
            exit(0)
        elif key == 'a':
            auto_frame = not auto_frame
            print(f'toggle auto_frame: {auto_frame}')


if __name__ == '__main__':
    point_mapping_main()
