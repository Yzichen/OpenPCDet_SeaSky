import os
import glob
import datetime
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

import torch
import torch.nn as nn
from pcdet.models import build_network, load_data_to_gpu, model_fn_decorator
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils

import copy
import rospy
import ros_numpy
import std_msgs.msg
from geometry_msgs.msg import Point
import sensor_msgs.point_cloud2 as pcl2
from sensor_msgs.msg import PointCloud2, PointField

from visualizer import vis_ros

ros_vis = vis_ros.ROS_MODULE()
last_box_num = 0
last_gtbox_num = 0


def mask_points_out_of_range(pc, pc_range):
    """
    Args:
        pc: (N, 3)
        pc_range: (x_min, y_min, z_min, x_max, y_max, z_max)
    """
    pc_range = np.array(pc_range)
    pc_range[3:6] -= 0.01  # np -> cuda .999999 = 1.0
    mask_x = (pc[:, 0] > pc_range[0]) & (pc[:, 0] < pc_range[3])
    mask_y = (pc[:, 1] > pc_range[1]) & (pc[:, 1] < pc_range[4])
    mask_z = (pc[:, 2] > pc_range[2]) & (pc[:, 2] < pc_range[5])
    mask = mask_x & mask_y & mask_z
    pc = pc[mask]
    return pc


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:
    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa, sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def boxes_to_corners_3d(boxes3d):
    """
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
    Returns:
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d


class ros_demo():
    def __init__(self, model, dataset, args=None):
        self.args = args
        self.model = model
        self.dataset = dataset
        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        # self.offset_angle = 0
        # self.offset_ground = 1.8
        self.point_cloud_range = [0, -20.48, -4, 102.4, 20.48, 2]

    def receive_from_ros(self, msg):
        pc = ros_numpy.numpify(msg)
        points_list = np.zeros((pc.shape[0], 4))
        points_list[:, 0] = copy.deepcopy(np.float32(pc['x']))
        points_list[:, 1] = copy.deepcopy(np.float32(pc['y']))
        points_list[:, 2] = copy.deepcopy(np.float32(pc['z']))
        points_list[:, 3] = copy.deepcopy(np.float32(pc['intensity']))

        # preprocess
        # points_list[:, 2] += points_list[:, 0] * np.tan(self.offset_angle / 180. * np.pi) + self.offset_ground
        rviz_points = copy.deepcopy(points_list)
        points_list = points_list[:, :3]
        points_list = mask_points_out_of_range(points_list, self.point_cloud_range)

        input_dict = {
            'points': points_list,
            'points_rviz': rviz_points
        }

        data_dict = input_dict
        return data_dict

    @staticmethod
    def load_data_to_gpu(batch_dict):
        for key, val in batch_dict.items():
            if not isinstance(val, np.ndarray):
                continue
            else:
                batch_dict[key] = torch.from_numpy(val).float().cuda()

    # @staticmethod
    # def collate_batch(batch_list, _unused=False):
    #     data_dict = defaultdict(list)
    #     for cur_sample in batch_list:
    #         for key, val in cur_sample.items():
    #             data_dict[key].append(val)
    #     batch_size = len(batch_list)
    #     ret = {}
    #     for key, val in data_dict.items():
    #         if key in ['points']:
    #             coors = []
    #             for i, coor in enumerate(val):
    #                 coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
    #                 coors.append(coor_pad)
    #             ret[key] = np.concatenate(coors, axis=0)
    #     ret['batch_size'] = batch_size
    #     return ret

    def online_inference(self, msg):
        data_dict = self.receive_from_ros(msg)
        # data_dict = {
        #     'points': (N, 3),
        #     'points_rviz': (N, 4)
        # }
        data_dict = self.dataset.prepare_data(data_dict)
        data_infer = self.dataset.collate_batch([data_dict])
        ros_demo.load_data_to_gpu(data_infer)

        self.model.eval()
        with torch.no_grad():
            torch.cuda.synchronize()
            self.starter.record()
            pred_dicts = self.model(data_infer)
            self.ender.record()
            torch.cuda.synchronize()
            curr_latency = self.starter.elapsed_time(self.ender)
        print('det_time(ms): ', curr_latency)

        # pred_dicts: Tuple(pred_dicts, recall_dicts)
        pred_dicts = pred_dicts[0]
        # pred_dicts: List[dict{
        #     'pred_boxes': (N_pred, 7),
        #     'pred_scores': (N_pred, ),
        #     'pred_labels': (N_pred, ),
        # }, ...]
        data_infer, pred_dicts = vis_ros.ROS_MODULE.gpu2cpu(data_infer, pred_dicts)

        global last_box_num
        last_box_num, _ = ros_vis.ros_print(data_dict['points_rviz'][:, 0:4], pred_dicts=pred_dicts,
                                            last_box_num=last_box_num)

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/seasky_models/centerpoint.yaml')
    parser.add_argument('--pt', type=str, default='../ckpts/checkpoint_epoch_80.pth', help='checkpoint to start from')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_config()
    cfg_from_yaml_file(args.cfg_file, cfg)
    logger = common_utils.create_logger()
    ros_vis.class_names = cfg.CLASS_NAMES
    dataset, dataloader, sampler = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, workers=1,
                                                    batch_size=1, dist=False, training=False, logger=logger)

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset).cuda()
    model.load_params_from_file(args.pt, logger)
    model.cuda()

    demo_ros = ros_demo(model, dataset, args)
    sub = rospy.Subscriber(
        "/livox/lidar", PointCloud2, queue_size=10, callback=demo_ros.online_inference)
    print("set up subscriber!")

    rospy.spin()