# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified to Conditional Top Down by Mu Zhou, Lucas Stoffl et al. (ICCV 2023)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints

from .pose_synthesis import synthesize_pose

import matplotlib.pyplot as plt
import os

logger = logging.getLogger(__name__)


class JointsDataset(Dataset):
    def __init__(self, cfg, image_dir, annotation_file, is_train, transform=None):
        self.pixel_std = 200
        self.flip_pairs = []
        self.parent_ids = []
        self.cfg = cfg

        self.is_train = is_train
        self.image_dir = image_dir
        self.annotation_file = annotation_file
        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.colored_kpt = cfg.DATASET.COLORED
        self.kpt_colors = self.get_colors_from_cmap('rainbow', self.num_joints)

        self.stacked_condition = cfg.DATASET.STACKED_CONDITION
        self.bu_bbox_margin = cfg.DATASET.BU_BBOX_MARGIN
        self.bbox_file = cfg.TEST.COCO_BBOX_FILE
        self.use_bu_bbox_test = cfg.TEST.USE_BU_BBOX

        self.best_bu_model_key = 'baseline_resnet_50_s4_60000'

        self.bbox_overlap_for_swapping_noise = cfg.DATASET.SWAP_OVERLAP
        self.synthesis_pose = cfg.DATASET.SYNTHESIS_POSE

        self.output_path = cfg.OUTPUT_DIR
        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY
        self.prob_half_body = cfg.DATASET.PROB_HALF_BODY
        self.color_rgb = cfg.DATASET.COLOR_RGB
        self.new_crop_aug = cfg.DATASET.NEW_AUGMENTATION
        self.bbox_aug = cfg.DATASET.BBOX_AUGMENTATION

        self.condition_topdown = cfg.MODEL.CONDITIONAL_TOPDOWN
        self.target_type = cfg.MODEL.TARGET_TYPE
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
        self.sigma = cfg.MODEL.SIGMA
        self.use_different_joints_weight = cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT
        self.joints_weight = 1

        self.transform = transform
        self.db = []

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def half_body_transform(self, joints, joints_vis):
        upper_joints = []
        lower_joints = []
        for joint_id in range(self.num_joints):
            if joints_vis[joint_id][0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints \
                if len(lower_joints) > 2 else upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio

        scale = np.array(
            [
                w * 1.0 / self.pixel_std,
                h * 1.0 / self.pixel_std
            ],
            dtype=np.float32
        )

        scale = scale * 1.5

        return center, scale

    def __len__(self,):
        return len(self.db)

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']
        # print("item", idx, image_file, db_rec["cond_max_iou"])

        filename = os.path.split(os.path.split(image_file)[0])[1] + '/' + os.path.split(image_file)[1]
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''

        if self.data_format == 'zip':
            from utils import zipreader
            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
        else:
            data_numpy = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )

        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        joints = db_rec['joints_3d']
        joints_vis = db_rec['joints_3d_vis']

        use_bu_bbox = db_rec['use_bu_bbox'] if 'use_bu_bbox' in db_rec else False

        if 'cond_joints' in db_rec.keys():
            conditions = db_rec['cond_joints']
            conditions_vis = db_rec['cond_joints_vis']
            
            # choose randomly one condition (during testing, we take the best condition)
            # if no condition is provided, we set the cond. heatmap to zero
            # if only one condition is provided (e.g. crowdpose), we simply take that one
            if not type(conditions) is dict:
                cond_joints = conditions
                cond_joints_vis = conditions_vis
            elif len(list(conditions)) == 0:
                cond_joints = np.zeros_like(joints)
                cond_joints_vis = np.zeros_like(joints_vis)
            else:
                if not self.is_train:
                    best_model_key = self.best_bu_model_key if not db_rec['best_model_key'] else db_rec['best_model_key']
                    try:
                        cond_joints = conditions[best_model_key]
                        cond_joints_vis = conditions_vis[best_model_key]
                    except:
                        random_key = random.choice(list(conditions))
                        cond_joints = conditions[random_key]
                        cond_joints_vis = conditions_vis[random_key]

                else:
                    random_key = random.choice(list(conditions))
                    cond_joints = conditions[random_key]
                    cond_joints_vis = conditions_vis[random_key]

                ## if use synthesised pose -> replace the original cond_joints
                if self.synthesis_pose and self.is_train:

                    xmin = np.min(cond_joints[:,0][np.nonzero(cond_joints[:,0])]) 
                    ymin = np.min(cond_joints[:,1][np.nonzero(cond_joints[:,1])])
                    xmax = np.max(cond_joints[:,0][np.nonzero(cond_joints[:,0])])
                    ymax = np.max(cond_joints[:,1][np.nonzero(cond_joints[:,1])])
                    w = xmax - xmin
                    h = ymax - ymin
                    area = w * h

                    near_joints = np.array(db_rec['near_joints']).reshape((-1, self.num_joints, 3))
                    synthesized_pose = synthesize_pose(self.cfg, np.array(joints).reshape((-1,3)), np.array(cond_joints).reshape((-1,3)),
                                                        near_joints=near_joints, area=area, num_overlap=0)
                    cond_joints = synthesized_pose

        if use_bu_bbox and cond_joints.sum()!=0 and 'cond_joints' in db_rec.keys():
            xmin = np.min(cond_joints[:,0][np.nonzero(cond_joints[:,0])]) - self.bu_bbox_margin
            ymin = np.min(cond_joints[:,1][np.nonzero(cond_joints[:,1])]) - self.bu_bbox_margin
            xmax = np.max(cond_joints[:,0][np.nonzero(cond_joints[:,0])]) + self.bu_bbox_margin
            ymax = np.max(cond_joints[:,1][np.nonzero(cond_joints[:,1])]) + self.bu_bbox_margin
            xmin = np.clip(xmin, 0, data_numpy.shape[1])
            ymin = np.clip(ymin, 0, data_numpy.shape[0])
            xmax = np.clip(xmax, 0, data_numpy.shape[1])
            ymax = np.clip(ymax, 0, data_numpy.shape[0])
            bbox = [xmin, ymin, xmax-xmin, ymax-ymin]
            c, s = self._xywh2cs(xmin, ymin, xmax-xmin, ymax-ymin)
        else:
            c = db_rec['center']
            s = db_rec['scale']
            bbox = db_rec['bbox']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        if self.is_train:
            if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body
                and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = self.half_body_transform(
                    joints, joints_vis
                )
                if c_half_body is not None and s_half_body is not None:
                    c, s = c_half_body, s_half_body

            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                c[0] = data_numpy.shape[1] - c[0] - 1

                if 'cond_joints' in db_rec.keys():
                    cond_joints, cond_joints_vis = fliplr_joints(
                        cond_joints, cond_joints_vis, data_numpy.shape[1], self.flip_pairs)


        trans = get_affine_transform(c, s, r, self.image_size)
        x, y, w, h = np.array(bbox).astype(int)
        data_numpy_copy = copy.deepcopy(data_numpy)

        if self.new_crop_aug:
            if self.bbox_aug:
                x_delta = w * random.randint(0, 20) // 10
                y_delta = h * random.randint(0, 20) // 10
                x = int(x-x_delta) if x-x_delta>0 else 0
                y = int(y-y_delta) if y-y_delta>0 else 0
                w = int(w + 2*x_delta)
                h = int(h + 2*y_delta)

            H, W = data_numpy_copy.shape[:2]
            data_numpy_copy[0:H,0:x] = 0
            data_numpy_copy[0:y,x:W] = 0
            data_numpy_copy[y+h:H,x:W] = 0
            data_numpy_copy[y:y+h,x+w:W] = 0

        input_ = cv2.warpAffine(
            data_numpy_copy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        if self.transform:
            input = self.transform(input_)

        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)
            if 'cond_joints' in db_rec.keys():
                if cond_joints_vis[i, 0] > 0.0:
                    cond_joints[i, 0:2] = affine_transform(cond_joints[i, 0:2], trans)

        target, target_weight = self.generate_target(joints, joints_vis)
        # cond_heatmap, cond_weight = self.generate_target(cond_joints, cond_joints_vis)

        # ## vis for debugging
        # input_imgs_dir = "/media0/data/temp/"
        # if not os.path.isdir(input_imgs_dir):
        #     os.mkdir(input_imgs_dir)
        # cond_hm = np.transpose(cond_heatmap, (1, 2, 0))
        # cv2.imwrite(f'{input_imgs_dir}/condition_{idx}.jpg', input_ + cond_hm)


        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        if 'annotation_id' in db_rec.keys():
            annotation_id = db_rec['annotation_id'] 
        else:
            annotation_id = -1

        if 'cond_joints' in db_rec.keys() and self.condition_topdown:
        
            if self.stacked_condition:
                cond_heatmap = self.get_stacked_condition(cond_joints, size=(int(self.image_size[1]), int(self.image_size[0])), image=input_)
                cond_heatmap = np.transpose(cond_heatmap, (2, 0, 1))
            elif self.colored_kpt:
                cond_heatmap = self.get_condition_image_colored(cond_joints, size=(int(self.image_size[1]), int(self.image_size[0]), 3), colors=self.kpt_colors, image=input_)
                cond_heatmap = np.transpose(cond_heatmap, (2, 0, 1))
            else:
                cond_heatmap = self.get_condition_image(cond_joints, size=(int(self.image_size[1]), int(self.image_size[0])))

            cond_heatmap = torch.from_numpy(cond_heatmap).float()

            meta = {
                'image': image_file,
                'input_img':input_,
                'filename': filename,
                'imgnum': imgnum,
                'joints': joints,
                'joints_vis': joints_vis,
                'cond_joints': cond_joints,
                'cond_joints_vis': cond_joints_vis,
                'center': c,
                'scale': s,
                'rotation': r,
                'score': score,
                'annotation_id': annotation_id,
                'cond_max_iou': db_rec["cond_max_iou"],
            }
            input = torch.cat((input, cond_heatmap))
        else:
            meta = {
                'image': image_file,
                'input_img':input_,
                'filename': filename,
                'imgnum': imgnum,
                'joints': joints,
                'joints_vis': joints_vis,
                'center': c,
                'scale': s,
                'rotation': r,
                'score': score,
                'annotation_id': annotation_id,
            }
            input = input
        return input, target, target_weight, meta


    def select_data(self, db):
        db_selected = []
        for rec in db:
            num_vis = 0
            joints_x = 0.0
            joints_y = 0.0
            for joint, joint_vis in zip(
                    rec['joints_3d'], rec['joints_3d_vis']):
                if joint_vis[0] <= 0:
                    continue
                num_vis += 1

                joints_x += joint[0]
                joints_y += joint[1]
            if num_vis == 0:
                continue

            joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

            area = rec['scale'][0] * rec['scale'][1] * (self.pixel_std**2)
            joints_center = np.array([joints_x, joints_y])
            bbox_center = np.array(rec['center'])
            diff_norm2 = np.linalg.norm((joints_center-bbox_center), 2)
            ks = np.exp(-1.0*(diff_norm2**2) / ((0.2)**2*2.0*area))

            metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16
            if ks > metric:
                db_selected.append(rec)

        logger.info('=> num db: {}'.format(len(db)))
        logger.info('=> num selected db: {}'.format(len(db_selected)))
        return db_selected

    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

            for joint_id in range(self.num_joints):
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight



    def generate_heatmap(self, heatmap, sigma):
        heatmap = cv2.GaussianBlur(heatmap, sigma, 0)
        am = np.amax(heatmap)
        if am == 0:
            return heatmap
        heatmap /= am / 255
        return heatmap

    def get_colors_from_cmap(self, cmap_name, num_colors):
        cmap = plt.get_cmap(cmap_name)
        colors_float = [cmap(i) for i in range(0, 256, 256 // num_colors)]
        colors = [(int(r*255), int(g*255), int(b*255)) for r, g, b, _ in colors_float]
        return colors

    def get_stacked_condition(self, kpts, size, image):

        kpts = np.array(kpts).astype(int)#.reshape(-1, 2).astype(int)
        zero_matrix = np.zeros(size)

        def _get_condition_matrix(zero_matrix, kpt):
            if 0 < kpt[0] < size[1] and 0 < kpt[1] < size[0] :
                zero_matrix[kpt[1]-1][kpt[0]-1] = 255
            return zero_matrix

        condition_heatmap_list = []
        for i, kpt in enumerate(kpts):
            condition = _get_condition_matrix(zero_matrix, kpt)
            condition_heatmap = self.generate_heatmap(condition, sigma=(15, 15))
            condition_heatmap_list.append(condition_heatmap)
            zero_matrix = np.zeros(size)

            # ### debug: visualization -> check conditions
            # condition_heatmap = np.expand_dims(condition_heatmap, axis=0)
            # condition = np.repeat(condition_heatmap, 3, axis=0)
            # print("condition", condition.shape)
            # condition = np.transpose(condition, (1, 2, 0))
            # cv2.imwrite(f'/media/data/mu/test/cond_{i}.jpg', condition+image)
            # cv2.imwrite(f'/media/data/mu/test/image.jpg', image)

        condition_heatmap_list = np.moveaxis(np.array(condition_heatmap_list), 0, -1)

        return condition_heatmap_list

    def get_condition_image(self, kpts, size):

        kpts = np.array(kpts).astype(int)#.reshape(-1, 2).astype(int)
        zero_matrix = np.zeros(size)

        def _get_condition_matrix(zero_matrix, kpts):
            for kpt in kpts:
                if 0 < kpt[0] < size[1] and 0 < kpt[1] < size[0]:
                    zero_matrix[kpt[1]-1][kpt[0]-1] = 255
            return zero_matrix

        condition = _get_condition_matrix(zero_matrix, kpts)
        _condition_heatmap = self.generate_heatmap(condition, sigma=(15, 15))
        _condition_heatmap = np.expand_dims(_condition_heatmap, axis=0)
        condition_heatmap = np.repeat(_condition_heatmap, 3, axis=0).astype(int)

        return condition_heatmap


    def get_condition_image_colored(self, kpts, size, colors=None, image=None):

        kpts = np.array(kpts).astype(int)#.reshape(-1, 2).astype(int)
        zero_matrix = np.zeros(size)

        def _get_condition_matrix(zero_matrix, kpts):
            for color, kpt in zip(colors, kpts):
                if 0 < kpt[0] < size[1] and 0 < kpt[1] < size[0]:
                    zero_matrix[kpt[1]-1][kpt[0]-1] = color
            return zero_matrix

        condition = _get_condition_matrix(zero_matrix, kpts)
        condition_heatmap = self.generate_heatmap(condition, sigma=(15, 15))
        # cv2.imwrite('/home/mu/documents/test.png', condition_heatmap)

        # ## vis for debugging
        # input_imgs_dir = "/media/data/mu/temp2/"
        # if not os.path.isdir(input_imgs_dir):
        #     os.mkdir(input_imgs_dir)
        # # cond_hm = np.transpose(cond_heatmap, (1, 2, 0))
        # cv2.imwrite(f'{input_imgs_dir}/condition_image.jpg',image)
        # cv2.imwrite(f'{input_imgs_dir}/condition_hm.jpg',condition_heatmap)
        # cv2.imwrite(f'{input_imgs_dir}/condtion_test.jpg', image + condition_heatmap)

        return condition_heatmap


    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            # scale = scale * 1.25
            scale = scale * self.scale_thre

        return center, scale


    ## bbox format = x, y, w, h
    def compute_iou(self, bbox_1, bbox_2):

        x1_l = bbox_1[0]
        x1_r = bbox_1[0] + bbox_1[2]
        y1_t = bbox_1[1]
        y1_b = bbox_1[1] + bbox_1[3]
        w1   = bbox_1[2]
        h1   = bbox_1[3]

        x2_l = bbox_2[0]
        x2_r = bbox_2[0] + bbox_2[2]
        y2_t = bbox_2[1]
        y2_b = bbox_2[1] + bbox_2[3]
        w2   = bbox_2[2]
        h2   = bbox_2[3]

        xi_l = max(x1_l, x2_l)
        xi_r = min(x1_r, x2_r)
        yi_t = max(y1_t, y2_t)
        yi_b = min(y1_b, y2_b)

        width  = max(0, xi_r - xi_l)
        height = max(0, yi_b - yi_t)
        a1 = w1 * h1
        a2 = w2 * h2

        if float(a1 + a2 - (width * height)) == 0:
            return 0
        else:
            iou = (width * height) / float(a1 + a2 - (width * height))

        return iou
