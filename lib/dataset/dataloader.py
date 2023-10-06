# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified to Conditional Top Down by Mu Zhou, Lucas Stoffl et al. (ICCV 2023)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from cmath import pi

from collections import defaultdict
from collections import OrderedDict
import logging
import os
import random
import pandas as pd

import cv2 
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json_tricks as json
import pickle
import numpy as np

from dataset.JointsDataset import JointsDataset
from nms.nms import oks_nms
from nms.nms import soft_oks_nms
from nms.nms import oks_merge


logger = logging.getLogger(__name__)


class DataLoader(JointsDataset):
    '''
    "keypoints": {}
    "skeleton": []
    '''
    def __init__(self, cfg, image_dir, annotation_file, is_train, transform=None):
        super().__init__(cfg, image_dir, annotation_file, is_train, transform)
        
        self.nms_thre = cfg.TEST.NMS_THRE
        self.image_thre = cfg.TEST.IMAGE_THRE
        self.soft_nms = cfg.TEST.SOFT_NMS
        self.oks_thre = cfg.TEST.OKS_THRE
        self.in_vis_thre = cfg.TEST.IN_VIS_THRE
        self.bbox_file = cfg.TEST.COCO_BBOX_FILE
        self.use_gt_bbox = cfg.TEST.USE_GT_BBOX
        self.use_bu_bbox_train = cfg.TRAIN.USE_BU_BBOX
        self.use_bu_bbox_test = cfg.TEST.USE_BU_BBOX
        self.test_gt_file = cfg.DATASET.TEST_ANNOTATION_FILE
        self.img_dir = cfg.DATASET.TRAIN_IMAGE_DIR if self.is_train else cfg.DATASET.TEST_IMAGE_DIR
        self.condition_topdown = cfg.MODEL.CONDITIONAL_TOPDOWN
        self.bbox_overlap_for_swapping_noise = False

        self.mode = 'train' if self.is_train else 'test'
        
        self.image_width = cfg.MODEL.IMAGE_SIZE[0]
        self.image_height = cfg.MODEL.IMAGE_SIZE[1]
        self.balanced_dataset = cfg.DATASET.BALANCED
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.pixel_std = 200
        self.scale_thre = cfg.TEST.SCALE_THRE


        self.coco = COCO(self._get_ann_file_keypoint())

        # deal with class names
        cats = [cat['name']
                for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats
        logger.info('=> classes: {}'.format(self.classes))
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict(
            [
                (self._class_to_coco_ind[cls], self._class_to_ind[cls])
                for cls in self.classes[1:]
            ]
        )

        # load image file names
        self.image_set_index = self._load_image_set_index()
        self.num_images = len(self.image_set_index)
        logger.info('=> num_images: {}'.format(self.num_images))

    def _get_ann_file_keypoint(self):
        return self.annotation_file

    def _load_image_set_index(self):
        """ image id: int """
        image_ids = self.coco.getImgIds()
        return image_ids

    def _get_db(self):
                
        if self.is_train:
            ## during training
            if self.use_bu_bbox_train:
                print("load bu derived bbox for train set")
                gt_db = self._load_coco_keypoint_annotations(bu_bbox=True)
            elif self.use_gt_bbox:
                # use ground truth bbox
                print("load gt bbox for train set")
                gt_db = self._load_coco_keypoint_annotations()
        
        else:
            ## during testing
            if self.use_bu_bbox_test and self.condition_topdown:
                # use bu bbox from predictions
                if self.bbox_file == '':
                    print("load bu bbox for testing")
                    gt_db = self._load_coco_keypoint_annotations(bu_bbox=True, best_model_key='baseline_resnet_50_s4_60000')
                else:
                    # use bbox from bu model
                    print("load BU bbox from result json")
                    gt_db = self._load_coco_person_BU_detection_results()
            else:
                ## load detector bbox (no conditions)
                print("load bbox from detector")
                gt_db = self._load_coco_person_detection_results()
        return gt_db
    

    def _load_coco_keypoint_annotations(self, bu_bbox=False, best_model_key='baseline_resnet_50_ms4_60000'):
        """ ground truth bbox and keypoints """
        gt_db = []
        for index in self.image_set_index:
            gt_db.extend(self._load_coco_keypoint_annotation_kernal(index, bu_bbox, best_model_key))
        return gt_db


    def _load_coco_keypoint_annotation_kernal(self, index, bu_bbox=False, best_model_key='baseline_resnet_50_ms4_60000'):
        """
        coco ann: [u'segmentation', u'area', u'iscrowd', u'image_id', u'bbox', u'category_id', u'id']
        iscrowd:
            crowd instances are handled by marking their overlaps with all categories to -1
            and later excluded in training
        bbox:
            [x1, y1, w, h]
        :param index: coco image id
        :return: db entry
        """
        im_ann = self.coco.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']

        annIds = self.coco.getAnnIds(imgIds=index, iscrowd=False)
        objs = self.coco.loadAnns(annIds)

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            x, y, w, h = obj['bbox']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2-x1, y2-y1]
                valid_objs.append(obj)
        objs = valid_objs

        rec = []
        for obj in objs:
            cls = self._coco_ind_to_class_ind[obj['category_id']]
            if cls != 1:
                continue

            # ignore objs without keypoints annotation
            if max(obj['keypoints']) == 0:
                continue

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float)
            for ipt in range(self.num_joints):
                joints_3d[ipt, 0] = obj['keypoints'][ipt * 3 + 0]
                joints_3d[ipt, 1] = obj['keypoints'][ipt * 3 + 1]
                joints_3d[ipt, 2] = 0
                t_vis = obj['keypoints'][ipt * 3 + 2]
                if t_vis > 1:
                    t_vis = 1
                joints_3d_vis[ipt, 0] = t_vis
                joints_3d_vis[ipt, 1] = t_vis
                joints_3d_vis[ipt, 2] = 0

            if 'cond_kpts' in obj.keys():
                cond_joints_3d_vis = dict()
                cond_joints_3d = dict()
                for k, cond in obj['cond_kpts'].items():
                    cond_joints_ = np.zeros((self.num_joints, 3), dtype=np.float)
                    cond_joints_vis_ = np.zeros((self.num_joints, 3), dtype=np.float)
                    for ipt in range(self.num_joints):
                        cond_joints_[ipt, 0] = cond[ipt * 3 + 0]
                        cond_joints_[ipt, 1] = cond[ipt * 3 + 1]
                        cond_joints_[ipt, 2] = 0
                        t_vis = cond[ipt * 3 + 2]
                        #if t_vis > 1:
                        if sum(cond_joints_[ipt]) > 0:
                            t_vis = 1
                        else:
                            t_vis = 0
                        cond_joints_vis_[ipt, 0] = t_vis
                        cond_joints_vis_[ipt, 1] = t_vis
                        cond_joints_vis_[ipt, 2] = 0
                    cond_joints_3d[k] = cond_joints_
                    cond_joints_3d_vis[k] = cond_joints_vis_

                
                if 'bbox_overlaps' in obj.keys():
                    if type(obj['bbox_overlaps']) is dict:
                        max_iou = max(list(obj['bbox_overlaps'].values())) if len(obj['bbox_overlaps'])!=0 else 0
                        #near_ids = [int(k) for k, v in list(obj['bbox_overlaps'].items()) if v >= 0.1]
                        if not self.bbox_overlap_for_swapping_noise:
                            near_joints = [np.array(ob['keypoints']).reshape((-1,3)) for ob in objs]
                        else:
                            #near_ids = [int(k) for k, v in list(obj['bbox_overlaps'].items()) if v >= self.bbox_overlap_for_swapping_noise]
                            #near_joints = [np.array(ob['keypoints']).reshape((-1,3)) for ob in objs if ob['id'] in near_ids]
                            raise NotImplementedError('')
                        #if len(near_ids) == 0:
                        if len(near_joints) == 0:
                            near_joints = [np.zeros([self.num_joints, 3])]
                    else:
                        max_iou = max(obj['bbox_overlaps'])
                        near_joints = [np.zeros((self.num_joints, 3))]
                else:
                    max_iou = 0
                    if not self.bbox_overlap_for_swapping_noise:
                        near_joints = [np.array(ob['keypoints']).reshape((-1,3)) for ob in objs]
                    else:
                        near_joints = [np.zeros((self.num_joints, 3))]



            center, scale = self._box2cs(obj['clean_bbox'][:4])
            image_file_name = im_ann['file_name']
            image_path = os.path.join(self.image_dir, image_file_name)

            if 'cond_kpts' in obj.keys():
                rec.append({
                    'image': image_path,
                    'center': center,
                    'scale': scale,
                    'joints_3d': joints_3d,
                    'joints_3d_vis': joints_3d_vis,
                    'cond_joints': cond_joints_3d,
                    'cond_joints_vis' : cond_joints_3d_vis,
                    'use_bu_bbox': bu_bbox,
                    'filename': '',
                    'imgnum': 0,
                    'annotation_id': obj['id'],
                    'cond_max_iou': max_iou,
                    'near_joints': near_joints,
                    'bbox': obj['clean_bbox'][:4],
                    'best_model_key': best_model_key, 
                    'image_id': obj['image_id'],
                })
            else:
                rec.append({
                    'image': image_path,
                    'center': center,
                    'scale': scale,
                    'joints_3d': joints_3d,
                    'joints_3d_vis': joints_3d_vis,
                    'filename': '',
                    'imgnum': 0,
                    'annotation_id': obj['id'],
                    'bbox': obj['clean_bbox'][:4],
                    'img_id': obj['image_id'],
                })
        return rec

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

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


    # load detections by using pose predicitions from BU model (during test time)
    def _load_coco_person_BU_detection_results(self):
        all_preds = None
        with open(self.bbox_file, 'r') as f:
            all_preds = json.load(f)

        if not all_preds:
            logger.error('=> Load %s fail!' % self.bbox_file)
            return None

        kpt_db = []
        for img_pred in all_preds:

            if 'preds' not in img_pred.keys():
                kpt_db = self._load_coco_pose_results()
                return kpt_db

            preds = img_pred['preds']
            scores = img_pred['scores']
            img_name = img_pred['image_paths'][0]

            all_boxes = []
            all_cond_joints = []
            all_cond_joints_vis = []
            for i in range(len(preds)):
                pred = np.array(preds[i])
                cond_joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
                cond_joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float)
                cond_joints_3d[:,:2] = pred[:,:2]
                cond_joints_3d_vis[:,0] = pred[:,2]
                cond_joints_3d_vis[:,1] = pred[:,2]
                all_cond_joints.append(cond_joints_3d)
                all_cond_joints_vis.append(cond_joints_3d_vis)
                xmin = np.min(cond_joints_3d[:,0][np.nonzero(cond_joints_3d[:,0])]) - self.bu_bbox_margin
                ymin = np.min(cond_joints_3d[:,1][np.nonzero(cond_joints_3d[:,1])]) - self.bu_bbox_margin
                xmax = np.max(cond_joints_3d[:,0][np.nonzero(cond_joints_3d[:,0])]) + self.bu_bbox_margin
                ymax = np.max(cond_joints_3d[:,1][np.nonzero(cond_joints_3d[:,1])]) + self.bu_bbox_margin
                box = [xmin, ymin, xmax-xmin, ymax-ymin]
                all_boxes.append(box)


            for i in range(len(preds)):
                score = scores[i]

                all_ious = [self.compute_iou(all_boxes[i], all_boxes[j]) for j in range(len(preds)) if not i==j]
                if len(all_ious) == 0:
                    cond_max_iou = 0
                else:
                    cond_max_iou = max(all_ious)

                if score < self.image_thre:
                    continue

                center, scale = self._box2cs(all_boxes[i])
                joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
                joints_3d_vis = np.ones(
                    (self.num_joints, 3), dtype=np.float)
                kpt_db.append({
                    'image': img_name,
                    'center': center,
                    'scale': scale,
                    'score': score,
                    'joints_3d': joints_3d,
                    'joints_3d_vis': joints_3d_vis,
                    'cond_joints': all_cond_joints[i],
                    'cond_joints_vis': all_cond_joints_vis[i],
                    'cond_max_iou': cond_max_iou,
                })

        return kpt_db


    def _load_coco_person_detection_results(self):
        all_boxes = None

        with open(self.test_gt_file, 'r') as f:
            test_gt = json.load(f)

        with open(self.bbox_file, 'rb') as f:
            results = pickle.load(f)

        if not results:
            logger.error('=> Load %s fail!' % self.bbox_file)
            return None

        len_all_boxes=0
        for img in results:
            len_all_boxes += len(img[0])
        logger.info('=> Total boxes: {}'.format(len_all_boxes))

        kpt_db = []
        num_boxes = 0
        for n_img in range(0, len(results)):
            
            det_results = results[n_img][0]
            for det_res in det_results:
                img_name =  os.path.join(self.img_dir, test_gt['images'][n_img]['file_name'])
                img_id = test_gt['images'][n_img]['id']
                _box = det_res[:4]
                w = _box[2] - _box[0]
                h = _box[3] - _box[1]
                box = (_box[0], _box[1], w, h)
                score = det_res[4]

                if score < self.image_thre:
                    continue

                num_boxes = num_boxes + 1

                center, scale = self._box2cs(box)
                joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
                joints_3d_vis = np.ones(
                    (self.num_joints, 3), dtype=np.float)
                kpt_db.append({
                    'image': img_name,
                    'center': center,
                    'scale': scale,
                    'score': score,
                    'joints_3d': joints_3d,
                    'joints_3d_vis': joints_3d_vis,
                    'bbox': box,
                    'image_id': img_id,
                })

        logger.info('=> Total boxes after filter low score@{}: {}'.format(
            self.image_thre, num_boxes))
        return kpt_db

    
    # load detections by using pose predicitions from BU/TD/CTD model (during test time)
    def _load_coco_pose_results(self):
        all_preds = None
        with open(self.bbox_file, 'r') as f:
            all_preds = json.load(f)

        with open(self.test_gt_file, 'r') as f:
            test_gt = json.load(f)
            
        if not all_preds:
            logger.error('=> Load %s fail!' % self.bbox_file)
            return None

        kpt_db = []
        for img_pred in all_preds:

            score = img_pred['score']
            file_name = [img['file_name'] for img in test_gt['images'] if img['id']==img_pred['image_id']][0]
            img_name =  os.path.join(self.img_dir, file_name)
            data_numpy = cv2.imread(
                img_name, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.ones(
                (self.num_joints, 3), dtype=np.float)

            cond_joints = np.array(img_pred['keypoints']).reshape((self.num_joints, 3))
            cond_joints_vis = np.ones(
                (self.num_joints, 3), dtype=np.float)

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

            kpt_db.append({
                'image': img_name,
                'center': c,
                'scale': s,
                'score': score,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'cond_joints': cond_joints,
                'cond_joints_vis': cond_joints_vis,
                'bbox': bbox,
                'cond_max_iou': 1,
                })

        return kpt_db

    def evaluate(self, cfg, preds, output_dir, all_boxes, img_path, epoch=-1,
                 *args, **kwargs):

        res_folder = os.path.join(output_dir, 'results')
        if not os.path.exists(res_folder):
            try:
                os.makedirs(res_folder)
            except Exception:
                logger.error('Fail to make {}'.format(res_folder))

        res_file = os.path.join(
            res_folder, 'keypoints_{}_results_epoch{}.json'.format(
                self.mode, epoch)
        )
        if cfg.OUTPUT_JSON:
            res_file = cfg.OUTPUT_JSON
        image_path_to_image_id = {}

        for index in self.image_set_index:
            im_ann = self.coco.loadImgs(index)[0]
            img_path_key = os.path.join(self.image_dir, im_ann['file_name'])
            image_path_to_image_id[img_path_key] = im_ann['id']

        areas = {}
        for ann in list(self.coco.anns.values()):
            areas[ann['id']] = ann['area']
                
        # person x (keypoints)
        _kpts = []
        for idx, kpt in enumerate(preds):
            if not self.is_train:
                if not self.use_gt_bbox or self.use_bu_bbox_test:
                    area = all_boxes[idx][4]
                else:
                    area = areas[int(all_boxes[idx][6])]
            else:
                area = areas[int(all_boxes[idx][6])]
            _kpts.append({
                'keypoints': kpt,
                'center': all_boxes[idx][0:2],
                'scale': all_boxes[idx][2:4],
                'area': area,
                'score': all_boxes[idx][5],
                'image': image_path_to_image_id[img_path[idx]],
                'image_path': img_path[idx],
                'annotation_id': int(all_boxes[idx][6]),
            })
        # image x person x (keypoints)
        kpts = defaultdict(list)
        for kpt in _kpts:
            kpts[kpt['image']].append(kpt)

        # rescoring and oks nms
        num_joints = self.num_joints
        in_vis_thre = self.in_vis_thre
        oks_thre = self.oks_thre
        oks_nmsed_kpts = []

        for img in kpts.keys():
            img_kpts = kpts[img]
            for n_p in img_kpts:
                box_score = n_p['score']
                kpt_score = 0
                valid_num = 0
                for n_jt in range(0, num_joints):
                    t_s = n_p['keypoints'][n_jt][2]
                    if t_s > in_vis_thre:
                        kpt_score = kpt_score + t_s
                        valid_num = valid_num + 1
                if valid_num != 0:
                    kpt_score = kpt_score / valid_num
                # rescoring
                n_p['score'] = kpt_score * box_score
                n_p['box_score'] = box_score
                n_p['keypoint_score'] = kpt_score

            if self.soft_nms:
                keep = soft_oks_nms(
                    [img_kpts[i] for i in range(len(img_kpts))],
                    oks_thre,
                    self.joints_weight/10
                )
            else:
                keep = oks_nms(
                    [img_kpts[i] for i in range(len(img_kpts))],
                    oks_thre,
                    self.joints_weight/10
                )

            if self.use_bu_bbox_test or self.use_bu_bbox_train or self.use_gt_bbox:
                keep = []
            if not self.is_train and '.json' in cfg.TEST.COCO_BBOX_FILE:
                keep = []
            if len(keep) == 0:
                oks_nmsed_kpts.append(img_kpts)
            else:
                oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])
        len_oks_kps = 0
        for temp in oks_nmsed_kpts:
            len_oks_kps += len(temp) 
        self._write_coco_keypoint_results(
            oks_nmsed_kpts, res_file)
        
        if not self.is_train:
            info_str = self._do_python_keypoint_eval(
                res_file)
            name_value = OrderedDict(info_str)
            return name_value, name_value['AP']

        else:
            return {'Null': 0}, 0
            

    # --------------------------------------------------------------------
    def _write_coco_keypoint_results(self, keypoints, res_file):
        data_pack = [
            {
                'cat_id': self._class_to_coco_ind[cls],
                'cls_ind': cls_ind,
                'cls': cls,
                'ann_type': 'keypoints',
                'keypoints': keypoints
            }
            for cls_ind, cls in enumerate(self.classes) if not cls == '__background__'
        ]

        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])
        logger.info('=> writing results json to %s' % res_file)
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)
        try:
            json.load(open(res_file))
        except Exception:
            content = []
            with open(res_file, 'r') as f:
                for line in f:
                    content.append(line)
            content[-1] = ']'
            with open(res_file, 'w') as f:
                for c in content:
                    f.write(c)

    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        cat_id = data_pack['cat_id']
        keypoints = data_pack['keypoints']
        cat_results = []

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            _key_points = np.array([img_kpts[k]['keypoints']
                                    for k in range(len(img_kpts))])
            key_points = np.zeros(
                (_key_points.shape[0], self.num_joints * 3), dtype=np.float
            )

            for ipt in range(self.num_joints):
                key_points[:, ipt * 3 + 0] = _key_points[:, ipt, 0]
                key_points[:, ipt * 3 + 1] = _key_points[:, ipt, 1]
                key_points[:, ipt * 3 + 2] = _key_points[:, ipt, 2]  # keypoints score.

            result = [
                {
                    'image_id': img_kpts[k]['image'],
                    'image_path': os.path.join(*img_kpts[k]['image_path'].split('/')[-3:]),
                    'category_id': cat_id,
                    'keypoints': list(key_points[k]),
                    'score': img_kpts[k]['score'],
                    'center': list(img_kpts[k]['center']),
                    'scale': list(img_kpts[k]['scale']),
                    'annotation_id': img_kpts[k]['annotation_id'],
                    'box_score': img_kpts[k]['box_score'],
                    'keypoint_score': img_kpts[k]['keypoint_score'],                    
                }
                for k in range(len(img_kpts))
            ]
            cat_results.extend(result)

        return cat_results

    def _do_python_keypoint_eval(self, res_file, eval_inds=None):
        coco_dt = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_dt, 'keypoints')
        coco_eval.params.useSegm = None
        if eval_inds:
            coco_eval.params.imgIds = eval_inds
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats_names = ['AP', 'AP .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']

        info_str = []
        for ind, name in enumerate(stats_names):
            info_str.append((name, coco_eval.stats[ind]))

        return info_str
