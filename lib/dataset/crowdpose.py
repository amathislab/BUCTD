# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bowen Cheng (bcheng9@illinois.edu) and Bin Xiao (leoxiaobin@gmail.com)
# Modified to Conditional Top Down by Mu Zhou, Lucas Stoffl et al. (ICCV 2023)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
from dataset.dataloader import DataLoader

from crowdposetools.coco import COCO
from crowdposetools.cocoeval import COCOeval
import os

from collections import defaultdict
from collections import OrderedDict

# -------------------------------------------
# crowdpose_sigmas = np.array([.79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89, .25, .25]) / 10.0

# -------------------------------------------

logger = logging.getLogger(__name__)


class CrowdPoseDataset(DataLoader):
    """`CrowdPose`_ Dataset.

    Args:
        root (string): Root directory where dataset is located to.
        dataset (string): Dataset name(train2017, val2017, test2017).
        data_format(string): Data format for reading('jpg', 'zip')
        transform (callable, optional): A function/transform that  takes in an opencv image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, cfg, image_dir, annotation_file, is_train, transform=None):
        super().__init__(cfg, image_dir, annotation_file, is_train, transform)

        self.num_joints = 14
        self.flip_pairs = [[0, 1], [2, 3], [4, 5], [6, 7],
                           [8, 9], [10, 11]]
        self.parent_ids = None
        self.upper_body_ids = (0, 1, 2, 3, 4, 5, 12, 13)
        self.lower_body_ids = (6, 7, 8, 9, 10, 11)
        self.crowdpose_sigma = np.array([.79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89, .25, .25]) / 10.0
        self.coco = COCO(self._get_ann_file_keypoint())
        self.kpt_colors = [[245, 53, 53], [245, 125, 45], [253, 206, 20], [206, 244, 54], [118, 253, 27], [47, 254, 47], [25, 245, 113], [15, 243, 197], [14, 199, 245], [44, 126, 249], [13, 13, 249], [128, 47, 249], [205, 38, 247], [245, 48, 206]]

        self.joints_weight = np.array(
            [
                1., 1., 1.2, 1.2,
                1.5, 1.5, 1., 1., 
                1.2, 1.2, 1.5, 1.5,
                1., 1.
            ],
            dtype=np.float32
        ).reshape((self.num_joints, 1))


        self.db = self._get_db()

        if is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)

        logger.info('=> load {} samples'.format(len(self.db)))

    def _get_db(self):
        
        if self.is_train:
            ## during training
            if self.use_bu_bbox_train:
                print("load bu derived bbox for train set")
                gt_db = self._load_coco_keypoint_annotations(bu_bbox=True, best_model_key='')
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
                    gt_db = self._load_coco_keypoint_annotations(bu_bbox=True, best_model_key='')
                else:
                    # use bbox from bu model
                    print("load BU bbox from result json")
                    gt_db = self._load_coco_person_BU_detection_results()
            
            else:
                ## load detector bbox (no conditions)
                print("load bbox from detector")
                gt_db = self._load_coco_person_detection_results()

        return gt_db

    def processKeypoints(self, keypoints):
        tmp = keypoints.copy()
        if keypoints[:, 2].max() > 0:
            p = keypoints[keypoints[:, 2] > 0][:, :2].mean(axis=0)
            num_keypoints = keypoints.shape[0]
            for i in range(num_keypoints):
                tmp[i][0:3] = [
                    float(keypoints[i][0]),
                    float(keypoints[i][1]),
                    float(keypoints[i][2])
                ]

        return tmp

    def evaluate(self, cfg, preds, output_dir, all_boxes, img_path, epoch=-1,
                 *args, **kwargs):
        '''
        Perform evaluation on CrowdPose keypoint task
        :param cfg: cfg dictionary
        :param preds: prediction
        :param output_dir: output directory
        :param args: 
        :param kwargs: 
        :return: 
        '''
        if all_boxes.shape[1] == 8:
            return self.evaluate_lambda(cfg, preds, output_dir, all_boxes, img_path, epoch, *args, **kwargs)

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

        # preds is a numpy array: person x (keypoints): N x 14 x 3
        # person x (keypoints)
        _kpts = []
        for idx, kpt in enumerate(preds):
            area = (np.max(kpt[:, 0]) - np.min(kpt[:, 0])) * (np.max(kpt[:, 1]) - np.min(kpt[:, 1]))
            kpt = self.processKeypoints(kpt)

            _kpts.append({
                'keypoints': kpt[:, 0:3],
                'center': all_boxes[idx][0:2],
                'scale': all_boxes[idx][2:4],
                'area': all_boxes[idx][4],
                'score': all_boxes[idx][5],
                'image': image_path_to_image_id[img_path[idx]],
                'annotation_id': int(all_boxes[idx][6]),
                'image_path': img_path[idx],
                
            })

        # keypoints: num_joints * 4 (x, y, score, tag)
        # image x person x (keypoints)
        kpts = defaultdict(list)
        for kpt in _kpts:
            kpts[kpt['image']].append(kpt)

        # rescoring and oks nms
        num_joints = self.num_joints
        in_vis_thre = self.in_vis_thre
        oks_thre = self.oks_thre
        oks_nmsed_kpts = []

        # image x person x (keypoints)
        for img in kpts.keys():
            # person x (keypoints)
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
                
            # person x (keypoints)
            if self.use_bu_bbox_test or self.use_bu_bbox_train or self.use_gt_bbox:
                keep = []
            if not self.is_train and '.json' in cfg.TEST.COCO_BBOX_FILE:
                keep = []
            if len(keep) == 0:
                oks_nmsed_kpts.append(img_kpts)
            else:
                oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])

        self._write_coco_keypoint_results(
            oks_nmsed_kpts, res_file
        )

        # CrowdPose `test` set has annotation.
        info_str = self._do_python_keypoint_eval(
            res_file, res_folder
        )
        name_value = OrderedDict(info_str)
        return name_value, name_value['AP']


    def _do_python_keypoint_eval(self, res_file, res_folder):
        coco_dt = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_dt, 'keypoints')
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats_names = ['AP', 'Ap .5', 'AP .75', 'AR', 'AR .5', 'AR .75', 'AP (easy)', 'AP (medium)', 'AP (hard)']
        stats_index = [0, 1, 2, 5, 6, 7, 8, 9, 10]

        info_str = []
        for ind, name in enumerate(stats_names):
            info_str.append((name, coco_eval.stats[stats_index[ind]]))

        return info_str