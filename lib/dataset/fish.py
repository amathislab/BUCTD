# ------------------------------------------------------------------------------
# Written by Lucas Stoffl (lucas.stoffl@epfl.ch), Mu Zhou
# ------------------------------------------------------------------------------

import numpy as np
import logging

from dataset.dataloader import DataLoader
logger = logging.getLogger(__name__)
from pycocotools.cocoeval import COCOeval

class FishDataset(DataLoader):
    '''
    "keypoints": {
        '0': 'tip',
        '1': 'gill',
        '2': 'peduncle',
        '3': 'caudaltip',
        '4': 'dfin1',
        '5': 'dfin2',
        '6': 'dfintip',
    }
    "skeleton": []
    '''
    def __init__(self, cfg, image_dir, annotation_file, is_train, transform=None):
        super().__init__(cfg, image_dir, annotation_file, is_train, transform)

        self.num_joints = 7
        self.flip_pairs = []
        self.parent_ids = None
        self.upper_body_ids = (0, 1)
        self.lower_body_ids = (2, 3, 4, 5, 6)
        self.bbox_file = cfg.TEST.COCO_BBOX_FILE
        self.condition_topdown = cfg.MODEL.CONDITIONAL_TOPDOWN

        self.joints_weight = np.array(
            [
                1., 1., 1., 1., 1., 1., 1.,
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
                gt_db = self._load_coco_keypoint_annotations(bu_bbox=True, best_model_key='baseline_resnet_50_s4_60000')
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

    def _do_python_keypoint_eval(self, res_file, eval_inds=None):
        coco_dt = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_dt, 'keypoints')
        coco_eval.params.useSegm = None
        if eval_inds:
            coco_eval.params.imgIds = eval_inds
        coco_eval.params.kpt_oks_sigmas = np.array(self.num_joints*[0.1])
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats_names = ['AP', 'AP .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']

        info_str = []
        for ind, name in enumerate(stats_names):
            info_str.append((name, coco_eval.stats[ind]))

        return info_str
