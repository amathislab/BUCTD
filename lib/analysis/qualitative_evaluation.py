import os
import numpy as np
import sys
import copy
import pickle
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from core.function import _print_name_value
from collections import OrderedDict
import logging
import matplotlib.pyplot as plt

import utilities
from itertools import cycle
from jinja2 import Template
import json
import cv2
from tqdm import tqdm

from utils.utils import vis_keypoints
from utils.utils import vis_segmentation
from utils.utils import vis_bbs


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# ------------------------------------------------------------------
## idx of the groups, rows and columns
def binwise_coco_evaluation(gt_file, dt_file, output_dir, images_dir, num_keypoints_group, num_overlaps_group):
	coco_gt = COCO(gt_file)
	coco_dt = coco_gt.loadRes(dt_file)

	coco_gt_json = json.load(open(gt_file))
	coco_dt_json = json.load(open(dt_file))


	###--- now only consider thos instances belong to a particular bin for evaluation
	overlap_groups  = [[0],[1,2],[3,4,5,6,7,8]]
	num_kpt_groups  = [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17]]

	# -------------------------------------------
	image_ids = coco_gt.getImgIds()
	all_annotation_ids = [] ## 6352 person annotations
	all_image_ids = [] ## total 2346 person images

	# -------------------------------------------
	for image_id in image_ids:
		valid_image_ann_ids, valid_image_ids = get_valid_bin_annotations(coco_gt, image_id, NUM_OVERLAPS=overlap_groups[num_overlaps_group], NUM_KEYPOINTS=num_kpt_groups[num_keypoints_group], IOU_FOR_OVERLAP=0.1)
		all_annotation_ids += valid_image_ann_ids
		all_image_ids += valid_image_ids

	# -------------------------------------------
	annotations = coco_gt.loadAnns(all_annotation_ids)
	
	pbar = tqdm(total=len(annotations))

	# -------------------------------------------
	coco_eval = COCOeval(coco_gt, coco_dt, iouType='keypoints')
	coco_eval.params.useSegm = None
	coco_eval.evaluate()
	coco_eval.accumulate()
	coco_eval.summarize()

	# -------------------------------------------
	matched_annotation_dict = {} ## gt_annotation_id: dt_annotation_id

	# -------------------------------------------	
	for eval_image in coco_eval.evalImgs:
		if eval_image is not None:
			gt_ids = eval_image['gtIds']
			dt_ids = eval_image['dtIds']
			matches = eval_image['gtMatches'][0] ## for iou 0.5, T x G
			for gt_id, dt_id in zip(gt_ids, matches): 
				matched_annotation_dict[gt_id] = int(dt_id)

	# -------------------------------------------
	for idx_gt, annotation_gt in enumerate(annotations):
		matched_annotation_dt_id = matched_annotation_dict[annotation_gt['id']]
		if matched_annotation_dt_id == 0:
			matched_annotation_dt = None
		else:
			matched_annotation_dt = coco_dt.loadAnns([matched_annotation_dt_id])[0]
			assert(annotation_gt['image_id'] == matched_annotation_dt['image_id'])

		save_detections(coco_gt, coco_dt, annotation_gt, matched_annotation_dt, images_dir, output_dir)
	
		pbar.update(1)

	pbar.close()
	# -------------------------------------------

	return 


# ------------------------------------------------------------------
def coco_evaluation(gt_file, dt_file, output_dir, images_dir, num_images=None):
	coco_gt = COCO(gt_file)
	coco_dt = coco_gt.loadRes(dt_file)

	coco_gt_json = json.load(open(gt_file))
	coco_dt_json = json.load(open(dt_file))

	# -------------------------------------------
	image_ids = coco_gt.getImgIds()


	all_annotation_ids = [] ## 6352 person annotations
	all_image_ids = [] ## total 2346 person images

	# -------------------------------------------
	for image_id in image_ids:
		valid_image_ann_ids, valid_image_ids = get_valid_annotations(coco_gt, image_id)
		all_annotation_ids += valid_image_ann_ids
		all_image_ids += valid_image_ids

	# -------------------------------------------
	## note for hard val, the annotations are already sorted by map of baseline hrnet.
	if 'hard_val_annotation_ids' in coco_gt_json.keys():
		all_annotation_ids = coco_gt_json['hard_val_annotation_ids']
		annotations = coco_gt.loadAnns(all_annotation_ids)
		for annotation in annotations:
			annotation['baseline_map'] = coco_gt_json['hard_val_baseline_maps'][str(annotation['id'])]
	else:
		annotations = coco_gt.loadAnns(all_annotation_ids)

	if num_images is not None:
		annotations = annotations[:num_images]
		
	pbar = tqdm(total=len(annotations))

	# -------------------------------------------
	coco_eval = COCOeval(coco_gt, coco_dt, iouType='keypoints')
	coco_eval.params.useSegm = None
	coco_eval.evaluate()
	coco_eval.accumulate()
	coco_eval.summarize()

	# -------------------------------------------
	matched_annotation_dict = {} ## gt_annotation_id: dt_annotation_id

	# -------------------------------------------	
	for eval_image in coco_eval.evalImgs:
		if eval_image is not None:
			gt_ids = eval_image['gtIds']
			dt_ids = eval_image['dtIds']
			matches = eval_image['gtMatches'][0] ## for iou 0.5, T x G
			for gt_id, dt_id in zip(gt_ids, matches): 
				matched_annotation_dict[gt_id] = int(dt_id)

	# -------------------------------------------
	for idx_gt, annotation_gt in enumerate(annotations):
		matched_annotation_dt_id = matched_annotation_dict[annotation_gt['id']]
		if matched_annotation_dt_id == 0:
			matched_annotation_dt = None
		else:
			matched_annotation_dt = coco_dt.loadAnns([matched_annotation_dt_id])[0]
			assert(annotation_gt['image_id'] == matched_annotation_dt['image_id'])

		save_detections(coco_gt, coco_dt, annotation_gt, matched_annotation_dt, images_dir, output_dir)

		# # if annotation_gt['id'] == 2675:
		# 	# save_detections(coco_gt, coco_dt, annotation_gt, matched_annotation_dt, images_dir, output_dir)

		# if annotation_gt['id'] == 559508:
		# 	save_detections(coco_gt, coco_dt, annotation_gt, matched_annotation_dt, images_dir, output_dir)

		# if annotation_gt['id'] == 2933:
		# 	import pdb; pdb.set_trace()
		# 	save_detections(coco_gt, coco_dt, annotation_gt, matched_annotation_dt, images_dir, output_dir)


		pbar.update(1)

	pbar.close()
	# -------------------------------------------

	return 

# ------------------------------------------------------------------
def save_detections(coco_gt, coco_dt, annotation_gt, annotation_dt, images_dir, output_dir, target_bbox_width=192*2, target_bbox_height=256*2, vis_score_thres=0.3):
	image_info = coco_gt.loadImgs(annotation_gt['image_id'])[0]
	image_path = os.path.join(images_dir, image_info['file_name'])
	image = cv2.imread(image_path)

	gt_keypoints = np.array(annotation_gt['keypoints'])
	gt_kps = np.zeros((3, 17)).astype(np.int16)
	gt_kps[0, :] = gt_keypoints[0::3]; gt_kps[1, :] = gt_keypoints[1::3]
	gt_kps[2, :] = gt_keypoints[2::3]

	if annotation_dt is None:
		dt_keypoints = np.array([0]*17*3)
	else:
		dt_keypoints = np.array(annotation_dt['keypoints'])
	dt_kps = np.zeros((3, 17)).astype(np.int16)
	dt_kps[0, :] = dt_keypoints[0::3]; dt_kps[1, :] = dt_keypoints[1::3]
	dt_kps[2, :] = gt_keypoints[2::3]

	image_height, image_width, image_channel = image.shape
	target_image_height = target_bbox_height
	target_image_width = round((image_width/image_height)*target_image_height)
	vis_gt_image = vis_keypoints(img=image.copy(), kps=gt_kps.copy(), kp_thresh=-1, alpha=0.7)	
	vis_gt_image = vis_segmentation(img=vis_gt_image.copy(), mask=coco_gt.annToMask(annotation_gt))

	##-------------- draw all gt bbs ----------------
	image_gt_annotation_ids, image_gt_ids = get_valid_annotations(coco_gt, annotation_gt['image_id'])
	image_gt_annotations = coco_gt.loadAnns(image_gt_annotation_ids)

	vis_all_gt_image = image.copy()

	for image_gt_annotation in image_gt_annotations:
		segmentation = coco_gt.annToMask(image_gt_annotation)
		# vis_all_gt_image = vis_segmentation(img=vis_all_gt_image.copy(), mask=segmentation) ## tunrn off
		vis_all_gt_image = draw_keypoints(vis_all_gt_image, image_gt_annotation, is_gt=True)

	##-------------- draw all dt bbs ----------------
	image_dt_annotation_ids, image_dt_ids = get_valid_annotations(coco_dt, annotation_gt['image_id'])
	image_dt_annotations = coco_dt.loadAnns(image_dt_annotation_ids)

	##------chop off low score detections------------
	valid_image_dt_annotations = [ann for ann in image_dt_annotations if ann['score'] >= vis_score_thres]
	image_dt_annotations = valid_image_dt_annotations

	##---------------------------------------------

	vis_all_gt_bb_image = image.copy()

	for image_gt_annotation in image_gt_annotations:
		vis_all_gt_bb_image = vis_bbs(img=vis_all_gt_bb_image.copy(), bbox=image_gt_annotation['clean_bbox'])

	##---------------------------------------------

	vis_all_dt_image = image.copy()
	vis_all_bb_image = image.copy()


	# image_dt_annotations = [image_dt_annotations[0], image_dt_annotations[1], image_dt_annotations[2]]
	# image_dt_annotations = [image_dt_annotations[0], image_dt_annotations[1]]
	# print(len(image_dt_annotations))
	# image_dt_annotations = [image_dt_annotations[0], image_dt_annotations[1]]
	# image_dt_annotations = [image_dt_annotations[0], image_dt_annotations[2]]

	# image_dt_annotations = image_dt_annotations[1:]
	# image_dt_annotations = image_dt_annotations[:1]
	# image_dt_annotations = []

	# # # ------------------------------------------
	# # image_dt_annotations = [image_dt_annotations[0]]
	# joint_idx = 1
	# image_dt_annotations[1]['keypoints'][joint_idx*3 + 2] = 0.0

	# joint_idx = 2
	# image_dt_annotations[1]['keypoints'][joint_idx*3 + 2] = 0.0

	# joint_idx = 3
	# image_dt_annotations[1]['keypoints'][joint_idx*3 + 2] = 0.0

	# joint_idx = 4
	# image_dt_annotations[1]['keypoints'][joint_idx*3 + 2] = 0.0

	# joint_idx = 7
	# image_dt_annotations[1]['keypoints'][joint_idx*3 + 2] = 0.0

	# joint_idx = 10
	# image_dt_annotations[1]['keypoints'][joint_idx*3 + 2] = 0.0

	# joint_idx = 7
	# image_dt_annotations[1]['keypoints'][joint_idx*3 + 2] = 0.0

	# ------------------------------------------

	for image_dt_annotation in image_dt_annotations:
		vis_all_dt_image = draw_keypoints(vis_all_dt_image.copy(), image_dt_annotation, is_gt=False, kp_thresh=0.2, alpha=0.9)
		score_dict = {'score':image_dt_annotation['score'], 'box_score':image_dt_annotation['box_score'], 'keypoint_score':image_dt_annotation['keypoint_score'],}
		vis_all_bb_image = vis_bbs(img=vis_all_bb_image.copy(), bbox=image_dt_annotation['clean_bbox'], score_dict=score_dict)

	##-----------------------------------------------------
	vis_gt_image_resized = cv2.resize(vis_gt_image, (target_image_width, target_image_height), interpolation=cv2.INTER_AREA)
	vis_all_gt_image_resized = cv2.resize(vis_all_gt_image, (target_image_width, target_image_height), interpolation=cv2.INTER_AREA)
	vis_all_dt_image_resized = cv2.resize(vis_all_dt_image, (target_image_width, target_image_height), interpolation=cv2.INTER_AREA)
	vis_all_bb_image_resized = cv2.resize(vis_all_bb_image, (target_image_width, target_image_height), interpolation=cv2.INTER_AREA)

	##-----------------------------------------------------
	bbox = annotation_gt['clean_bbox'] ## x, y, w, h
	bbox_x = int(bbox[0]); bbox_y = int(bbox[1]); bbox_w = int(bbox[2]); bbox_h = int(bbox[3]);
	bbox_image = image.copy()[bbox_y:bbox_y+bbox_h, bbox_x:bbox_x+bbox_w, :]
	bbox_image_resized = cv2.resize(bbox_image, (target_bbox_width, target_bbox_height), interpolation=cv2.INTER_AREA)

	bbox_gt_kps = gt_kps.copy()
	bbox_gt_kps[0, :] = ((bbox_gt_kps[0, :] - bbox_x)/bbox_w)*target_bbox_width
	bbox_gt_kps[1, :] = ((bbox_gt_kps[1, :] - bbox_y)/bbox_h)*target_bbox_height
	vis_gt_bbox = vis_keypoints(img=bbox_image_resized.copy(), kps=bbox_gt_kps.copy(), kp_thresh=-1, alpha=0.7)

	bbox_dt_kps = dt_kps.copy()
	bbox_dt_kps[0, :] = ((bbox_dt_kps[0, :] - bbox_x)/bbox_w)*target_bbox_width
	bbox_dt_kps[1, :] = ((bbox_dt_kps[1, :] - bbox_y)/bbox_h)*target_bbox_height
	vis_dt_bbox = vis_keypoints(img=bbox_image_resized.copy(), kps=bbox_dt_kps.copy(), kp_thresh=-1, alpha=0.7)

	file_name = image_info['file_name'].split('/')[-1]
	if 'baseline_map' in annotation_gt.keys():
		save_image_path = os.path.join(output_dir, 'baseline_map:{}_img:{}_ann:{:09d}.jpg'.format(round(annotation_gt['baseline_map'], 2), file_name.replace('.jpg', ''), annotation_gt['id']))
	else:
		save_image_path = os.path.join(output_dir, '{}_{:09d}.jpg'.format(file_name.replace('.jpg', ''), annotation_gt['id']))

	vis_image = np.concatenate((vis_gt_image_resized, vis_gt_bbox, vis_dt_bbox, vis_all_gt_image_resized, vis_all_dt_image_resized, vis_all_bb_image_resized), axis=1)
	
	cv2.imwrite(save_image_path, vis_image)

	save_image_path = os.path.join(output_dir, 'gt_bb_{}_{:09d}.jpg'.format(file_name.replace('.jpg', ''), annotation_gt['id']))
	cv2.imwrite(save_image_path, vis_all_gt_bb_image)

	save_image_path = os.path.join(output_dir, 'dt_all_{}_{:09d}.jpg'.format(file_name.replace('.jpg', ''), annotation_gt['id']))
	cv2.imwrite(save_image_path, vis_all_dt_image)

	save_image_path = os.path.join(output_dir, 'cheat_all_{}_{:09d}.jpg'.format(file_name.replace('.jpg', ''), annotation_gt['id']))
	cv2.imwrite(save_image_path, vis_all_gt_image)


	return

# ------------------------------------------------------------------
def draw_keypoints(image, annotation, is_gt=True, kp_thresh=-1, alpha=0.7):
	keypoints = np.array(annotation['keypoints'])
	kps = np.zeros((3, 17))
	kps[0, :] = keypoints[0::3]; kps[1, :] = keypoints[1::3]
	if is_gt:
		kps[2, :] = keypoints[2::3] ## 0 flags are not drawn
	else:
		# kps[2, :] = kps[2, :] + 1 ## everything is drawn drawn
		kps[2, :] = keypoints[2::3] ## 0 flags are not drawn


	vis_gt_image = vis_keypoints(img=image.copy(), kps=kps.copy(), kp_thresh=kp_thresh, alpha=alpha)	

	return vis_gt_image

# ------------------------------------------------------------------
def get_valid_annotations(coco, image_id):
    annotation_ids = coco.getAnnIds(imgIds=image_id)
    annotations = coco.loadAnns(annotation_ids)

    # ----------------------
    # sanitize bboxes
    image_info = coco.loadImgs(image_id)[0]
    width = image_info['width']
    height = image_info['height']
    valid_objs = []
    for obj in annotations:
        # ignore objs without keypoints annotation
        if max(obj['keypoints']) == 0:
            continue
        x, y, w, h = obj['bbox']
        x1 = np.max((0, x))
        y1 = np.max((0, y))
        x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
        y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
        if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
            obj['clean_bbox'] = [x1, y1, x2-x1, y2-y1]
            valid_objs.append(obj)
    annotations = valid_objs
    # ----------------------

    ious = utilities.compute_ious(annotations)
    eye  = np.eye(len(annotations))

    # ------------------------------------------------
    valid_annotation_ids = []
    valid_image_ids = []

    for annotation_idx, annotation in enumerate(annotations):
        valid_annotation_ids.append(annotation['id'])

    if len(valid_annotation_ids) > 0:
    	valid_image_ids.append(image_id)

    return valid_annotation_ids, valid_image_ids


# ------------------------------------------------------------------
def get_valid_bin_annotations(coco, image_id, NUM_OVERLAPS, NUM_KEYPOINTS, IOU_FOR_OVERLAP=0.1):
    annotation_ids = coco.getAnnIds(imgIds=image_id)
    annotations = coco.loadAnns(annotation_ids)

    # ----------------------
    # sanitize bboxes
    image_info = coco.loadImgs(image_id)[0]
    width = image_info['width']
    height = image_info['height']
    valid_objs = []
    for obj in annotations:
        # ignore objs without keypoints annotation
        if max(obj['keypoints']) == 0:
            continue
        x, y, w, h = obj['bbox']
        x1 = np.max((0, x))
        y1 = np.max((0, y))
        x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
        y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
        if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
            obj['clean_bbox'] = [x1, y1, x2-x1, y2-y1]
            valid_objs.append(obj)
    annotations = valid_objs
    # ----------------------

    ious = utilities.compute_ious(annotations)
    eye  = np.eye(len(annotations))

    # ------------------------------------------------
    valid_annotation_ids = []
    valid_image_ids = []

    for annotation_idx, annotation in enumerate(annotations):
        if 'num_overlaps' in annotation.keys():
        	num_keypoints = int(annotation['num_keypoints'])
        	num_overlaps = int(annotation['num_overlaps'])
        else:
        	num_overlaps  = sum((ious[annotation_idx, :] - eye[annotation_idx, :]) > IOU_FOR_OVERLAP)
        	num_keypoints = annotation['num_keypoints'] ##num_keypoints = (kps_vis == 1) + (kps_vis == 2)

        if num_overlaps in NUM_OVERLAPS and num_keypoints in NUM_KEYPOINTS:
            valid_annotation_ids.append(annotation['id'])

    if len(valid_annotation_ids) > 0:
    	valid_image_ids.append(image_id)

    return valid_annotation_ids, valid_image_ids

# ------------------------------------------------------------------
def print_evaluation(coco_gt, coco_dt, print=True):
	coco_eval = COCOeval(coco_gt, coco_dt, iouType='keypoints')
	coco_eval.params.useSegm = None
	coco_eval.evaluate()
	coco_eval.accumulate()
	coco_eval.summarize()

	stats_names = ['AP', 'AP .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']


	info_str = []
	for ind, name in enumerate(stats_names):
		info_str.append((name, coco_eval.stats[ind]))

	# info_str.append(('AP .6', summarize_oks(coco_eval=coco_eval, iouThr=0.6)))

	if print:
		name_values = OrderedDict(info_str)
		_print_name_value(name_value=name_values, full_arch_name='pose_hrnet')

	return info_str

# -----------------------------------------------------------------
def summarize_oks(coco_eval, ap=1, iouThr=0.85, areaRng='all', maxDets=20):
    p = coco_eval.params
    iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
    titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
    typeStr = '(AP)' if ap==1 else '(AR)'
    iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
        if iouThr is None else '{:0.2f}'.format(iouThr)

    aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
    mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
    if ap == 1:
        # dimension of precision: [TxRxKxAxM]
        s = coco_eval.eval['precision']
        # IoU
        if iouThr is not None:
            t = np.where(iouThr == p.iouThrs)[0]
            s = s[t]
        s = s[:,:,:,aind,mind]
    else:
        # dimension of recall: [TxKxAxM]
        s = coco_eval.eval['recall']
        if iouThr is not None:
            t = np.where(iouThr == p.iouThrs)[0]
            s = s[t]
        s = s[:,:,aind,mind]
    if len(s[s>-1])==0:
        mean_s = -1
    else:
        mean_s = np.mean(s[s>-1])
    print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
    return mean_s


