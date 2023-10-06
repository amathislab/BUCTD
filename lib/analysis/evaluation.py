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


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# ------------------------------------------------------------------
def coco_evaluation(gt_file, dt_file, output_dir):
	coco_gt = COCO(gt_file)
	coco_dt = coco_gt.loadRes(dt_file)

	info_str = print_evaluation(coco_gt, coco_dt)

	###--- now only consider thos instances belong to a particular bin for evaluation
	overlap_groups  = [[0],[1,2],[3,4,5,6,7,8]]
	num_kpt_groups  = [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17]]

	og = overlap_groups[0]
	ng = num_kpt_groups[0]
	# -------------------------------------------

	# -------------------------------------------
	all_stats = {'num_instances': np.zeros((len(overlap_groups), len(num_kpt_groups)))}
	for perf_val in info_str:
		all_stats[perf_val[0]] = np.zeros((len(overlap_groups), len(num_kpt_groups)))

	# -------------------------------------------
	print('overlap_groups:{} num_keypoints:{}'.format(og, ng))
	for i, og in enumerate(overlap_groups):
	    for j, ng in enumerate(num_kpt_groups):
	        bin_info = bin_evaluate(coco_gt=coco_gt, coco_dt_file=dt_file, overlap_group=og, num_kpt_group=ng)

	        for stat_name in bin_info.keys():
	        	all_stats[stat_name][i, j] = bin_info[stat_name]

	# -------------------------------------------
	cmaps = [plt.cm.Greens, plt.cm.Blues, plt.cm.YlOrBr, plt.cm.RdPu, plt.cm.YlOrRd, plt.cm.Reds, plt.cm.PuRd, plt.cm.BuPu, plt.cm.PuBu]
	cmaps = cycle(cmaps)
	# -------------------------------------------
	for idx, stat_name in enumerate(all_stats.keys()):
		benchmark_mat = all_stats[stat_name]

		fig = plt.figure(figsize=(6,6))
		plt.clf()
		ax = fig.add_subplot(111)
		ax.set_aspect(1)
		res = ax.imshow(benchmark_mat, cmap=next(cmaps), interpolation='nearest')
		width, height = benchmark_mat.shape
		for x in range(width):
			for y in range(height):
				ax.annotate('{}'.format((benchmark_mat[x,y])), xy=(y, x),
				            horizontalalignment='center',
				            verticalalignment='center',fontsize=20)
		plt.xticks(range(height),['<=5','<=10','<=15','>15'])
		plt.yticks(range(width),['0','1/2','>=3'])
		plt.title("{}".format(stat_name),fontsize=20)
		plt.xlabel("Num. keypoints",fontsize=20)
		plt.ylabel("Num. overlapping instances",fontsize=20)
		path = '{}/benchmark_{}.pdf'.format(output_dir, stat_name)
		plt.savefig(path, bbox_inches='tight')
		plt.close()
	# -------------------------------------------

	return 

# ------------------------------------------------------------------
### need to overload getAnns and getImgIds
## 11004 all categories! 6352 for person category with keypoints
def bin_evaluate(coco_gt, coco_dt_file, overlap_group, num_kpt_group):
	bin_coco_gt = copy.deepcopy(coco_gt)
	bin_coco_dt = bin_coco_gt.loadRes(coco_dt_file)

	image_ids = coco_gt.getImgIds()

	# -------------------------------------------
	## nowtrim coco_gt.anns and coco_gt.imgs
	valid_coco_image_ids = []
	valid_coco_ann_ids = []

	for idx, image_id in enumerate(image_ids):
		valid_image_ann_ids, valid_image_ids = check_valid_annotations(coco_gt, image_id, NUM_OVERLAPS=overlap_group, NUM_KEYPOINTS=num_kpt_group)

		valid_coco_ann_ids.extend(valid_image_ann_ids)
		valid_coco_image_ids.extend(valid_image_ids)

	bin_coco_gt.dataset['annotations'] = [coco_gt.anns[id] for id in valid_coco_ann_ids]
	bin_coco_gt.imgs = {id: coco_gt.imgs[id] for id in valid_coco_image_ids}
	bin_coco_gt.createIndex() ## create index again

	# -------------------------------------------
	## now trim coco_dt as well
	bin_coco_dt_json = json.load(open(coco_dt_file))

	valid_dt_annotations = []
	for annotation in bin_coco_dt.dataset['annotations']:
		if annotation['annotation_id'] in valid_coco_ann_ids:
			valid_dt_annotations.append(annotation)

	bin_coco_dt.dataset['annotations'] = valid_dt_annotations
	bin_coco_dt.imgs = {id: bin_coco_dt.imgs[id] for id in valid_coco_image_ids}
	bin_coco_dt.createIndex() ## create index again

	# -------------------------------------------
	info_str = print_evaluation(bin_coco_gt, bin_coco_dt)

	bin_info = {
				'num_instances': len(valid_coco_ann_ids),
				 }
	for perf_val in info_str:
		bin_info[perf_val[0]] = round(perf_val[1], 3)

	return bin_info

# ------------------------------------------------------------------
def check_valid_annotations(coco, image_id, NUM_OVERLAPS, NUM_KEYPOINTS, IOU_FOR_OVERLAP=0.1):
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

# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
def sort_instance_ap(gt_file, dt_file, output_dir):
	coco_gt = COCO(gt_file)
	coco_dt = coco_gt.loadRes(dt_file)

	info_str = print_evaluation(coco_gt, coco_dt)
	
	# -------------------------------------------
	image_ids = coco_gt.getImgIds()
	all_annotation_ids = [] ## 6352 person annotations
	all_image_ids = [] ## total 2346 person images

	# -------------------------------------------
	ALL_OVERLAPS = list(range(30))
	ALL_NUM_KEYPOINTS = list(range(1, 18))

	for image_id in image_ids:
		valid_image_ann_ids, valid_image_ids = check_valid_annotations(coco_gt, image_id, ALL_OVERLAPS, ALL_NUM_KEYPOINTS, IOU_FOR_OVERLAP=0.1)
		all_annotation_ids += valid_image_ann_ids
		all_image_ids += valid_image_ids

	# -------------------------------------------
	performance_dict = {} ## {annotation_id: performance_string)

	for idx, annotation_id in enumerate(all_annotation_ids):		
		annotation = coco_gt.loadAnns([annotation_id])[0]
		instance_info = instance_evaluate(coco_gt, dt_file, instance_id=annotation_id, image_id=annotation['image_id'])
		performance_dict[annotation_id] = instance_info

		print('{}/{} Done evaluation for annotation id:{}, image_id:{}. AP:{}'.format(idx, len(all_annotation_ids), annotation_id, annotation['image_id'], round(instance_info['AP'], 3)))


	# -------------------------------------------
	with open(os.path.join(output_dir, 'instance_ap.json'), 'w') as f:
		json.dump(performance_dict, f,  indent=4)

	# -------------------------------------------

	
	return 

# ------------------------------------------------------------------
### need to overload getAnns and getImgIds
## 11004 all categories! 6352 for person category with keypoints
def instance_evaluate(coco_gt, coco_dt_file, instance_id, image_id):
	instance_coco_gt = copy.deepcopy(coco_gt)
	instance_coco_dt = instance_coco_gt.loadRes(coco_dt_file)

	image_ids = coco_gt.getImgIds()
	
	# -------------------------------------------
	instance_coco_gt.dataset['annotations'] = [coco_gt.anns[instance_id]]
	instance_coco_gt.imgs = {image_id: coco_gt.imgs[image_id]}
	instance_coco_gt.createIndex() ## create index again

	# -------------------------------------------
	## now trim coco_dt as well
	instance_coco_dt_json = json.load(open(coco_dt_file))

	valid_dt_annotations = []
	for annotation in instance_coco_dt.dataset['annotations']:
		if annotation['annotation_id'] == instance_id:
			valid_dt_annotations.append(annotation)

	instance_coco_dt.dataset['annotations'] = valid_dt_annotations
	instance_coco_dt.imgs = {image_id: instance_coco_dt.imgs[image_id]}
	instance_coco_dt.createIndex() ## create index again

	# -------------------------------------------
	info_str = print_evaluation(instance_coco_gt, instance_coco_dt, print=False)

	instance_info = {}
	for perf_val in info_str:
		instance_info[perf_val[0]] = perf_val[1]

	return instance_info































