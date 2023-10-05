import json
import numpy as np

import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
    
def calc_bboxes_from_keypoints(data, slack=0, offset=0, clip=True):
    data = np.asarray(data)
    if data.ndim != 3:
        data = np.expand_dims(data, axis=0)
    bboxes = np.full((data.shape[0], 4), np.nan)
    bboxes[:, :2] = np.nanmin(data[..., :2], axis=1) - slack  # X1, Y1
    bboxes[:, 2:4] = np.nanmax(data[..., :2], axis=1) + slack  # X2, Y2
    bboxes[:, [0, 2]] += offset
    if clip:
        coord = bboxes[:, :4]
        coord[coord < 0] = 0
    return bboxes


def _get_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou



if __name__ == '__main__':

    cond_json = 'your_data_folder/annotations/train.json'
    cid_cond_json = 'your_data_folder/annotations/train_cond.json'
    
    with open(cond_json, 'rb') as f:
        gt_annotations = json.load(f)

    for epoch in range(40, 139, 1):
        model = f'cid_{epoch}'
        cid_results_json = f'your_BU_folder/results_train/keypoints_train_results_{epoch}.json'

        try:
            # Load your predicted results and ground truth annotations
            with open(cid_results_json, 'r') as f:
                pred_results = json.load(f)
        except:
            continue

        # get num kpts
        num_joints = int(len(pred_results[0]['keypoints']) / 3)

        # Get the matching between predictions and ground truth
        # Add the matched keypoints to the ground truth annotations

        for ann in gt_annotations['annotations']:

            gt_kpts = [kpt for kpt in np.array(ann['keypoints']).reshape(-1, 3)[:, :2].tolist() if all(kpt) != 0]
            gt_bbox = calc_bboxes_from_keypoints(gt_kpts)[0]

            image_preds = [np.array(pred['keypoints']).reshape(-1, 3)[:, :2] for pred in pred_results if pred['image_id'] == ann['image_id'] and pred['category_id'] == ann['category_id']]
            
            image_pred_bboxes = calc_bboxes_from_keypoints(image_preds)
            
            iou_list = [_get_iou(gt_bbox, pred_bbox) for pred_bbox in image_pred_bboxes]

            matched_pred = image_preds[np.argmax(iou_list)]            

            _matched_pred = []
            for i , pred in enumerate(matched_pred):
                v = np.array(ann['keypoints']).reshape(-1, 3)[i][2]
                if v == 0:
                    pred = (0,0)
                _matched_pred.extend([pred[0], pred[1], v])
            matched_pred = _matched_pred

            # print("matched_preds", matched_pred)
            # print()
                
            ann['cond_kpts'].update({model:matched_pred})

        # Save the updated ground truth annotations
        with open(cid_cond_json, 'w') as f:
            json.dump(gt_annotations, f)
        print(f"saved to: {cid_cond_json}")
