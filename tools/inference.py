# ------------------------------------------------------------------------------
# Written by Mu Zhou
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import cv2

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import models
from core.inference import get_final_preds
from utils.transforms import get_affine_transform, affine_transform
from config import cfg, update_config

import numpy as np
from vis import plot_keypoints
from matplotlib import pyplot as plt


def run_ctd_inference(images, 
                      conditions, 
                      model_path=None,
                      vis_thres=0.0,
                      args=None,
                      ):

    model_dir = model_path
                          
    
    args.modelDir = model_dir
    args.logDir = ''

    model = get_model(args, model_dir)
    
    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    all_preds = []

    for i, image in enumerate(images):
        joints_list = conditions[i]
        preds = get_pose_feature(model, normalize, image, joints_list, vis_thres=vis_thres)
        all_preds.append(preds)

    all_preds = np.array(all_preds)

    return all_preds



def get_pose_feature(model, 
                     normalize, 
                     image_input, 
                     cond_joints_list, 
                     bboxes=None,
                     vis_thres=0.0):


    num_joints = cfg.MODEL.NUM_JOINTS

    if num_joints == 14:  ### 'crowdpose' dataset
        colors = [[245, 53, 53], [245, 125, 45], [253, 206, 20], [206, 244, 54], [118, 253, 27], [47, 254, 47], [25, 245, 113], [15, 243, 197], [14, 199, 245], [44, 126, 249], [13, 13, 249], [128, 47, 249], [205, 38, 247], [245, 48, 206]]
    elif num_joints == 17:
        colors = [[245, 59, 59], [249, 104, 25], [253, 183, 15], [233, 245, 41], [162, 252, 32], [84, 247, 34], [31, 252, 57], [20, 246, 126], [5, 249, 206], [52, 215, 249], [33, 136, 252], [11, 39, 248], [93, 46, 249], [156, 29, 244], [235, 49, 247], [245, 47, 187], [253, 44, 117]]
    
    image_input = np.array(image_input)

    concat_inputs = []
    center_list = []
    scale_list = []
    for i, cond_joints in enumerate(cond_joints_list):

        bbox = joints2box(cond_joints, image_input.shape, margin=25)

        center, scale = _box2cs(cfg, bbox)
        center_list.append(center)
        scale_list.append(scale)

        trans = get_affine_transform(center, scale, 0, cfg.MODEL.IMAGE_SIZE)
        
        data_numpy_copy = image_input.copy()

        data_numpy_copy = cv2.warpAffine(
            data_numpy_copy,
            trans,
            (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
            flags=cv2.INTER_LINEAR
        )
            
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        input = transform(data_numpy_copy)

        trans_joints = cond_joints.copy()
        for i, cond in enumerate(trans_joints):
            trans_joints[i, 0:2] = affine_transform(trans_joints[i, 0:2], trans)

        #  concate conditions
        cond_heatmap = get_condition_image_colored(trans_joints, size= (int(cfg.MODEL.IMAGE_SIZE[1]), int(cfg.MODEL.IMAGE_SIZE[0]), 3), colors=colors)
        cond_heatmap = np.transpose(cond_heatmap, (2, 0, 1))
        cond_heatmap = torch.from_numpy(cond_heatmap).float()

        ### vis conditions
        cond_heatmap_test = cond_heatmap.cpu().numpy().copy()
        cond_heatmap_test = np.transpose(cond_heatmap_test, (1, 2, 0))
        cv2.imwrite('cond_heatmap_test.jpg', cond_heatmap_test + data_numpy_copy)
        cv2.imwrite('input.jpg', data_numpy_copy)
        
        input = torch.cat((input, cond_heatmap))

        concat_inputs.append(input)
    
    # input = torch.unsqueeze(input, dim=0)
    concat_inputs = torch.stack(concat_inputs)
    outputs = model(concat_inputs)

    if isinstance(outputs, list):
        output = outputs[-1]
    else:
        output = outputs

    preds, maxvals = get_final_preds(cfg, output.cpu().detach().numpy(), center_list, scale_list)

    ##  filter low conf pred
    valid_preds = []
    for idx, xy in enumerate(preds):
        confs = maxvals[idx]

        for i in range(len(confs)):
            conf = confs[i][0]
            if conf < vis_thres:
                xy[i] = [np.nan, np.nan]

        valid_preds.append(xy)

    # ### visualize
    # colors = plt.cm.get_cmap('hsv', len(valid_preds))
    # print(len(valid_preds))
    # colors = [[colors(i)[0]*255, colors(i)[1]*255, colors(i)[2]*255 ] for i in range(colors.N)]
    # image2 = image_input.copy()[:, :, ::-1].astype(np.uint8)
    # for i, valid_pred in enumerate(valid_preds):
    #     image2 = plot_keypoints(image2, valid_pred, dataset='coco', color=colors[i])
    # cv2.imwrite('../pred.jpg', image2)

    return valid_preds


def generate_heatmap(heatmap, sigma):
    heatmap = cv2.GaussianBlur(heatmap, sigma, 0)
    am = np.amax(heatmap)
    if am == 0:
        return heatmap
    heatmap /= am / 255
    return heatmap

def get_condition_image_colored( kpts, size, colors=None):

    kpts = np.array(kpts).astype(int)
    zero_matrix = np.zeros(size)

    def _get_condition_matrix(zero_matrix, kpts):
        for color, kpt in zip(colors, kpts):
            if 0 < kpt[0] < size[1] and 0 < kpt[1] < size[0]:
                zero_matrix[kpt[1]-1][kpt[0]-1] = color
        return zero_matrix

    condition = _get_condition_matrix(zero_matrix, kpts)
    condition_heatmap = generate_heatmap(condition, sigma=(15, 15))

    return condition_heatmap


def get_model(args_cfg, model_state_file):

    update_config(cfg, args_cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(cfg, is_train=False)

    print('=> loading model from {}'.format(model_state_file))
    model.load_state_dict(torch.load(model_state_file, map_location=torch.device('cpu')))

    if torch.cuda.is_available():
        model = model.to(torch.device("cuda"))

    return model


def joints2box(joints, image_shape, margin=0):
    """
    :param joints: 2D joints, shape = (n_joints, 3)
    :return: bbox, shape = (4, )
    """    

    # joints shape (17, 3)
    if len(joints.shape) != 2:
        joints = np.array(joints).reshape(-1, 3)
    
    # nan to 0
    joints[np.isnan(joints)] = 0
    
    xmin = np.min(joints[:,0][np.nonzero(joints[:,0])]) - margin
    ymin = np.min(joints[:,1][np.nonzero(joints[:,1])]) - margin
    xmax = np.max(joints[:,0][np.nonzero(joints[:,0])]) + margin
    ymax = np.max(joints[:,1][np.nonzero(joints[:,1])]) + margin
    xmin = np.clip(xmin, 0, image_shape[1])
    ymin = np.clip(ymin, 0, image_shape[0])
    xmax = np.clip(xmax, 0, image_shape[1])
    ymax = np.clip(ymax, 0, image_shape[0])
    bbox = [xmin, ymin, xmax-xmin, ymax-ymin]

    return bbox

def _box2cs(cfg, box):
    x, y, w, h = box[:4]
    return _xywh2cs(cfg, x, y, w, h)

def _xywh2cs(cfg, x, y, w, h):
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    image_width = cfg.MODEL.IMAGE_SIZE[0]
    image_height = cfg.MODEL.IMAGE_SIZE[1]
    aspect_ratio = image_width * 1.0 / image_height

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    pixel_std = 200
    scale_thre = 1.25
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * scale_thre

    return center, scale



if __name__ == '__main__':

    image_file = '../media/000000.jpg'
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images = np.expand_dims(image, axis=0)
    print(images.shape)

    conditions = np.array([[[[402.34375, 277.03125], [407.96875, 274.21875], [402.34375, 274.21875], [419.21875, 277.03125], [399.53125, 274.21875], [436.09375, 293.90625], [393.90625, 293.90625], [469.84375, 296.71875], [360.15625, 293.90625], [447.34375, 305.15625], [371.40625, 302.34375], [427.65625, 361.40625], [399.53125, 361.40625], [447.34375, 412.03125], [382.65625, 420.46875], [464.21875, 448.59375], [365.78125, 468.28125]], [[509.21875, 361.40625], [509.21875, 355.78125], [506.40625, 355.78125], [506.40625, 358.59375], [495.15625, 358.59375], [512.03125, 383.90625], [478.28125, 381.09375], [537.34375, 378.28125], [467.03125, 372.65625], [540.15625, 381.09375], [489.53125, 378.28125], [512.03125, 445.78125], [483.90625, 448.59375], [528.90625, 485.15625], [481.09375, 496.40625], [542.96875, 513.28125], [455.78125, 535.78125]], [[835.46875, 248.90625], [835.46875, 246.09375], [829.84375, 246.09375], [838.28125, 243.28125], [821.40625, 246.09375], [835.46875, 257.34375], [815.78125, 262.96875], [866.40625, 257.34375], [829.84375, 282.65625], [852.34375, 257.34375], [843.90625, 268.59375], [829.84375, 316.40625], [812.96875, 319.21875], [841.09375, 355.78125], [801.71875, 364.21875], [846.71875, 392.34375], [773.59375, 403.59375]], [[683.59375, 347.34375], [689.21875, 341.71875], [683.59375, 341.71875], [700.46875, 344.53125], [677.96875, 344.53125], [706.09375, 361.40625], [677.96875, 361.40625], [720.15625, 361.40625], [649.84375, 361.40625], [725.78125, 355.78125], [655.46875, 367.03125], [694.84375, 423.28125], [675.15625, 420.46875], [711.71875, 454.21875], [675.15625, 468.28125], [706.09375, 482.34375], [652.65625, 502.03125]], [[914.21875, 291.09375], [917.03125, 288.28125], [911.40625, 288.28125], [917.03125, 288.28125], [900.15625, 291.09375], [905.78125, 307.96875], [897.34375, 313.59375], [919.84375, 307.96875], [925.46875, 327.65625], [928.28125, 305.15625], [933.90625, 307.96875], [908.59375, 372.65625], [888.90625, 372.65625], [919.84375, 414.84375], [880.46875, 423.28125], [925.46875, 457.03125], [857.96875, 468.28125]], [[638.59375, 257.34375], [644.21875, 254.53125], [635.78125, 254.53125], [647.03125, 254.53125], [630.15625, 251.71875], [649.84375, 265.78125], [621.71875, 265.78125], [675.15625, 265.78125], [635.78125, 277.03125], [675.15625, 265.78125], [655.46875, 277.03125], [647.03125, 324.84375], [624.53125, 324.84375], [652.65625, 367.03125], [616.09375, 375.46875], [661.09375, 406.40625], [587.96875, 414.84375]], [[78.90625, 46.40625], [78.90625, 40.78125], [78.90625, 40.78125], [87.34375, 43.59375], [84.53125, 40.78125], [101.40625, 63.28125], [87.34375, 63.28125], [101.40625, 94.21875], [87.34375, 94.21875], [95.78125, 119.53125], [81.71875, 116.71875], [95.78125, 122.34375], [84.53125, 122.34375], [92.96875, 161.71875], [81.71875, 161.71875], [104.21875, 201.09375], [84.53125, 201.09375]], [[1209.53125, 544.21875], [1209.53125, 541.40625], [1209.53125, 541.40625], [1215.15625, 544.21875], [1237.65625, 538.59375], [1217.96875, 580.78125], [1262.96875, 566.71875], [1201.09375, 620.15625], [1271.40625, 597.65625], [1170.15625, 614.53125], [1265.78125, 634.21875], [1234.84375, 642.65625], [1268.59375, 639.84375], [1203.90625, 639.84375], [1268.59375, 676.40625], [1206.71875, 676.40625], [1271.40625, 679.21875]], [[180.15625, 99.84375], [182.96875, 97.03125], [177.34375, 97.03125], [185.78125, 99.84375], [171.71875, 99.84375], [188.59375, 113.90625], [166.09375, 116.71875], [199.84375, 133.59375], [163.28125, 139.21875], [202.65625, 142.03125], [182.96875, 147.65625], [194.21875, 150.46875], [177.34375, 153.28125], [208.28125, 153.28125], [197.03125, 156.09375], [227.96875, 198.28125], [188.59375, 184.21875]], [[62.03125, 52.03125], [67.65625, 49.21875], [59.21875, 46.40625], [70.46875, 52.03125], [56.40625, 49.21875], [73.28125, 74.53125], [45.15625, 68.90625], [73.28125, 97.03125], [31.09375, 88.59375], [53.59375, 66.09375], [33.90625, 111.09375], [62.03125, 122.34375], [45.15625, 119.53125], [56.40625, 158.90625], [45.15625, 156.09375], [59.21875, 198.28125], [45.15625, 198.28125]], [[1201.09375, 600.46875], [1203.90625, 597.65625], [1203.90625, 597.65625], [1209.53125, 603.28125], [1232.03125, 600.46875], [1209.53125, 639.84375], [1243.28125, 628.59375], [1201.09375, 620.15625], [1257.34375, 665.15625], [1164.53125, 614.53125], [1170.15625, 614.53125], [1223.59375, 673.59375], [1254.53125, 679.21875], [1203.90625, 684.84375], [1254.53125, 696.09375], [1201.09375, 687.65625], [1254.53125, 710.15625]]]])

    model_path = 'COCO-BUCTD-CoAM-W48.pth'  ### run !wget https://zenodo.org/records/10039883/files/COCO-BUCTD-CoAM-W48.pth

    class Args:
        cfg = "../experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml"
        opts = [
            'MODEL.CONDITIONAL_TOPDOWN', True,
            'TEST.FLIP_TEST', True,
            'MODEL.NAME', 'pose_hrnet_coam',
            'MODEL.EXTRA.USE_ATTENTION', True,
            'MODEL.ATT_MODULES', '[False, True, False, False]',
            'MODEL.ATT_CHANNEL_ONLY', False,
            'MODEL.ATTENTION_HEADS', 1,
        ]
        dataDir = ''

    all_preds = run_ctd_inference(images, 
                                    conditions, 
                                    model_path=model_path,
                                    vis_thres=0.0,
                                    args = Args()
                                    )


    ### visualization  -- conditions
    conditions0 = conditions[0]
    colors = plt.cm.get_cmap('hsv', len(conditions0))
    colors = [[colors(i)[0]*255, colors(i)[1]*255, colors(i)[2]*255 ] for i in range(colors.N)]
    image2 = images[0].copy()
    for i, cond_joints in enumerate(conditions0):
        image2 = plot_keypoints(image2, cond_joints, dataset='coco', color=colors[i])
    cv2.putText(image2, 'conditions', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    ### visiualization -- predictions
    colors = plt.cm.get_cmap('hsv', len(all_preds[0]))
    colors = [[colors(i)[0]*255, colors(i)[1]*255, colors(i)[2]*255 ] for i in range(colors.N)]
    image3 = images[0].copy()
    for i, preds in enumerate(all_preds[0]):
        image3 = plot_keypoints(image3, preds, dataset='coco', color=colors[i])
    cv2.putText(image3, 'predictions', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    ## concat visualization
    image4 = np.concatenate((image2, image3), axis=1)
    cv2.imwrite('../vis.jpg', image4)