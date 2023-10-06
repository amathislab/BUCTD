# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from collections import namedtuple
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms

import random
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import cv2

from PIL import Image

# ----------------------------------------
# standard COCO format, 17 joints
COCO_KP_ORDER = [
        'nose',
        'left_eye',
        'right_eye',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle'
    ]

def kp_connections(keypoints):
    kp_lines = [
        [keypoints.index('left_eye'), keypoints.index('right_eye')],
        [keypoints.index('left_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('right_ear')],
        [keypoints.index('left_eye'), keypoints.index('left_ear')],
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        [keypoints.index('right_hip'), keypoints.index('right_knee')],
        [keypoints.index('right_knee'), keypoints.index('right_ankle')],
        [keypoints.index('left_hip'), keypoints.index('left_knee')],
        [keypoints.index('left_knee'), keypoints.index('left_ankle')],
        [keypoints.index('right_shoulder'), keypoints.index('left_shoulder')],
        [keypoints.index('right_hip'), keypoints.index('left_hip')],
    ]
    return kp_lines


COCO_KP_CONNECTIONS = kp_connections(COCO_KP_ORDER)

# ----------------------------------------
def makedir(dir_list):
    for dir in dir_list:
        if not os.path.exists(dir):
            os.makedirs(dir)
    return

# --------------------------------------------------------------------------------
def set_seed(seed_id=0):
    random.seed(seed_id)
    np.random.seed(seed_id)
    torch.manual_seed(seed_id)
    torch.cuda.manual_seed_all(seed_id)
    return

# --------------------------------------------------------------------------------
def batch_unnormalize_image(images, normalize=transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])):
    images[:, 0, :, :] = (images[:, 0, :, :]*normalize.std[0]) + normalize.mean[0] 
    images[:, 1, :, :] = (images[:, 1, :, :]*normalize.std[1]) + normalize.mean[1] 
    images[:, 2, :, :] = (images[:, 2, :, :]*normalize.std[2]) + normalize.mean[2] 
    images = 255*images
    return images

# ----------------------------------------
def vis_segmentation(img, mask, alpha=0.5):
    if mask is not None:
        color = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3)).astype(img.dtype)
        not_mask = (mask == 0)
        for i in range(3):
            color_mask[:, :, i] = not_mask[:, :]*img[:, :, i] + mask[:, :]*int(color[i]*255)

        result = cv2.addWeighted(img, 1.0 - alpha, color_mask, alpha, 0)
    else:
        x1, y1, w, h = bbox
        result = cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (255, 255, 0), 1)

    return result

# ----------------------------------------
def vis_bbs(img, bbox, score_dict=None):
    if score_dict is None:
        return vis_intro_bbs(img, bbox)

    x1, y1, w, h = bbox
    result = cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (255, 255, 0), 1)
    # ax.text(x1, y1 - 2, str(round(score, 2)), fontsize=3, family='serif', bbox=dict(facecolor='g', alpha=0.4, pad=0, edgecolor='none'), color='white')
    string = '[{},{},{}]'.format(round(score_dict['score'], 2), round(score_dict['box_score'], 1), round(score_dict['keypoint_score'],2))
    result = cv2.putText(result, string, (int(x1), int(y1-2)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    return result

def vis_intro_bbs(img, bbox):
    x1, y1, w, h = bbox
    result = cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (255, 255, 0), 2)
    return result


def vis_keypoints(img, kps, kp_thresh=-1, alpha=0.7):
    """Visualizes keypoints (adapted from vis_one_image).
    kps has shape (3, #keypoints) where 3 rows are (x, y, depth z).
    needs a BGR image as it only uses opencv functions, returns a bgr image
    """

    # line_thickness = 2 ## default
    # line_thickness = 5 ## default
    line_thickness = 8 ## default


    ## -------- do not draw invisible joints---------
    invalid_kps = (kps[2, :] == 0)
    kps[2, invalid_kps] = kp_thresh - 1


    sc_kps = kps.copy()
    kps = kps.copy().astype(np.int16)

    # ------------------------------------------------
    dataset_keypoints = COCO_KP_ORDER
    kp_lines = COCO_KP_CONNECTIONS

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw mid shoulder / mid hip first for better visualization.
    mid_shoulder = (
        kps[:2, dataset_keypoints.index('right_shoulder')] +
        kps[:2, dataset_keypoints.index('left_shoulder')]) // 2
    sc_mid_shoulder = np.minimum(
        sc_kps[2, dataset_keypoints.index('right_shoulder')],
        sc_kps[2, dataset_keypoints.index('left_shoulder')])
    mid_hip = (
        kps[:2, dataset_keypoints.index('right_hip')] +
        kps[:2, dataset_keypoints.index('left_hip')]) // 2
    sc_mid_hip = np.minimum(
        sc_kps[2, dataset_keypoints.index('right_hip')],
        sc_kps[2, dataset_keypoints.index('left_hip')])
    nose_idx = dataset_keypoints.index('nose')

    # ----------------------------------------------------


    # ----------------------------------------------------
    if sc_mid_shoulder > kp_thresh and kps[2, nose_idx] > kp_thresh:
        kp_mask = cv2.line(
            kp_mask, tuple(mid_shoulder), tuple(kps[:2, nose_idx]),
            color=colors[len(kp_lines)], thickness=line_thickness, lineType=cv2.LINE_AA)
    if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
        kp_mask = cv2.line(
            kp_mask, tuple(mid_shoulder), tuple(mid_hip),
            color=colors[len(kp_lines) + 1], thickness=line_thickness, lineType=cv2.LINE_AA)

    # Draw the keypoints.
    for l in range(len(kp_lines)):
        i1 = kp_lines[l][0]
        i2 = kp_lines[l][1]
        p1 = kps[0, i1], kps[1, i1]
        p2 = kps[0, i2], kps[1, i2]
        if sc_kps[2, i1] > kp_thresh and sc_kps[2, i2] > kp_thresh:
            kp_mask = cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=line_thickness, lineType=cv2.LINE_AA)
        if sc_kps[2, i1] > kp_thresh:
            kp_mask = cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if sc_kps[2, i2] > kp_thresh:
            kp_mask = cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    ## weird opencv bug on cv2UMat vs numpy
    if type(kp_mask) != type(img):
        kp_mask = kp_mask.get()

    # Blend the keypoints.
    result = cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)
    return result

# ----------------------------------------
def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir(parents=True)

    dataset = cfg.DATASET.DATASET + '_' + cfg.DATASET.HYBRID_JOINTS_TYPE \
        if cfg.DATASET.HYBRID_JOINTS_TYPE else cfg.DATASET.DATASET
    dataset = dataset.replace(':', '_')
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / model / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
        (cfg_name + '_' + time_str)

    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.TRAIN.LR
        )

    return optimizer

## we only use adam
def get_last_layer_optimizer(cfg, model):
    optimizer = None

    ## freeze everything except last layer
    for name, param in model.named_parameters():
        if 'final_layer' not in name:
            param.requires_grad = False

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.TRAIN.LR
    )

    return optimizer

# --------------------------------------------------------------------------------
def get_network_grad_flow(model):
    total_avg_gradient = 0.0
    for name, param in model.named_parameters():
        if param.requires_grad:
            avg_gradient = param.grad.abs().mean().item()
            total_avg_gradient += avg_gradient

    return total_avg_gradient


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        torch.save(states['best_state_dict'],
                   os.path.join(output_dir, 'model_best.pth'))


def get_model_summary(model, *input_tensors, item_length=26, verbose=False):
    """
    :param model:
    :param input_tensors:
    :param item_length:
    :return:
    """

    summary = []

    ModuleDetails = namedtuple(
        "Layer", ["name", "input_size", "output_size", "num_parameters", "multiply_adds"])
    hooks = []
    layer_instances = {}

    def add_hooks(module):

        def hook(module, input, output):
            class_name = str(module.__class__.__name__)

            instance_index = 1
            if class_name not in layer_instances:
                layer_instances[class_name] = instance_index
            else:
                instance_index = layer_instances[class_name] + 1
                layer_instances[class_name] = instance_index

            layer_name = class_name + "_" + str(instance_index)

            params = 0

            if class_name.find("Conv") != -1 or class_name.find("BatchNorm") != -1 or \
               class_name.find("Linear") != -1:
                for param_ in module.parameters():
                    params += param_.view(-1).size(0)

            flops = "Not Available"
            if class_name.find("Conv") != -1 and hasattr(module, "weight"):
                flops = (
                    torch.prod(
                        torch.LongTensor(list(module.weight.data.size()))) *
                    torch.prod(
                        torch.LongTensor(list(output.size())[2:]))).item()
            elif isinstance(module, nn.Linear):
                flops = (torch.prod(torch.LongTensor(list(output.size()))) \
                         * input[0].size(1)).item()

            if isinstance(input[0], list):
                input = input[0]
            if isinstance(output, list):
                output = output[0]

            summary.append(
                ModuleDetails(
                    name=layer_name,
                    input_size=list(input[0].size()),
                    output_size=list(output.size()),
                    num_parameters=params,
                    multiply_adds=flops)
            )

        if not isinstance(module, nn.ModuleList) \
           and not isinstance(module, nn.Sequential) \
           and module != model:
            hooks.append(module.register_forward_hook(hook))

    model.eval()
    model.apply(add_hooks)

    space_len = item_length

    model(*input_tensors)
    for hook in hooks:
        hook.remove()

    details = ''
    if verbose:
        details = "Model Summary" + \
            os.linesep + \
            "Name{}Input Size{}Output Size{}Parameters{}Multiply Adds (Flops){}".format(
                ' ' * (space_len - len("Name")),
                ' ' * (space_len - len("Input Size")),
                ' ' * (space_len - len("Output Size")),
                ' ' * (space_len - len("Parameters")),
                ' ' * (space_len - len("Multiply Adds (Flops)"))) \
                + os.linesep + '-' * space_len * 5 + os.linesep
    params_sum = 0
    flops_sum = 0
    for layer in summary:
        params_sum += layer.num_parameters
        if layer.multiply_adds != "Not Available":
            flops_sum += layer.multiply_adds
        if verbose:
            details += "{}{}{}{}{}{}{}{}{}{}".format(
                layer.name,
                ' ' * (space_len - len(layer.name)),
                layer.input_size,
                ' ' * (space_len - len(str(layer.input_size))),
                layer.output_size,
                ' ' * (space_len - len(str(layer.output_size))),
                layer.num_parameters,
                ' ' * (space_len - len(str(layer.num_parameters))),
                layer.multiply_adds,
                ' ' * (space_len - len(str(layer.multiply_adds)))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    details += os.linesep \
        + "Total Parameters: {:,}".format(params_sum) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Total Multiply Adds (For Convolution and Linear Layers only): {:,} GFLOPs".format(flops_sum/(1024**3)) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Number of Layers" + os.linesep
    for layer in layer_instances:
        details += "{} : {} layers   ".format(layer, layer_instances[layer])

    return details

def get_lambda_model_summary(model, input_tensors, lambda_tensors, item_length=26, verbose=False):
    """
    :param model:
    :param input_tensors:
    :param item_length:
    :return:
    """

    summary = []

    ModuleDetails = namedtuple(
        "Layer", ["name", "input_size", "output_size", "num_parameters", "multiply_adds"])
    hooks = []
    layer_instances = {}

    def add_hooks(module):

        def hook(module, input, output):
            class_name = str(module.__class__.__name__)

            instance_index = 1
            if class_name not in layer_instances:
                layer_instances[class_name] = instance_index
            else:
                instance_index = layer_instances[class_name] + 1
                layer_instances[class_name] = instance_index

            layer_name = class_name + "_" + str(instance_index)
            params = 0

            if class_name.find("Conv") != -1 or class_name.find("BatchNorm") != -1 or \
               class_name.find("Linear") != -1:
                for param_ in module.parameters():
                    params += param_.view(-1).size(0)

            flops = "Not Available"
            if class_name.find("Conv") != -1 and hasattr(module, "weight"):
                flops = (
                    torch.prod(
                        torch.LongTensor(list(module.weight.data.size()))) *
                    torch.prod(
                        torch.LongTensor(list(output.size())[2:]))).item()
            elif isinstance(module, nn.Linear):
                flops = (torch.prod(torch.LongTensor(list(output.size()))) \
                         * input[0].size(1)).item()

            if isinstance(input[0], list):
                input = input[0]
            if isinstance(output, list):
                output = output[0]

            summary.append(
                ModuleDetails(
                    name=layer_name,
                    input_size=list(input[0].size()),
                    output_size=list(output.size()),
                    num_parameters=params,
                    multiply_adds=flops)
            )

        if not isinstance(module, nn.ModuleList) \
           and not isinstance(module, nn.Sequential) \
           and module != model:
            hooks.append(module.register_forward_hook(hook))

    model.eval()
    model.apply(add_hooks)

    space_len = item_length

    model(input_tensors, lambda_tensors)
    for hook in hooks:
        hook.remove()

    details = ''
    if verbose:
        details = "Model Summary" + \
            os.linesep + \
            "Name{}Input Size{}Output Size{}Parameters{}Multiply Adds (Flops){}".format(
                ' ' * (space_len - len("Name")),
                ' ' * (space_len - len("Input Size")),
                ' ' * (space_len - len("Output Size")),
                ' ' * (space_len - len("Parameters")),
                ' ' * (space_len - len("Multiply Adds (Flops)"))) \
                + os.linesep + '-' * space_len * 5 + os.linesep
    params_sum = 0
    flops_sum = 0

    common_sum = 0
    lambda_started = False

    for layer in summary:

        if 'AdaptiveAvgPool' in layer.name and lambda_started == False:
            lambda_started = True
            common_sum = flops_sum

        params_sum += layer.num_parameters
        if layer.multiply_adds != "Not Available":
            flops_sum += layer.multiply_adds
        if verbose:
            details += "{}{}{}{}{}{}{}{}{}{}".format(
                layer.name,
                ' ' * (space_len - len(layer.name)),
                layer.input_size,
                ' ' * (space_len - len(str(layer.input_size))),
                layer.output_size,
                ' ' * (space_len - len(str(layer.output_size))),
                layer.num_parameters,
                ' ' * (space_len - len(str(layer.num_parameters))),
                layer.multiply_adds,
                ' ' * (space_len - len(str(layer.multiply_adds)))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    diff_flops_sum = flops_sum - common_sum
    total_flops_sum = common_sum + 2*diff_flops_sum
    flops_sum = total_flops_sum

    details += os.linesep \
        + "Total Parameters: {:,}".format(params_sum) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Total Multiply Adds (For Convolution and Linear Layers only): {:,} GFLOPs".format(flops_sum/(1024**3)) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Number of Layers" + os.linesep
    for layer in layer_instances:
        details += "{} : {} layers   ".format(layer, layer_instances[layer])

    return details

def compute_iou(bbox_1, bbox_2):

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

def compute_ious(anns):
    num_boxes = len(anns)
    ious = np.zeros((num_boxes, num_boxes))

    for i in range(num_boxes):
        for j in range(i,num_boxes):
            ious[i,j] = compute_iou(anns[i]['bbox'],anns[j]['bbox'])
            if i!=j:
                ious[j,i] = ious[i,j]
    return ious
