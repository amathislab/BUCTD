# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import time
import logging
import os

import numpy as np
import torch

from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------
def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None, epoch=-1, print_prefix=''):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6+1)) ## update to add annotation ids
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):

            if not config.MODEL.CONDITIONAL_TOPDOWN:
                input = input[:,:3]

            # compute output
            outputs = model(input)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                outputs_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()


                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()
            annotation_id = meta['annotation_id'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1) ## this is the area of the detection
            all_boxes[idx:idx + num_images, 5] = score
            all_boxes[idx:idx + num_images, 6] = annotation_id
            image_path.extend(meta['image'])

            idx += num_images

            if (i % config.PRINT_FREQ == 0) or (i == (len(val_loader)-1)):
                save_size = 16
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.6f} ({loss.avg:.6f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader)-1, batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_epoch_{:09d}_iter_{}_{}'.format(os.path.join(output_dir, 'val'), epoch, i, print_prefix)

                save_debug_images(config, input[:save_size, [2,1,0], :, :], meta, target[:save_size], (pred*4)[:save_size], output[:save_size],
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path, epoch,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator

# --------------------------------------------------------------------------------
def validate_lambda_quantitative(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None, epoch=-1, print_prefix='', lambda_vals=[0, 1]):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # switch to evaluate mode
    model.eval()

    num_samples = len(lambda_vals)*len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6+1+1)) ## update to add annotation ids and mode l=0,1
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):

            B, C, H, W = input.shape

            for lambda_idx, lambda_val in enumerate(lambda_vals):
                lambda_a = lambda_val*torch.ones(B, 1).cuda()
                lambda_vec = torch.cat([lambda_a, 1 - lambda_a], dim=1)        
                outputs = model(input, lambda_vec)
                output = outputs

                if config.TEST.FLIP_TEST:
                    input_flipped = input.flip(3)
                    outputs_flipped = model(input_flipped, lambda_vec)
                    output_flipped = outputs_flipped

                    output_flipped = flip_back(output_flipped.cpu().numpy(),
                                               val_dataset.flip_pairs)
                    output_flipped = torch.from_numpy(output_flipped.copy()).cuda()


                    # feature is not aligned, shift flipped heatmap for higher accuracy
                    if config.TEST.SHIFT_HEATMAP:
                        output_flipped[:, :, :, 1:] = \
                            output_flipped.clone()[:, :, :, 0:-1]

                    output = (output + output_flipped) * 0.5

                target = target.cuda(non_blocking=True)
                target_weight = target_weight.cuda(non_blocking=True)

                loss = criterion(output, target, target_weight)

                num_images = input.size(0)
                # measure accuracy and record loss
                losses.update(loss.item(), num_images)
                _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                                 target.cpu().numpy())

                acc.update(avg_acc, cnt)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                c = meta['center'].numpy()
                s = meta['scale'].numpy()
                
                if lambda_val == 0:
                    score = meta['score'].numpy()*config.TEST.DECAY_THRE

                else:
                    score = meta['score'].numpy()

                annotation_id = meta['annotation_id'].numpy()

                preds, maxvals = get_final_preds(
                    config, output.clone().cpu().numpy(), c, s)

                all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
                all_preds[idx:idx + num_images, :, 2:3] = maxvals
                # double check this all_boxes parts
                all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
                all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
                all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
                all_boxes[idx:idx + num_images, 5] = score
                all_boxes[idx:idx + num_images, 6] = annotation_id
                all_boxes[idx:idx + num_images, 7] = lambda_a.detach().cpu().numpy().reshape(-1)

                image_path.extend(meta['image'])

                idx += num_images

                if (i % config.PRINT_FREQ == 0) or (i == (len(val_loader)-1)):
                    save_size = min(16, B)
                    msg = 'Test: [{0}/{1}]\t' \
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                          'Loss {loss.val:.6f} ({loss.avg:.6f})\t' \
                          'Accuracy {acc.val:.3f} ({acc.avg:.3f}) \t' \
                          'Lambda_a {lambda_val:.3f}'.format(
                              i, len(val_loader)-1, batch_time=batch_time,
                              loss=losses, acc=acc, lambda_val=lambda_val)
                    logger.info(msg)

                    meta['pred_joints_vis'] = torch.ones_like(meta['joints_vis'])
                    
                    prefix = '{}_epoch_{:09d}_iter_{}_{}'.format(os.path.join(output_dir, 'val'), epoch, i, print_prefix)
                    suffix = 'a'
                    for count in range(min(save_size, len(lambda_a))):
                        suffix += '_[{}:{}]'.format(count, round(lambda_a[count].item(), 2))

                    save_debug_images(config, input[:save_size, [2,1,0], :, :], meta, target[:save_size], (pred*4)[:save_size], output[:save_size],
                                      prefix, suffix)

        name_values, name_values_mode0, \
        name_values_mode1, name_values_mode2, \
        name_values_mode3, perf_indicator = val_dataset.evaluate(
                                    config, all_preds, output_dir, all_boxes, image_path, epoch,
                                    filenames, imgnums
                                )

        model_name = config.MODEL.NAME
    
        _print_name_value(name_values, 'l0,1:{}'.format(model_name))
        _print_name_value(name_values_mode0, 'l0:{}'.format(model_name))
        _print_name_value(name_values_mode1, 'l1:{}'.format(model_name))

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator

# --------------------------------------------------------------------------------
def validate_lambda(config, val_loader, val_dataset, model, criterion_lambda, criterion, output_dir,
             tb_log_dir, writer_dict, epoch=-1, print_prefix=''):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target_a, target_weight_a, meta_a) in enumerate(val_loader):

            import copy
            target_b = copy.deepcopy(target_a)
            target_weight_b = copy.deepcopy(target_weight_a)
            meta_b = copy.deepcopy(meta_a)

            B, C, H, W = input.shape

            for lambda_a_val in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
                lambda_a = lambda_a_val*torch.ones(B, 1).cuda()
                lambda_b = 1 - lambda_a
                lambda_vec = torch.cat([lambda_a, lambda_b], dim=1)

                # compute output
                outputs = model(input, lambda_vec)
                output = outputs

                # print('lambda_a:{} out:{}, mu:{}, sigma:{}'.format(lambda_a_val, output.sum(), mu.sum(), sigma.sum()))
                
                target_a = target_a.cuda(non_blocking=True)
                target_weight_a = target_weight_a.cuda(non_blocking=True)

                target_b = target_b.cuda(non_blocking=True)
                target_weight_b = target_weight_b.cuda(non_blocking=True)

                loss_a_lambda = criterion_lambda(output, target_a, target_weight_a)
                loss_b_lambda = criterion_lambda(output, target_b, target_weight_b)

                loss = (loss_a_lambda*lambda_a.view(-1)).mean() + (loss_b_lambda*lambda_b.view(-1)).mean()

                num_images = input.size(0)
                # measure accuracy and record loss
                losses.update(loss.item(), num_images)
                _, avg_acc, cnt, pred_a = accuracy(output.cpu().numpy(),
                                                 target_a.cpu().numpy())

                _, avg_acc, cnt, pred_b = accuracy(output.cpu().numpy(),
                                                 target_b.cpu().numpy())

                acc.update(avg_acc, cnt)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if (i % config.PRINT_FREQ == 0) or (i == (len(val_loader)-1)):
                    save_size = 16
                    msg = 'Test: [{0}/{1}]\t' \
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                          'Loss {loss.val:.6f} ({loss.avg:.6f})\t' \
                          'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t' \
                          'lambda_a {lambda_val:.3f}\tout {out:.3f}'.format(
                              i, len(val_loader)-1, batch_time=batch_time,
                              loss=losses, acc=acc, lambda_val=lambda_a_val, out=output.sum()
                            )
                    logger.info(msg)

                    prefix = '{}_epoch_{:09d}_iter_{}_{}'.format(os.path.join(output_dir, 'val'), epoch, i, print_prefix)
                    suffix = 'a'
                    for count in range(min(save_size, len(lambda_a))):
                        suffix += '_[{}:{}]'.format(count, round(lambda_a[count].item(), 2))

                    save_debug_images(config, input[:save_size, [2,1,0], :, :], meta_a, target_a[:save_size], (pred_a*4)[:save_size], output[:save_size],
                                      prefix, suffix)

                    prefix = '{}_epoch_{:09d}_iter_{}_{}'.format(os.path.join(output_dir, 'val'), epoch, i, print_prefix)
                    suffix = 'b'
                    for count in range(min(save_size, len(lambda_b))):
                        suffix += '_[{}:{}]'.format(count, round(lambda_b[count].item(), 2))

                    save_debug_images(config, input[:save_size, [2,1,0], :, :], meta_b, target_b[:save_size], (pred_b*4)[:save_size], output[:save_size],
                                      prefix, suffix)

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar(
            'valid_loss',
            losses.avg,
            global_steps
        )
        writer.add_scalar(
            'valid_acc',
            acc.avg,
            global_steps
        )
        writer_dict['valid_global_steps'] = global_steps + 1
        
    return 0


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
