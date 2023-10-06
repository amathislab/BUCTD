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
from core.inference import get_max_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images
from utils.utils import get_network_grad_flow
from torch.nn.utils import clip_grad_norm

# from utils.vis import save_pretty_debug_images as save_debug_images

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------
def synthetic_train(config, synthetic_train_loader, real_train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    real_loader_iter = iter(real_train_loader)

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(synthetic_train_loader):
        batch_size, _, _, _ = input.shape

        # measure data loading time
        data_time.update(time.time() - end)

        # -----------------------------------------------
        # compute output
        outputs = model(input)

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        if isinstance(outputs, list):
            loss = criterion(outputs[0], target, target_weight)
            for output in outputs[1:]:
                loss += criterion(output, target, target_weight)
        else:
            output = outputs
            loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Syn Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(synthetic_train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('syn_train_loss', losses.val, global_steps)
            writer.add_scalar('syn_train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_epoch_{}_iter_{}_syn'.format(os.path.join(output_dir, 'train'), epoch, i)
            save_debug_images(config, input[:16, [2,1,0], :, :], meta, target[:16], (pred*4)[:16], output[:16],
                                  prefix)

# --------------------------------------------------------------------------------
def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict, print_prefix=''):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if not config.MODEL.EXTRA.USE_ATTENTION:
            input = input[:,:3]

        # compute output
        outputs = model(input)

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        if isinstance(outputs, list):
            loss = criterion(outputs[0], target, target_weight)
            for output in outputs[1:]:
                loss += criterion(output, target, target_weight)
        else:
            output = outputs
            loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        # gradient clipping
        clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_epoch_{}_iter_{}_{}'.format(os.path.join(output_dir, 'train'), epoch, i, print_prefix)
            save_debug_images(config, input[:16, [2,1,0], :, :], meta, target[:16], (pred*4)[:16], output[:16],
                                  prefix)

    return

# --------------------------------------------------------------------------------
def train_cutmix(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict, print_prefix=''):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target_f, target_weight_f, meta_f, target_b, target_weight_b, meta_b) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs = model(input)

        target_f = target_f.cuda(non_blocking=True)
        target_weight_f = target_weight_f.cuda(non_blocking=True)

        target_b = target_b.cuda(non_blocking=True)
        target_weight_b = target_weight_b.cuda(non_blocking=True)
        
        output = outputs

        ## weight target_weights by lambda
        lambda_f = meta_f['lambda'].cuda().view(-1, 1, 1).repeat(1, 17, 1) ## shape: B x 17 x 1
        lambda_b = meta_b['lambda'].cuda().view(-1, 1, 1).repeat(1, 17, 1) ## shape: B x 17 x 1

        weighted_target_weight_f = target_weight_f*lambda_f
        weighted_target_weight_b = target_weight_b*lambda_b

        loss_f = criterion(output, target_f, weighted_target_weight_f)
        loss_b = criterion(output, target_b, weighted_target_weight_b)

        loss = loss_f + loss_b

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred_f = accuracy(output.detach().cpu().numpy(),
                                         target_f.detach().cpu().numpy())
        _, avg_acc, cnt, pred_b = accuracy(output.detach().cpu().numpy(),
                                         target_b.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_epoch_{}_iter_{}_{}_foreground'.format(os.path.join(output_dir, 'train'), epoch, i, print_prefix)
            save_debug_images(config, input[:16, [2,1,0], :, :], meta_f, target_f[:16], (pred_f*4)[:16], output[:16],
                                  prefix)

            prefix = '{}_epoch_{}_iter_{}_{}_background'.format(os.path.join(output_dir, 'train'), epoch, i, print_prefix)
            save_debug_images(config, input[:16, [2,1,0], :, :], meta_b, target_b[:16], (pred_b*4)[:16], output[:16],
                                  prefix)

    return
## --------------------------------------------------------------------------------
def train_mixup(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict, print_prefix=''):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target_f, target_weight_f, meta_f, target_b, target_weight_b, meta_b) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs = model(input)

        target_f = target_f.cuda(non_blocking=True)
        target_weight_f = target_weight_f.cuda(non_blocking=True)

        target_b = target_b.cuda(non_blocking=True)
        target_weight_b = target_weight_b.cuda(non_blocking=True)
        
        output = outputs

        ## weight target_weights by lambda
        lambda_f = meta_f['lambda'].cuda().view(-1, 1, 1).repeat(1, 17, 1) ## shape: B x 17 x 1
        lambda_b = meta_b['lambda'].cuda().view(-1, 1, 1).repeat(1, 17, 1) ## shape: B x 17 x 1

        weighted_target_weight_f = target_weight_f*lambda_f
        weighted_target_weight_b = target_weight_b*lambda_b

        loss_f = criterion(output, target_f, weighted_target_weight_f)
        loss_b = criterion(output, target_b, weighted_target_weight_b)

        loss = loss_f + loss_b

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred_f = accuracy(output.detach().cpu().numpy(),
                                         target_f.detach().cpu().numpy())
        _, avg_acc, cnt, pred_b = accuracy(output.detach().cpu().numpy(),
                                         target_b.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_epoch_{}_iter_{}_{}_foreground'.format(os.path.join(output_dir, 'train'), epoch, i, print_prefix)
            save_debug_images(config, input[:16, [2,1,0], :, :], meta_f, target_f[:16], (pred_f*4)[:16], output[:16], prefix)

            prefix = '{}_epoch_{}_iter_{}_{}_background'.format(os.path.join(output_dir, 'train'), epoch, i, print_prefix)
            save_debug_images(config, input[:16, [2,1,0], :, :], meta_b, target_b[:16], (pred_b*4)[:16], output[:16], prefix)

    return


# --------------------------------------------------------------------------------
def train_lambda(config, train_loader, model, criterion_lambda, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict, print_prefix=''):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    model_grads = AverageMeter()
    diversity_losses = AverageMeter()
    pose_losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target_a, target_weight_a, meta_a, target_b, target_weight_b, meta_b) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        B, C, H, W = input.shape

        ##--- uniformly random--------
        # lambda_a = torch.rand(B, 1).cuda()
        
        ##--- 0s and 1s--------
        lambda_a = torch.rand(B, 1).cuda()        
        lambda_a = (lambda_a > 0.5).float()*1.0
        complement_lambda_a = (lambda_a == 0).float()*1.0

        # --------------duplicate-----------------------------
        lambda_a = torch.cat([lambda_a, complement_lambda_a], dim=0)
        input = torch.cat([input, input], dim=0)
        target_a = torch.cat([target_a, target_a], dim=0)
        target_weight_a = torch.cat([target_weight_a, target_weight_a], dim=0)
        target_b = torch.cat([target_b, target_b], dim=0)
        target_weight_b = torch.cat([target_weight_b, target_weight_b], dim=0)
        meta_a['joints'] = torch.cat([meta_a['joints'], meta_a['joints']], dim=0)
        meta_a['joints_vis'] = torch.cat([meta_a['joints_vis'], meta_a['joints_vis']], dim=0)
        meta_b['joints'] = torch.cat([meta_b['joints'], meta_b['joints']], dim=0)
        meta_b['joints_vis'] = torch.cat([meta_b['joints_vis'], meta_b['joints_vis']], dim=0)

        # --------------------------------
        lambda_b = 1 - lambda_a
        lambda_vec = torch.cat([lambda_a, lambda_b], dim=1) ### B x 2

        # compute output
        outputs = model(input, lambda_vec)

        target_a = target_a.cuda(non_blocking=True)
        target_weight_a = target_weight_a.cuda(non_blocking=True)

        target_b = target_b.cuda(non_blocking=True)
        target_weight_b = target_weight_b.cuda(non_blocking=True)
        
        output = outputs

        loss_a_lambda = criterion_lambda(output, target_a, target_weight_a)
        loss_b_lambda = criterion_lambda(output, target_b, target_weight_b)

        ## as input[:B] == input[B:]; lambda_a[:B] == lambda_b[B:]
        diversity_loss = -1*criterion(output[:B], output[B:], target_weight_a[:B]*target_weight_b[:B]) ## try to push apart the common joints

        pose_loss = (loss_a_lambda*lambda_a.view(-1)).mean() + (loss_b_lambda*lambda_b.view(-1)).mean()
        # loss = pose_loss + 0.1*diversity_loss
        loss = pose_loss

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model_grad = get_network_grad_flow(model)
        model_grads.update(model_grad)

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        diversity_losses.update(diversity_loss.item(), input.size(0))
        pose_losses.update(pose_loss.item(), input.size(0))

        _, avg_acc, cnt, pred_a = accuracy(output.detach().cpu().numpy(),
                                         target_a.detach().cpu().numpy())
        _, avg_acc, cnt, pred_b = accuracy(output.detach().cpu().numpy(),
                                         target_b.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        msg = 'Epoch: [{0}][{1}/{2}]\t' \
              'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
              'Speed {speed:.1f} samples/s\t' \
              'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
              'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
              'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t' \
              'model_grad {model_grad.val:.6f} ({model_grad.avg:.6f})\t' \
              'DivLoss {diversity_loss.val:.5f} ({diversity_loss.avg:.5f})\t' \
              'PoseLoss {pose_loss.val:.5f} ({pose_loss.avg:.5f})\t'.format(
                  epoch, i, len(train_loader), batch_time=batch_time,
                  speed=input.size(0)/batch_time.val,
                  data_time=data_time, loss=losses, acc=acc,
                  model_grad=model_grads, diversity_loss=diversity_losses,
                  pose_loss=pose_losses)
        logger.info(msg)

        if i % config.PRINT_FREQ == 0:
            save_size = 16
            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_epoch_{:09d}_iter_{}_{}'.format(os.path.join(output_dir, 'train'), epoch, i, print_prefix)
            suffix = 'a'
            for count in range(min(save_size, len(lambda_a))):
                suffix += '_[{}:{}]'.format(count, round(lambda_a[count].item(), 2))

            meta_a['pred_joints_vis'] = torch.ones_like(meta_a['joints_vis'])
            save_debug_images(config, input[:save_size, [2,1,0], :, :], meta_a, target_a[:save_size], (pred_a*4)[:save_size], output[:save_size], prefix, suffix)

            prefix = '{}_epoch_{:09d}_iter_{}_{}'.format(os.path.join(output_dir, 'train'), epoch, i, print_prefix)
            suffix = 'b'
            for count in range(min(save_size, len(lambda_a))):
                suffix += '_[{}:{}]'.format(count, round(lambda_a[count + B].item(), 2))

            meta_b['pred_joints_vis'] = torch.ones_like(meta_b['joints_vis'])
            save_debug_images(config, input[B:B+save_size, [2,1,0], :, :], meta_b, target_b[B:B+save_size], (pred_b*4)[B:B+save_size], output[B:B+save_size], prefix, suffix)


    return

# --------------------------------------------------------------------------------
def train_cutout(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict, print_prefix=''):
    return train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict, print_prefix)

# --------------------------------------------------------------------------------
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
