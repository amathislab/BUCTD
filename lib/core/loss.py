# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.gaussian import GaussianSmoothing


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze() ## size=(B, 64 x 48)
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints

class JointsLambdaMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsLambdaMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze() ## size=(B, 64 x 48)
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        loss = loss.mean(dim=1) / num_joints
        return loss


class JointsExpectationLoss(nn.Module):
    def __init__(self):
        super(JointsExpectationLoss, self).__init__()
        self.criterion = nn.L1Loss(reduction='mean')
        self.gaussian_smoothing = GaussianSmoothing(channels=17, kernel_size=11, sigma=6) ## 11 copied from dark
        # output = self.gaussian_smoothing(F.pad(output, (5, 5, 5, 5), mode='reflect'))
        # self.cutoff_threshold = 0.0001234098 ##this is the min value in gt heatmaps

        return

    def forward(self, output, target_joint, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        width = output.size(3)

        device = output.get_device()

        # --------------------------------------------
        heatmaps_pred = output.split(1, 1) ## split B x 17 x 64 x 48 tensor into 17 single chunks along dim 1
        gt_joints = target_joint.split(1, 1) ## split B x 17 x 64 x 48 tensor into 17 single chunks along dim 1
        locs = torch.arange(output.size(2)*output.size(3)).to(device)

        loss = 0
        # --------------------------------------------
        for idx in range(num_joints):
            original_heatmap_pred = heatmaps_pred[idx].squeeze() ## size=(B, 64 , 48)
            gt_joint = gt_joints[idx].squeeze() ## size=(B, 2)

            heatmap_pred = original_heatmap_pred.view(batch_size, -1)

            # #---------------------------------
            # heatmap_pred = F.softmax(heatmap_pred, dim=1)
            # heatmap_pred = heatmap_pred.clamp(min=1e-10)
            # expected_pred = (heatmap_pred*locs).sum(dim=1)/heatmap_pred.sum(dim=1)

            #---------------------------------
            # heatmap_pred = F.relu(heatmap_pred)
            # heatmap_pred = heatmap_pred.clamp(min=1e-10)
            # expected_pred = (heatmap_pred*locs).sum(dim=1)/heatmap_pred.sum(dim=1)

            #---------------------------------
            heatmap_pred = heatmap_pred.clamp(min=1e-10)
            expected_pred = (heatmap_pred*locs).sum(dim=1)/25.0813 ## B
            
            #---------------------------------
            # expected_pred = expected_pred.view(batch_size, 1) ## B x 1

            # expected_pred = expected_pred.repeat(1, 2)
            # expected_pred[:, 0] = expected_pred[:, 0] % width
            # expected_pred[:, 1] = expected_pred[:, 1] / width

            # loss += self.criterion(
            #     expected_pred.mul(target_weight[:, idx]),
            #     gt_joint.mul(target_weight[:, idx])
            # )

            #---------------------------------
            expected_pred = expected_pred.view(-1, 1)
            linear_gt_joint = width*gt_joint[:, 1] + gt_joint[:, 0]
            linear_gt_joint = linear_gt_joint.view(-1, 1)

            loss += self.criterion(
                expected_pred.mul(target_weight[:, idx]),
                linear_gt_joint.mul(target_weight[:, idx])
            )

        loss = loss / num_joints
        return loss

class JointsOHKMMSELoss(nn.Module):
    def __init__(self, use_target_weight, topk=8):
        super(JointsOHKMMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk

    def ohkm(self, loss):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False
            )
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss.append(0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                ))
            else:
                loss.append(
                    0.5 * self.criterion(heatmap_pred, heatmap_gt)
                )

        loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
        loss = torch.cat(loss, dim=1)

        return self.ohkm(loss)

# ----------------------------------------------------------------------





