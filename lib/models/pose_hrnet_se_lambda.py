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

import torch
import torch.nn as nn
from torch.nn import init
import torchvision.transforms.functional as TF


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion,
                    momentum=BN_MOMENTUM
                ),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample
            )
        )
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index]
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels)
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1, 1, 0, bias=False
                            ),
                            nn.BatchNorm2d(num_inchannels[i]),
                            nn.Upsample(scale_factor=2**(j-i), mode='nearest')
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3)
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    nn.ReLU(True)
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class PoseHighResolutionNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg['MODEL']['EXTRA']
        super(PoseHighResolutionNet, self).__init__()

        self.cfg = cfg

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4)

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=False)

        self.final_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=cfg['MODEL']['NUM_JOINTS'],
            kernel_size=extra['FINAL_CONV_KERNEL'],
            stride=1,
            padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0
        )

        self.pretrained_layers = extra['PRETRAINED_LAYERS']

        # ------------------------------------------------

        if not cfg.MODEL.EXTRA.USE_ATTENTION:

            self.stage1_se = None
            self.stage2_se = None
            self.stage3_se = None
            self.stage4_se = None
            self.se_config = cfg.MODEL.SE_MODULES

            if self.se_config[0]:
                self.stage1_se = SELambdaModule(channel_list=self.stage2_cfg['NUM_CHANNELS'])
            if self.se_config[1]:
                self.stage2_se = SELambdaModule(channel_list=self.stage3_cfg['NUM_CHANNELS'])
            if self.se_config[2]:
                self.stage3_se = SELambdaModule(channel_list=self.stage4_cfg['NUM_CHANNELS'])
            if self.se_config[3]:
                self.stage4_se = SELambdaModule(channel_list=[self.stage4_cfg['NUM_CHANNELS'][0]]) ## last stage

        else:

            att_heads = self.cfg['MODEL']['ATTENTION_HEADS']

            # self.stage1_att = None
            # self.stage2_att = None
            # self.stage3_att = None
            # self.stage4_att = None
            # self.att_config = cfg.MODEL.ATT_MODULES

            # spat_dims = [int(cfg.MODEL.IMAGE_SIZE[0]/4),
            #              int(cfg.MODEL.IMAGE_SIZE[0]/8),
            #              int(cfg.MODEL.IMAGE_SIZE[0]/16),
            #              int(cfg.MODEL.IMAGE_SIZE[0]/32)]

            # if self.att_config[0]:
            #     self.stage1_att = AttentionModule(spat_dims=spat_dims[:2])
            # if self.att_config[1]:
            #     self.stage2_att = AttentionModule(spat_dims=spat_dims[:3])
            # if self.att_config[2]:
            #     self.stage3_att = AttentionModule(spat_dims=spat_dims[:])
            # if self.att_config[3]:
            #     #self.stage4_att = AttentionModule(spat_dims=spat_dims[:])
            #     self.stage4_att = AttentionModule(spat_dims=[spat_dims[0]])


            self.stage1_att = None
            self.stage2_att = None
            self.stage3_att = None
            self.stage4_att = None
            self.att_config = cfg.MODEL.ATT_MODULES
            self.selfatt_config = cfg.MODEL.SELFATT_MODULES

            spat_dims = [(int(cfg.MODEL.IMAGE_SIZE[0]/4),int(cfg.MODEL.IMAGE_SIZE[1]/4)),
                         (int(cfg.MODEL.IMAGE_SIZE[0]/8),int(cfg.MODEL.IMAGE_SIZE[1]/8)),
                         (int(cfg.MODEL.IMAGE_SIZE[0]/16),int(cfg.MODEL.IMAGE_SIZE[1]/16)),
                         (int(cfg.MODEL.IMAGE_SIZE[0]/32),int(cfg.MODEL.IMAGE_SIZE[1]/32))]

            assert not self.att_config[0] or not self.selfatt_config[0]
            assert not self.att_config[1] or not self.selfatt_config[1]
            assert not self.att_config[2] or not self.selfatt_config[2]
            assert not self.att_config[3] or not self.selfatt_config[3]

            if self.att_config[0]:
                self.stage1_att = AttentionModule(spat_dims=spat_dims[:2], channel_list=self.stage2_cfg['NUM_CHANNELS'],
                                                    cond_stacked=(self.cfg['DATASET']['STACKED_CONDITION'], self.cfg['MODEL']['NUM_JOINTS']),
                                                    cond_colored=self.cfg['DATASET']['COLORED'], n_heads=att_heads,
                                                    channel_only=self.cfg['MODEL']['ATT_CHANNEL_ONLY'],
                                                    spatial_only=self.cfg['MODEL']['ATT_SPATIAL_ONLY'])
            if self.att_config[1]:
                self.stage2_att = AttentionModule(spat_dims=spat_dims[:3], channel_list=self.stage3_cfg['NUM_CHANNELS'],
                                                    cond_stacked=(self.cfg['DATASET']['STACKED_CONDITION'], self.cfg['MODEL']['NUM_JOINTS']),
                                                    cond_colored=self.cfg['DATASET']['COLORED'], n_heads=att_heads,
                                                    channel_only=self.cfg['MODEL']['ATT_CHANNEL_ONLY'],
                                                    spatial_only=self.cfg['MODEL']['ATT_SPATIAL_ONLY'])
            if self.att_config[2]:
                self.stage3_att = AttentionModule(spat_dims=spat_dims[:], channel_list=self.stage4_cfg['NUM_CHANNELS'],
                                                    cond_stacked=(self.cfg['DATASET']['STACKED_CONDITION'], self.cfg['MODEL']['NUM_JOINTS']),
                                                    cond_colored=self.cfg['DATASET']['COLORED'], n_heads=att_heads,
                                                    channel_only=self.cfg['MODEL']['ATT_CHANNEL_ONLY'],
                                                    spatial_only=self.cfg['MODEL']['ATT_SPATIAL_ONLY'])
            if self.att_config[3]:
                #self.stage4_att = AttentionModule(spat_dims=spat_dims[:], channel_list=self.stage4_cfg['NUM_CHANNELS]')
                self.stage4_att = AttentionModule(spat_dims=[spat_dims[0]], channel_list=[self.stage4_cfg['NUM_CHANNELS'][0]],
                                                    cond_stacked=(self.cfg['DATASET']['STACKED_CONDITION'], self.cfg['MODEL']['NUM_JOINTS']),
                                                    cond_colored=self.cfg['DATASET']['COLORED'], n_heads=att_heads,
                                                    channel_only=self.cfg['MODEL']['ATT_CHANNEL_ONLY'],
                                                    spatial_only=self.cfg['MODEL']['ATT_SPATIAL_ONLY'])

            if self.selfatt_config[0]:
                self.stage1_att = SelfAttentionModule(spat_dims=spat_dims[:2], channel_list=self.stage2_cfg['NUM_CHANNELS'])
            if self.selfatt_config[1]:
                self.stage2_att = SelfAttentionModule(spat_dims=spat_dims[:3], channel_list=self.stage3_cfg['NUM_CHANNELS'])
            if self.selfatt_config[2]:
                self.stage3_att = SelfAttentionModule(spat_dims=spat_dims[:], channel_list=self.stage4_cfg['NUM_CHANNELS'])
            if self.selfatt_config[3]:
                #self.stage4_att = AttentionModule(spat_dims=spat_dims[:], channel_list=self.stage4_cfg['NUM_CHANNELS]')
                self.stage4_att = SelfAttentionModule(spat_dims=[spat_dims[0]], channel_list=[self.stage4_cfg['NUM_CHANNELS'][0]])

        # ------------------------------------------------

        return

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3, 1, 1, bias=False
                            ),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(inplace=True)
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(
                                inchannels, outchannels, 3, 2, 1, bias=False
                            ),
                            nn.BatchNorm2d(outchannels),
                            nn.ReLU(inplace=True)
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels


    def forward(self, x, lambda_vec=None):

        # # -------------------
        # ##------log time----------
        # import time
        # start_time = time.time()
        # # -------------------
        x = x.cuda()

        if self.cfg.MODEL.EXTRA.USE_ATTENTION:
            x_ = x[:,:3]
            cond_hm = x[:,3:]
        else:
            x_ = x

        x = self.conv1(x_)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)

        # -------------------
        if not self.cfg.MODEL.EXTRA.USE_ATTENTION:
            if self.se_config[0]:
                x_list = self.stage1_se(x_list, lambda_vec) ## 2, (32, 64), space_res = 64 x 48
        else:
            if self.att_config[0]:
                x_list = self.stage1_att(x_list, cond_hm)
        # -------------------

        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
    
        # -------------------
        if not self.cfg.MODEL.EXTRA.USE_ATTENTION:
            if self.se_config[1]:
                x_list = self.stage2_se(x_list, lambda_vec) ## 3, (32, 64, 128) , space_res = 64 x 48
        else:
            if self.att_config[1]:
                x_list = self.stage2_att(x_list, cond_hm)
        # -------------------

        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])

        # # -----------------------------
        # ##------log time----------
        # mid_time = time.time() - start_time
        # print("--- %s common seconds ---" % (mid_time))

        # -------------------
        if not self.cfg.MODEL.EXTRA.USE_ATTENTION:
            if self.se_config[2]:
                x_list = self.stage3_se(x_list, lambda_vec)  ## 4, (32, 64, 128, 256) , space_res = 64 x 48
        else:
            if self.att_config[2]:
                x_list = self.stage3_att(x_list, cond_hm)
        # -------------------

        y_list = self.stage4(x_list)

        # -------------------
        if not self.cfg.MODEL.EXTRA.USE_ATTENTION:
            if self.se_config[3]:
                y_list = self.stage4_se(y_list, lambda_vec)  ## 1, (32) , space_res = 64 x 48
        else:
            if self.att_config[3]:
                x_list = self.stage4_att(y_list, cond_hm)
        # -------------------

        x = self.final_layer(y_list[0])

        # end_time = time.time() - start_time
        # diff_time = end_time - mid_time
        # print("--- %s end seconds ---" % (end_time))
        # print("--- %s lambda seconds ---" % (mid_time + 2*diff_time))

        return x

    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                   or self.pretrained_layers[0] is '*':
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)
        elif pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(pretrained))


def get_pose_net(cfg, is_train, **kwargs):
    model = PoseHighResolutionNet(cfg, **kwargs)

    if is_train and cfg['MODEL']['INIT_WEIGHTS']:
        model.init_weights(cfg['MODEL']['PRETRAINED'])

    return model


# ---------------------------------------------------
class SELambdaLayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELambdaLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

        self.fc_lambda = nn.Sequential(
            nn.Linear(2, channel // reduction),
            nn.BatchNorm1d(channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

        return

    def forward(self, x, lambda_vec):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        z = self.fc_lambda(lambda_vec).view(b, c, 1, 1)

        out = x * y.expand_as(x) * z.expand_as(x)
        return out

# ---------------------------------------------------
class SELambdaModule(nn.Module):
    def __init__(self, channel_list):
        super(SELambdaModule, self).__init__()
        self.se_layers = []

        for i in range(len(channel_list)):
            se_layer = SELambdaLayer(channel=channel_list[i], reduction=4)
            self.se_layers.append(se_layer)

        self.se_layers = nn.ModuleList(self.se_layers)
        return

    def forward(self, y_list, lambda_vec):
        y_list_se = []
        for i in range(len(y_list)):
            y_se = self.se_layers[i](y_list[i], lambda_vec)
            y_list_se.append(y_se)

        return y_list_se

# ---------------------------------------------------




# modified from https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/model/attention/DANet.py


from models.self_attention import ScaledDotProductAttention
from models.self_attention import SimplifiedScaledDotProductAttention


class PositionAttentionModule(nn.Module):

    def __init__(self, d_model=512, d_cond=3, kernel_size=3, H=7, W=7, n_heads=1, self_att=False):
        super().__init__()
        self.cnn = nn.Conv2d(d_model, d_model, kernel_size=kernel_size,
                             padding=(kernel_size-1)//2)
        self.pa = ScaledDotProductAttention(in_dim_q = d_model, in_dim_k = d_model,
                                            d_k = d_model, d_v = d_model, h = n_heads)
        self.self_att = self_att
        if not self_att:
            self.cnn_cond = nn.Conv2d(d_cond, d_cond, kernel_size=kernel_size, padding=(kernel_size-1)//2)
            self.pa = ScaledDotProductAttention(in_dim_q = d_cond, in_dim_k = d_model,
                                                d_k = d_model, d_v = d_model, h = n_heads)
    
    def forward(self,x,cond=None):
        bs,c,h,w = x.shape
        y = self.cnn(x)
        y = y.view(bs,c,-1).permute(0,2,1) #bs,h*w,c

        if not self.self_att:
            _,c_cond,_,_ = cond.shape
            y_cond = self.cnn_cond(cond)
            y_cond = y_cond.view(bs,c_cond,-1).permute(0,2,1)
            y = self.pa(y_cond, y, y) #bs,h*w,c
            #y = self.pa(y, y_cond, y_cond) #bs,h*w,c
        
        else:
            y = self.pa(y,y,y)
            
        return y

class ChannelAttentionModule(nn.Module):
    
    def __init__(self, d_model=512, d_cond=3, kernel_size=3, H=7, W=7, n_heads=1, self_att=False):
        super().__init__()
        self.cnn = nn.Conv2d(d_model, d_model, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.self_att = self_att
        if not self_att:
            #self.cnn_cond = nn.Conv2d(d_cond, d_cond, kernel_size=kernel_size, padding=(kernel_size-1)//2)
            self.cnn_cond = nn.Conv2d(d_cond, d_model, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.pa = SimplifiedScaledDotProductAttention(H*W, h = n_heads)
    
    def forward(self,x,cond=None):
        bs,c,h,w = x.shape
        y = self.cnn(x)
        y = y.view(bs,c,-1) #bs,c,h*w

        if not self.self_att:
            #_,c_cond,_,_ = cond.shape
            y_cond = self.cnn_cond(cond)
            #y_cond = y_cond.view(bs,c_cond,-1)
            y_cond = y_cond.view(bs,c,-1)
            y = self.pa(y_cond, y, y) #bs,c_cond,h*w
            #y = self.pa(y, y_cond, y_cond)

        else:
            y = self.pa(y,y,y) #bs,c,h*w
            
        return y


class DAModule(nn.Module):

    def __init__(self, d_model=512, d_cond=3, kernel_size=3, H=7, W=7, n_heads=1,
                    channel_only=False, spatial_only=False):
        super().__init__()
        self.channel_only = channel_only
        self.spatial_only = spatial_only
        if not channel_only:
            self.position_attention_module=PositionAttentionModule(d_model=d_model, d_cond=d_cond,
                                                                   kernel_size=kernel_size, H=H, W=W,
                                                                   n_heads=n_heads)
        if not self.spatial_only:
            self.channel_attention_module=ChannelAttentionModule(d_model=d_model, d_cond=d_cond,
                                                                 kernel_size=kernel_size, H=H, W=W,
                                                                 n_heads=n_heads)
    
    def forward(self,input,cond):

        bs,c,h,w = input.shape
        # give condition to the same number of channels as feature maps
        #cond = cond.expand(-1,c,-1,-1)
        #_,c_cond,_,_ = cond.shape

        if self.channel_only and self.spatial_only:
            return input

        if not self.spatial_only:
            c_out = self.channel_attention_module(input, cond)
            c_out = c_out.view(bs,c,h,w)
            #c_out = c_out.view(bs,c_cond,h,w)

            if self.channel_only:
                #return input * c_out
                return input + c_out

        if not self.channel_only:
            p_out = self.position_attention_module(input, cond)
            p_out = p_out.permute(0,2,1).view(bs,c,h,w)
            #p_out = p_out.permute(0,2,1).view(bs,c_cond,h,w)

            if self.spatial_only:
                return input + p_out
        
        #return p_out + c_out
        return input + (p_out + c_out)
        #return (input + p_out) + (input + c_out)


class AttentionModule(nn.Module):

    def __init__(self, spat_dims, channel_list, cond_stacked, cond_colored, n_heads=1,
                    channel_only=False, spatial_only=False):
        super(AttentionModule, self).__init__()
        self.att_layers = []
        self.spat_dims = spat_dims
        self.cond_color = cond_colored
        self.cond_stacked = cond_stacked
        if cond_stacked[0]:
            d_cond = cond_stacked[1]
        elif cond_colored:
            d_cond = 3
        else:
            d_cond = 1
        for i in range(len(spat_dims)):
            att_layer = DAModule(d_model = channel_list[i],
                                 d_cond = d_cond, kernel_size = 3,
                                 H = spat_dims[i][1], W = spat_dims[i][0],
                                 n_heads = n_heads,
                                 channel_only = channel_only,
                                 spatial_only = spatial_only)
            self.att_layers.append(att_layer)
        self.att_layers = nn.ModuleList(self.att_layers)

    def forward(self, y_list, cond_hm):
        if not self.cond_color and not self.cond_stacked[0]:
            cond_hm = cond_hm[:,0].unsqueeze(1) # we only want one channel of the heatmap
        y_list_att = []
        for i in range(len(y_list)):
            y_att = self.att_layers[i](y_list[i], TF.resize(cond_hm, (self.spat_dims[i][1],self.spat_dims[i][0])))
            y_list_att.append(y_att)
        return y_list_att



class SelfDAModule(nn.Module):

    def __init__(self, d_model=512, kernel_size=3, H=7, W=7):
        super().__init__()
        self.position_attention_module=PositionAttentionModule(d_model=d_model, d_cond=None,
                                                               kernel_size=kernel_size, H=H, W=W,
                                                               self_att=True)
        self.channel_attention_module=ChannelAttentionModule(d_model=d_model, d_cond=None,
                                                             kernel_size=kernel_size, H=H, W=W,
                                                             self_att=True)
    
    def forward(self,input):

        bs,c,h,w = input.shape

        p_out = self.position_attention_module(input)
        c_out = self.channel_attention_module(input)
        
        p_out = p_out.permute(0,2,1).view(bs,c,h,w)
        c_out = c_out.view(bs,c,h,w)
        
        return p_out + c_out


class SelfAttentionModule(nn.Module):

    def __init__(self, spat_dims, channel_list):
        super(SelfAttentionModule, self).__init__()
        self.att_layers = []
        for i in range(len(spat_dims)):
            att_layer = SelfDAModule(d_model = channel_list[i], kernel_size = 3,
                                     H = spat_dims[i][0], W = spat_dims[i][1])
            self.att_layers.append(att_layer)
        self.att_layers = nn.ModuleList(self.att_layers)

    def forward(self, y_list, *args):
        y_list_att = []
        for i in range(len(y_list)):
            y_att = self.att_layers[i](y_list[i])
            y_list_att.append(y_att)
        return y_list_att

