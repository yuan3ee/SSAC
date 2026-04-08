#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com), Xiangtai(lxtpku@pku.edu.cn)
# Modified by: RainbowSecret(yuyua@microsoft.com)
# Select Seg Model for img segmentation.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import pdb
import math
import logging
import torch
import torch.nn as nn
from collections import OrderedDict


# import our stuffs
# for train
# from .bn_helper import BatchNorm2d, BatchNorm2d_class
# for test
BatchNorm2d_class = BatchNorm2d = torch.nn.BatchNorm2d


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.relu_in = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes)
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

        out = out + residual
        out = self.relu_in(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_in = nn.ReLU(inplace=True)
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

        out = out + residual
        out = self.relu_in(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, deep_base=False):
        super(ResNet, self).__init__()
        self.inplanes = 128 if deep_base else 64
        if deep_base:
            self.resinit = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)),
                ('bn1', BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=False)),
                ('conv2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)),
                ('bn2', BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=False)),
                ('conv3', nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)),
                ('bn3', BatchNorm2d(self.inplanes)),
                ('relu3', nn.ReLU(inplace=False))]
            ))
        else:
            self.resinit = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)),
                ('bn1', BatchNorm2d(self.inplanes)),
                ('relu1', nn.ReLU(inplace=False))]
            ))

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change.

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm2d_class):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.resinit(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNetModels(object):

    def __init__(self, configer):
        self.configer = configer

    def resnet18(self, **kwargs):
        """Constructs a ResNet-18 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on Places
        """
        model = ResNet(BasicBlock, [2, 2, 2, 2], deep_base=False, **kwargs)
        model = self.load_model(model, pretrained=self.configer)
        return model

    def resnet34(self, **kwargs):
        """Constructs a ResNet-34 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on Places
        """
        model = ResNet(BasicBlock, [3, 4, 6, 3], deep_base=False, **kwargs)
        model = self.load_model(model, pretrained=self.configer)
        return model

    def resnet50(self, **kwargs):
        """Constructs a ResNet-50 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on Places
        """
        model = ResNet(Bottleneck, [3, 4, 6, 3], deep_base=False, **kwargs)
        model = self.load_model(model, pretrained=self.configer)
        return model

    def resnet101(self, **kwargs):
        """Constructs a ResNet-101 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on Places
        """
        model = ResNet(Bottleneck, [3, 4, 23, 3], deep_base=False, **kwargs)
        model = self.load_model(model, pretrained=self.configer)
        return model

    @staticmethod
    def load_model(model, pretrained=None, network='resnet50'):
        if pretrained is None:
            return model

        logging.info('Loading pretrained model:{}'.format(pretrained))
        pretrained_dict = torch.load(pretrained)
        model_dict = model.state_dict()
        load_dict = dict()
        for k, v in pretrained_dict.items():
            if 'resinit.{}'.format(k) in model_dict:
                load_dict['resinit.{}'.format(k)] = v
            else:
                load_dict[k] = v
        for k, _ in load_dict.items():
            print('=> loading {} pretrained model {}'.format(k, pretrained))
        model.load_state_dict(load_dict)
        return model


class NormalResnetBackbone(nn.Module):
    
    def __init__(self, orig_resnet):
        super(NormalResnetBackbone, self).__init__()

        self.num_features = 2048
        # take pretrained resnet, except AvgPool and FC
        self.resinit = orig_resnet.resinit
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def get_num_features(self):
        return self.num_features

    def forward(self, x):
        tuple_features = list()
        x = self.resinit(x)
        # tuple_features.append(x)
        x = self.maxpool(x)
        # tuple_features.append(x)
        x = self.layer1(x)
        tuple_features.append(x)
        x = self.layer2(x)
        tuple_features.append(x)
        x = self.layer3(x)
        tuple_features.append(x)
        x = self.layer4(x)
        tuple_features.append(x)

        return tuple_features[0], tuple_features[1], tuple_features[2], tuple_features[3]


class ResNetBackbone(object):

    def __init__(self, configer):
        self.configer = configer
        self.resnet_models = ResNetModels(self.configer)

    def __call__(self, arch = 'resnet50'):
        if arch == 'resnet18':
            orig_resnet = self.resnet_models.resnet18()
            arch_net = NormalResnetBackbone(orig_resnet)
        elif arch == 'resnet34':
            orig_resnet = self.resnet_models.resnet34()
            arch_net = NormalResnetBackbone(orig_resnet)
            arch_net.num_features = 512
        elif arch == 'resnet50':
            orig_resnet = self.resnet_models.resnet50()
            arch_net = NormalResnetBackbone(orig_resnet)
        elif arch == 'resnet101':
            orig_resnet = self.resnet_models.resnet101()
            arch_net = NormalResnetBackbone(orig_resnet)
        else:
            raise Exception('Architecture undefined!')

        return arch_net