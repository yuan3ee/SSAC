# -*- coding: utf-8 -*-
"""
 @Time    : 2019/1/30 23:19
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""

from . import deeplabv2, deeplabv3, deeplabv3plus_resnet, deeplabv2_ly, deeplabv2_unimatch
from .base.msc import MSC


def get_models(args):
    if args.model == 'deeplabv2':
        if args.msc:
            return MSC(scale= deeplabv2.resnet101(n_class=args.num_classes, pretrained=True),  pyramids=[0.5, 0.75])
        return deeplabv2.resnet101(n_class=args.num_classes, pretrained=True)
    elif args.model == 'deeplabv2_ly':
        if args.msc:
            return MSC(scale= deeplabv2_ly.resnet101(n_class=args.num_classes, pretrained=True),  pyramids=[0.5, 0.75])
        return deeplabv2_ly.resnet101(n_class=args.num_classes, pretrained=True)
    elif args.model == 'deeplabv2_unimatch':
        if args.msc:
            return MSC(scale= deeplabv2_unimatch.resnet101(n_class=args.num_classes, pretrained=True),  pyramids=[0.5, 0.75])
        return deeplabv2_unimatch.resnet101(n_class=args.num_classes, pretrained=True)
    elif args.model == 'deeplabv3':
        if args.msc:
            return MSC(scale= deeplabv3.resnet101(n_class=args.num_classes, output_stride=16, pretrained=True),
                       pyramids=[0.5, 0.75])
        return deeplabv3.resnet101(n_class=args.num_classes, output_stride=16, pretrained=True)
    elif args.model == 'deeplabv3plus':
        if args.msc:
            return MSC(scale= deeplabv3plus_resnet.resnet101(n_class=args.num_classes, output_stride=16, pretrained=True),
                       pyramids=[0.5, 0.75])
        return deeplabv3plus_resnet.resnet101(n_class=args.num_classes, output_stride=16, pretrained=True)
    else:
        print('Model {} not implemented.'.format(args.model))
        raise NotImplementedError
