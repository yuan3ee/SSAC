# -*- coding: utf-8 -*-
"""
 @Time    : 2019/1/30 16:59
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""

from torch.utils import model_zoo

from .base.resnet import *
from .base.oprations import *


class DeeplabV2(ResNet):
    def __init__(self, n_class, block, layers, pyramids):
        print("Constructing DeepLabv2 model...")
        print("Number of classes: {}".format(n_class))
        super(DeeplabV2, self).__init__()

        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, rate=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, rate=4)

        self.aspp1 = ASPP_module(2048, n_class, pyramids[0])
        self.aspp2 = ASPP_module(2048, n_class, pyramids[1])
        self.aspp3 = ASPP_module(2048, n_class, pyramids[2])
        self.aspp4 = ASPP_module(2048, n_class, pyramids[3])

        self.init_weight()
        prev_final_channel = 1024
        self.dsn = nn.Sequential(
            nn.Conv2d(prev_final_channel, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, n_class, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, input, input_j=None, gts=None, aux_gts=None, fp=False):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        aux_out = x
        x = self.layer4(x)

        if fp is True:
            x1 = self.aspp1(torch.cat([x, nn.Dropout2d(0.5)(x)]))
            x2 = self.aspp2(torch.cat([x, nn.Dropout2d(0.5)(x)]))
            x3 = self.aspp3(torch.cat([x, nn.Dropout2d(0.5)(x)]))
            x4 = self.aspp4(torch.cat([x, nn.Dropout2d(0.5)(x)]))

            x = x1 + x2 + x3 + x4
            x = F.upsample(x, size=input.size()[2:], mode='bilinear', align_corners=True)
            out, out_drop = x.chunk(2)
            return out, out_drop

        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)

        x = x1 + x2 + x3 + x4
        x = F.upsample(x, size=input.size()[2:], mode='bilinear', align_corners=True)


        if aux_gts is not None:
            aux_out = self.dsn(aux_out)
            if aux_gts.dim() == 1:
                aux_gts = gts
            aux_gts = aux_gts.unsqueeze(1).float()
            aux_gts = nn.functional.interpolate(aux_gts, size=aux_out.shape[2:], mode='nearest')
            aux_gts = aux_gts.squeeze(1).long()

        if input_j is not None:
            x_j = self.conv1(input_j)
            x_j = self.bn1(x_j)
            x_j = self.relu(x_j)
            x_j = self.maxpool(x_j)

            x_j = self.layer1(x_j)
            x_j = self.layer2(x_j)
            x_j = self.layer3(x_j)
            x_j = self.layer4(x_j)

            x1_j = self.aspp1(x_j)
            x2_j = self.aspp2(x_j)
            x3_j = self.aspp3(x_j)
            x4_j = self.aspp4(x_j)

            x_j = x1_j + x2_j + x3_j + x4_j
            x_j = F.upsample(x_j, size=input_j.size()[2:], mode='bilinear', align_corners=True)
            return x, x_j, aux_out, aux_gts
        return x

    def get_1x_lr_params(self):
        b = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        for i in range(len(b)):
            for k in b[i].parameters():
                if k.requires_grad:
                    yield k

    def get_10x_lr_params(self):
        b = [self.aspp1, self.aspp2, self.aspp3, self.aspp4]
        for j in range(len(b)):
            for k in b[j].parameters():
                if k.requires_grad:
                    yield k

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def freeze_backbone_bn(self):
        self.bn1.eval()

        for m in self.layer1:
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        for m in self.layer2:
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        for m in self.layer3:
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        for m in self.layer4:
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


def resnet101(n_class, pretrained=True):

    model = DeeplabV2(n_class=n_class, block=Bottleneck, layers=[3, 4, 23, 3], pyramids=[6, 12, 18, 24])

    if pretrained:
        pretrain_dict = model_zoo.load_url(model_urls['resnet101'])
        model_dict = {}
        state_dict = model.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        model.load_state_dict(state_dict)

    return model


if __name__ == '__main__':
    model = resnet101(n_class=21, pretrained=True)

    img = torch.randn(4, 3, 513, 513)

    with torch.no_grad():
        output = model.forward(img)

    print(output.size())
