from pyexpat import features

from base_model import VGG
import torch.nn as nn
import torch


class UNet_CL(nn.Module):
    def __init__(self, band, num_classes):
        super().__init__()
        self.num_classes = num_classes
        base_model = VGG(layer=16, band=band)
        cn = [64, 128, 256, 512, 512]
        self.backbone = base_model
        self.layers = ["concat1", "concat2", "concat3", "concat4", "downsample4"]
        self.relu = nn.ReLU(inplace=True)
        self.decoder_layer1 = nn.Sequential(
            nn.Conv2d(cn[3], cn[3] * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cn[3] * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(cn[3] * 2, cn[3] * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cn[3] * 2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(cn[3] * 2, cn[3], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cn[3]),
            nn.ReLU(inplace=True)
        )
        self.decoder_layer2 = nn.Sequential(
            nn.Conv2d(cn[3] * 2, cn[3], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cn[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(cn[3], cn[3], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cn[3]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(cn[3], cn[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cn[2]),
            nn.ReLU(inplace=True)
        )
        self.decoder_layer3 = nn.Sequential(
            nn.Conv2d(cn[2] * 2, cn[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cn[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(cn[2], cn[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cn[2]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(cn[2], cn[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cn[1]),
            nn.ReLU(inplace=True)
        )
        self.decoder_layer4 = nn.Sequential(
            nn.Conv2d(cn[1] * 2, cn[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cn[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(cn[1], cn[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cn[1]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(cn[1], cn[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cn[0]),
            nn.ReLU(inplace=True)
        )
        self.decoder_layer5 = nn.Sequential(
            nn.Conv2d(cn[0] * 2, cn[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cn[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(cn[0], cn[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cn[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(cn[0], num_classes, kernel_size=1, padding=0, bias=False),
        )
        self.fea_contrast = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        self.pred_contrast = nn.Sequential(nn.Conv2d(256, self.num_classes, 1))


    def forward(self, x, is_train=True, fp=False):
        output = {}
        for name, layer in self.backbone._modules.items():
            x = layer(x)
            if name in self.layers:
                output[name] = x
        x1 = output['concat1']
        x2 = output['concat2']
        x3 = output['concat3']
        x4 = output['concat4']
        x5 = output['downsample4']

        # 对比学习用
        fea_contrast = self.fea_contrast(x5)
        pre_contrast = self.pred_contrast(fea_contrast)

        score = self.decoder_layer1(x5)
        score = torch.cat([score, x4], dim=1)
        score = self.decoder_layer2(score)
        score = torch.cat([score, x3], dim=1)
        score = self.decoder_layer3(score)
        score = torch.cat([score, x2], dim=1)
        score = self.decoder_layer4(score)
        score = torch.cat([score, x1], dim=1)
        output = self.decoder_layer5(score)
        if fp is True:
            score = self.decoder_layer1(nn.Dropout2d(0.5)(x5))
            score = torch.cat([score, nn.Dropout2d(0.5)(x4)], dim=1)
            score = self.decoder_layer2(score)
            score = torch.cat([score, nn.Dropout2d(0.5)(x3)], dim=1)
            score = self.decoder_layer3(score)
            score = torch.cat([score, nn.Dropout2d(0.5)(x2)], dim=1)
            score = self.decoder_layer4(score)
            score = torch.cat([score, nn.Dropout2d(0.5)(x1)], dim=1)
            output_fp = self.decoder_layer5(score)
            return output, output_fp, fea_contrast, pre_contrast

        return output, fea_contrast, pre_contrast