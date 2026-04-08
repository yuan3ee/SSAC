import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = super().forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super().__init__()
        out_channels = 256
        modules = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        ]

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 处理低层特征（来自ResNet的layer2）
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(512, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )

        # 最终输出层
        self.final_conv = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x_aspp, x_low_level):
        x_low_level = self.low_level_conv(x_low_level)
        x_aspp = F.interpolate(x_aspp, size=x_low_level.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x_aspp, x_low_level], dim=1)
        return self.final_conv(x)


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Backbone (ResNet50 with dilated convolutions)
        backbone = resnet50(pretrained=True, replace_stride_with_dilation=[False, False, True])
        self.layer0 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )
        self.layer1 = backbone.layer1  # 256 channels
        self.layer2 = backbone.layer2  # 512 channels
        self.layer3 = backbone.layer3  # 1024 channels
        self.layer4 = backbone.layer4  # 2048 channels

        # ASPP模块（输入通道2048）
        self.aspp = ASPP(2048, [6, 12, 18])

        # 解码器
        self.decoder = Decoder(num_classes)

        # 初始化权重
        self._init_weight()

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]

        # 前向传播
        x = self.layer0(x)  # 1/4
        x = self.layer1(x)  # 1/4
        x_low = self.layer2(x)  # 1/8
        x = self.layer3(x_low)  # 1/16
        x = self.layer4(x)  # 1/16

        # ASPP处理
        x_aspp = self.aspp(x)

        # 解码器
        x_decoder = self.decoder(x_aspp, x_low)

        # 上采样到原图尺寸
        x_out = F.interpolate(x_decoder, size=(h, w), mode='bilinear', align_corners=False)
        return x_out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    # 测试代码
    model = DeepLabV3Plus(num_classes=21)
    input_tensor = torch.randn(2, 3, 512, 512)
    output = model(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")  # 应该输出 torch.Size([2, 21, 512, 512])