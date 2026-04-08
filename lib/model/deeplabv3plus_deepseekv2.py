import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

# ASPP 模块
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPP, self).__init__()
        modules = []
        # 1x1 卷积
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ))
        # 空洞卷积
        for rate in atrous_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ))
        # 全局平均池化
        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ))
        self.convs = nn.ModuleList(modules)
        # 融合层
        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x) if not isinstance(conv[-1], nn.AdaptiveAvgPool2d) else
                       F.interpolate(conv(x), size=x.shape[2:], mode='bilinear', align_corners=False))
        # for i, t in enumerate(res):
        #     print(f"Tensor {i}: {t.shape}")
        res[-1] = F.interpolate(res[-1], size=(64, 64), mode='bilinear', align_corners=True)
        res = torch.cat(res, dim=1)
        return self.project(res)

# 解码器模块
class Decoder(nn.Module):
    def __init__(self, low_level_channels, num_classes):
        super(Decoder, self).__init__()
        # 低层特征处理
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
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

# DeepLabV3+ 网络
class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes, backbone='resnet50'):
        super(DeepLabV3Plus, self).__init__()
        # Backbone (ResNet50)
        self.backbone = resnet50(pretrained=True, replace_stride_with_dilation=[False, True, True])
        self.layer0 = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool
        )
        self.layer1 = self.backbone.layer1  # 256 channels
        self.layer2 = self.backbone.layer2  # 512 channels
        self.layer3 = self.backbone.layer3  # 1024 channels
        self.layer4 = self.backbone.layer4  # 2048 channels

        # ASPP 模块
        self.aspp = ASPP(2048, 256, [6, 12, 18])

        # 解码器
        self.decoder = Decoder(512, num_classes)

        # 初始化权重
        self._init_weight()

    def forward(self, x):
        # 编码器部分
        x0 = self.layer0(x)  # 1/4
        x1 = self.layer1(x0)  # 1/4
        x2 = self.layer2(x1)  # 1/8
        x3 = self.layer3(x2)  # 1/16
        x4 = self.layer4(x3)  # 1/16

        # ASPP 处理
        x_aspp = self.aspp(x4)

        # 解码器部分
        x_decoder = self.decoder(x_aspp, x2)

        # 上采样到原图尺寸
        x_out = F.interpolate(x_decoder, size=x.shape[2:], mode='bilinear', align_corners=False)

        # 返回预测结果和编码器特征
        return x_out  #{
            # 'layer0': x0,  # 1/4 特征
            # 'layer1': x1,  # 1/4 特征
            # 'layer2': x2,  # 1/8 特征
            # 'layer3': x3,  # 1/16 特征
            # 'layer4': x4   # 1/16 特征
        # }

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

# 测试代码
if __name__ == "__main__":
    model = DeepLabV3Plus(num_classes=21)
    input_tensor = torch.randn(2, 3, 512, 512)
    output, encoder_features = model(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")  # 预测结果
    for name, feature in encoder_features.items():
        print(f"{name} feature shape: {feature.shape}")  # 编码器特征