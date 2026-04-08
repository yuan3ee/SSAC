import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# -------------------------------
# ASPP 模块
# -------------------------------
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        """
        Args:
            in_channels: 输入特征通道数（例如 2048）
            out_channels: 每个分支输出的通道数（例如 256）
            atrous_rates: 空洞卷积的膨胀率列表，例如 [6, 12, 18]
        """
        super(ASPP, self).__init__()
        modules = []
        # 1x1卷积分支
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        # 使用不同膨胀率的 3x3 空洞卷积分支
        for rate in atrous_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3,
                          padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        self.convs = nn.ModuleList(modules)
        # 全局平均池化分支
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            # nn.InstanceNorm2d(out_channels, affine=True),
            nn.GroupNorm(num_groups=32, num_channels=out_channels),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 拼接后用 1x1 卷积融合
        total_channels = len(modules) * out_channels + out_channels
        self.project = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        # 计算各分支输出
        for conv in self.convs:
            res.append(conv(x))
        # 全局平均池化分支，并上采样到输入尺寸
        global_feature = self.global_avg_pool(x)
        global_feature = F.interpolate(global_feature, size=x.size()[2:], mode='bilinear', align_corners=True)
        res.append(global_feature)
        # 拼接所有分支，并融合
        x = torch.cat(res, dim=1)
        return self.project(x)

# -------------------------------
# DeepLabV3+ 模型
# -------------------------------
class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes):
        """
        Args:
            num_classes: 分割类别数（例如 21、19 等）
        """
        super(DeepLabV3Plus, self).__init__()
        self.num_classes = num_classes

        # 使用预训练的 ResNet50 作为编码器，提取其中部分层作为不同尺度的特征
        resnet = models.resnet50(pretrained=True)

        # layer0：conv1 + bn1 + relu + maxpool（下采样因子为4）
        self.layer0 = nn.Sequential(
            resnet.conv1,  # 7x7 卷积，步长为2
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        # layer1: 输出低级特征，通道数为 256（通常用于 decoder）
        self.layer1 = resnet.layer1
        # layer2、layer3、layer4 为逐步下采样，layer4 的输出通道数为 2048（用于 ASPP 模块）
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # ASPP 模块（输入为 layer4 的特征）
        self.aspp = ASPP(in_channels=2048, out_channels=256, atrous_rates=[6, 12, 18])

        # 对低级特征做降维处理（低级特征来自 layer1，通道数为 256，降到 48）
        self.low_level_proj = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # Decoder 模块：将上采样的 ASPP 特征与低级特征拼接后进一步卷积处理
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # 分类器：将 decoder 输出映射到类别数
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)
        self.fea_contrast = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        self.pred_contrast = nn.Sequential(nn.Conv2d(256, num_classes, 1))

    def forward(self, x, need_fp=False):
        input_size = x.size()[2:]  # 原始输入的空间尺寸

        # -------------------------------
        # 编码器前向传播，提取多尺度特征
        # -------------------------------
        x = self.layer0(x)              # 下采样因子 4
        low_level_features = self.layer1(x)  # 低级特征，尺寸较大（例如 1/4 输入尺寸），通道数为 256
        x = self.layer2(low_level_features)  # 下采样因子进一步加倍
        x = self.layer3(x)
        encoder_features = self.layer4(x)    # 高级特征，尺寸较小，通道数为 2048
        # 此处将 encoder_features 作为额外的输出返回
        fea_con = self.fea_contrast(encoder_features)
        pre_con = self.pred_contrast(fea_con)
        # -------------------------------
        # ASPP 模块
        # -------------------------------
        x_aspp = self.aspp(encoder_features)
        # 上采样 ASPP 输出到与低级特征相同的空间尺寸
        x_aspp = F.interpolate(x_aspp, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)

        # -------------------------------
        # Decoder 部分
        # -------------------------------
        low_level_features = self.low_level_proj(low_level_features)
        x_cat = torch.cat([x_aspp, low_level_features], dim=1)
        x_decoder = self.decoder(x_cat)
        out = self.classifier(x_decoder)
        # 将分割结果上采样到原始输入尺寸
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)

        # 返回 (分割预测, 编码器高层特征)
        return out, fea_con, pre_con

# -------------------------------
# 测试代码
# -------------------------------
if __name__ == '__main__':
    # 假设输入为 1 张 RGB 图像，尺寸为 512x512
    model = DeepLabV3Plus(num_classes=21)
    model.eval()
    input_tensor = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        seg_pred, encoder_feat = model(input_tensor)
    print("分割输出尺寸：", seg_pred.shape)         # 应为 [1, num_classes, 512, 512]
    print("编码器特征尺寸：", encoder_feat.shape)     # 一般为 [1, 2048, H, W]，H、W 取决于 backbone 下采样倍数
