import torch
import numpy as np
from sklearn.metrics import f1_score


def compute_f1_score(model, dataloader, device):
    """
    计算语义分割任务的 F1 分数。

    Args:
        model (torch.nn.Module): 训练好的模型。
        dataloader (torch.utils.data.DataLoader): 测试集 DataLoader。
        device (torch.device): 运行设备。

    Returns:
        float: 测试集的 F1 分数。
    """
    model.eval()  # 设置为评估模式
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in dataloader:
            # 将数据加载到设备
            images = images.to(device)
            labels = labels.cpu().numpy()  # 将标签移动到 CPU

            # 模型预测
            outputs = model(images)

            # 如果是概率图，取 Sigmoid 激活后将其二值化
            if outputs.shape[1] == 1:  # 单通道二分类
                preds = (torch.sigmoid(outputs).cpu().numpy() > 0.5).astype(np.uint8)
            else:  # 两通道，取 argmax
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

            # 展平标签和预测，添加到列表
            y_true.extend(labels.flatten())
            y_pred.extend(preds.flatten())

    # 计算 F1 分数
    f1 = f1_score(y_true, y_pred, average="binary")
    return f1


# 示例用法
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torchvision.transforms import ToTensor
    from torchvision.datasets import FakeData

    # 示例：创建一个伪造数据集和模型
    dataset = FakeData(size=100, image_size=(3, 256, 256), num_classes=2, transform=ToTensor())
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)


    # 模型假设
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 2, kernel_size=3, padding=1)

        def forward(self, x):
            return self.conv(x)


    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DummyModel().to(device)

    # 计算 F1 分数
    f1 = compute_f1_score(model, dataloader, device)
    print(f"F1 Score: {f1:.4f}")