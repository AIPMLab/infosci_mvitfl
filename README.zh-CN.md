# Masked ViTFL

论文中联邦图像分类方法的 PyTorch 核心实现。

[English README](README.md)

## 方法

ViT-S/16 编码器提取最终 CLS 特征。初始化为全 1 的可学习掩码逐维加权特征，再由线性分类器输出结果。

    损失 = 交叉熵 + 1e-2 × 掩码 L1 + 5e-4 × 其他可训练参数 L2 平方

客户端本地训练后，服务器按训练样本数加权聚合所有模型参数，包括掩码。

## 安装

需要 Python 3.8 以上版本。先安装适合 CPU/CUDA 环境的 PyTorch，再运行：

    python -m pip install -e .

## 使用

    import torch
    from infosci_mvitfl import MaskedViTFL, MaskedViTFLoss

    model = MaskedViTFL(num_classes=7, pretrained=True)
    criterion = MaskedViTFLoss()
    images = torch.randn(2, 3, 224, 224)
    labels = torch.randint(0, 7, (2,))
    loss = criterion(model(images), labels, model)

合成数据示例位于 examples/minimal_usage.py。论文配置为 Adam 学习率 1e-4、每轮一个本地 epoch、共 50 轮。

## 许可证

上传前添加许可证文件。
