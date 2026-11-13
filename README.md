# Masked ViTFL

Core PyTorch implementation of the paper's masked ViT model for federated image classification.

[中文说明](README.zh-CN.md)

## Method

A ViT-S/16 encoder extracts the final CLS feature. A learnable mask initialized to ones reweights it before the linear classifier.

    Loss = CrossEntropy + 1e-2 * L1(mask) + 5e-4 * L2(other trainable weights)^2

Clients train locally. The server applies sample-size-weighted FedAvg to all model parameters, including the mask.

## Install

Requires Python 3.8+. Install PyTorch for your CPU/CUDA environment, then run:

    python -m pip install -e .

## Use

    import torch
    from infosci_mvitfl import MaskedViTFL, MaskedViTFLoss

    model = MaskedViTFL(num_classes=7, pretrained=True)
    criterion = MaskedViTFLoss()
    images = torch.randn(2, 3, 224, 224)
    labels = torch.randint(0, 7, (2,))
    loss = criterion(model(images), labels, model)

A synthetic-data example is in examples/minimal_usage.py. The paper uses Adam at 1e-4, one local epoch per round, and 50 rounds.

## License

Add a license file before publishing.
