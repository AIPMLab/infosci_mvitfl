import torch
from torch import nn
from torch.nn import functional as F


class MaskedViTFLoss(nn.Module):
    def __init__(self, lambda_mask=1e-2, lambda_theta=5e-4):
        super().__init__()
        self.lambda_mask = lambda_mask
        self.lambda_theta = lambda_theta

    def forward(self, logits, labels, model):
        theta_l2 = logits.new_zeros(())
        for param in model.parameters():
            if param.requires_grad and param is not model.mask:
                theta_l2 += param.square().sum()

        return (
            F.cross_entropy(logits, labels)
            + self.lambda_mask * model.mask.abs().sum()
            + self.lambda_theta * theta_l2
        )
