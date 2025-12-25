import torch
import torch.nn.functional as F

def dice_loss(preds, targets, smooth=1e-6):
    preds = preds.contiguous().view(-1)
    targets = targets.contiguous().view(-1)

    preds = preds.float()
    targets = targets.float()

    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()

    dice = (2. * intersection + smooth) / (union + smooth)

    return 1 - dice