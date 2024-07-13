import torch
from torch.nn import functional as F
import numpy as np

def pixel_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)
        # Mask out ignore index (-1)
        valid = mask >= 0
        correct = torch.eq(output[valid], mask[valid]).int()
        accuracy = float(correct.sum()) / float(valid.sum())
    return accuracy


def mIoU(pred_mask, mask, smooth=1e-10, n_classes=16):
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        # Ignore index for invalid labels
        valid = mask >= 0
        pred_mask = pred_mask[valid]
        mask = mask[valid]

        iou_per_class = []
        for clas in range(0, n_classes):  # loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0:  # no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union + smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)
