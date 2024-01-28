import torch
import torch.nn.functional as F
from inference import inference
from model import UNet
from torch.utils.data import DataLoader


def pixel_accuracy(unet: UNet, loader: DataLoader, device="cpu"):
    unet.eval()

    total_correct = 0
    total_pixels = 0

    unet.to(device)

    with torch.no_grad():
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device).int()
            pred = inference(unet, X)
            num_correct = (pred == Y).sum()

            B, _, H, W = Y.shape
            num_pixels = B * H * W

            total_correct += num_correct.item()
            total_pixels += num_pixels

    accuracy = total_correct / total_pixels
    return accuracy


def iou(unet: UNet, loader: DataLoader, device="cpu"):
    unet.eval()
    ious = 0

    with torch.no_grad():
        for (
            X,
            Y,
        ) in loader:
            X, Y = X.to(device), Y.to(device).long()
            pred = inference(unet, X)

            intersection = X[pred == Y].sum()
            union = pred.sum() + Y.sum() - intersection

            iou = intersection / union
            ious += iou

    iou = ious / len(loader)
    return iou.item()


def dice_score(unet: UNet, loader: DataLoader, device="cpu"):
    unet.eval()

    total_dice = 0

    with torch.no_grad():
        for (
            X,
            Y,
        ) in loader:
            X, Y = X.to(device), Y.to(device).long()
            pred = inference(unet, X)

            dice = (2 * X[pred == Y].sum()) / (pred.sum() + Y.sum())
            total_dice += dice

    dice = total_dice / len(loader)
    return dice.item()
