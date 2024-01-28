import argparse
import math
import os

import evals
import torch
from dataset import MedicalDataset
from model import UNet
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the UNet on images and target masks"
    )
    parser.add_argument(
        "--data_dir", default="data", help="Directory containing the dataset"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--save_dir", default="checkpoints", help="Save directory")

    args = parser.parse_args()
    return args


def eval_loader(unet, loader, device="cpu"):
    accuracy = evals.pixel_accuracy(unet, loader, device=device)
    iou = evals.iou(unet, loader, device=device)
    dice = evals.dice_score(unet, loader, device=device)

    return dict(accuracy=accuracy, iou=iou, dice=dice)


if __name__ == "__main__":
    args = parse_args()

    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # prepare dataset and dataloader
    transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(size=(512, 512), antialias=True),
        ]
    )
    train_dataset = MedicalDataset(
        path=args.data_dir,
        split="train",
        transform=transforms,
        target_transform=transforms,
    )
    test_dataset = MedicalDataset(
        path=args.data_dir,
        split="test",
        transform=transforms,
        target_transform=transforms,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # prepare model and optimizer
    unet = UNet(image_channels=1, n_channels=64, ch_mults=[128, 256, 512, 1024]).to(
        device
    )
    optimizer = torch.optim.Adam(unet.parameters(), lr=args.lr)

    # num steps
    num_steps = args.epochs * (math.ceil(len(train_loader) / args.batch_size))
    pbar = tqdm(total=num_steps, position=0)

    # create checkpoint dir
    os.makedirs(args.save_dir, exist_ok=True)

    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            pred = unet(X)
            target = Y.squeeze(1).long()
            loss = criterion(pred, target)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item())
            pbar.update(1)

        train_evals = eval_loader(unet, train_loader, device=device)
        test_evals = eval_loader(unet, test_loader, device=device)

        tqdm.write(f"==Epoch {epoch + 1}==")
        tqdm.write("Train Evals:")
        tqdm.write(str(train_evals))
        tqdm.write("Test Evals:")
        tqdm.write(str(test_evals))

        # checkpoint
        torch.save(
            unet.state_dict(), os.path.join(args.save_dir, f"checkpoint-{epoch}.pt")
        )
