import argparse
import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from model import UNet
from PIL import Image
from torchvision.transforms import v2


def inference(unet: UNet, X: torch.Tensor):
    pred = unet(X)

    # apply softmax across dim=1 to calculate probabilities
    scores = F.softmax(pred, dim=1)

    # get class with highest probability
    _, classes = scores.max(dim=1)

    return classes


def parse_args():
    parser = argparse.ArgumentParser(description="Inference on a single image")
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--output", type=str, required=True, help="Path to output")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load image
    image = Image.open(args.image)
    image = image.convert("L")

    # convert to tensor
    transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(size=(512, 512), antialias=True),
        ]
    )

    image = transforms(image).unsqueeze(0).to(device)

    # load model
    unet = UNet(image_channels=1, n_channels=64, ch_mults=[128, 256, 512, 1024]).to(
        device
    )

    unet.load_state_dict(torch.load(args.model))

    # inference
    classes = inference(unet, image)

    # save output
    directory = os.path.dirname(args.output)
    os.makedirs(directory, exist_ok=True)

    # display image next to prediction
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image.squeeze().cpu().numpy(), cmap="gray")
    ax[1].imshow(classes.squeeze().cpu().numpy(), cmap="gray")

    plt.savefig(args.output)
