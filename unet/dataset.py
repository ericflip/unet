import os

from PIL import Image
from torch.utils.data import Dataset


class MedicalDataset(Dataset):
    def __init__(self, path: str, split="train", transform=None, target_transform=None):
        super().__init__()

        assert split in ["train", "test"]

        self.transform = transform
        self.target_transform = target_transform

        if split == "train":
            self.root_path = os.path.join(path, "train")

            images_path = os.path.join(self.root_path, "image")
            labels_path = os.path.join(self.root_path, "label")
            augs_path = os.path.join(self.root_path, "aug")

            data = []

            for item in os.listdir(images_path):
                image_path = os.path.join(images_path, item)
                label_path = os.path.join(labels_path, item)
                data.append((image_path, label_path))

            for item in os.listdir(augs_path):
                if item.startswith("image"):
                    id, _ = os.path.splitext(item[6:])
                    image_path = os.path.join(augs_path, item)
                    label_path = os.path.join(augs_path, f"mask_{id}.png")
                    data.append((image_path, label_path))

            self.data = data

        else:
            self.root_path = os.path.join(path, "test")

            data = []
            for item in os.listdir(self.root_path):
                if "_" in item:
                    label_path = os.path.join(self.root_path, item)
                    id = item.split("_")[0]
                    image_path = os.path.join(self.root_path, f"{id}.png")
                    data.append((image_path, label_path))

            self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        image_path, label_path = self.data[index]

        # open in black and white mode
        image = Image.open(image_path)
        image = image.convert("L")

        label = Image.open(label_path)
        label = label.convert("L")

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label
