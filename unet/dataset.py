from torch.utils.data import Dataset


class MedicalDataset(Dataset):
    def __init__(self, path: str, split="train", transform=None):
        super().__init__()

    def __len__(self):
        pass

    def __getitem__(self):
        pass
