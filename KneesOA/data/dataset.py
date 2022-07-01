import pandas as pd
import torch
from PIL import Image

from KneesOA.config import transform, augmentation
from torch.utils.data import Dataset


class KneeOADataset(Dataset):
    def __init__(self, split_path: str):
        super().__init__()
        self.dataframe = pd.read_csv(split_path)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx, :]
        image_path, target = row.values
        image = augmentation(transform(Image.open(image_path)))
        target_one_hot = torch.zeros(5)
        target_one_hot[target] = 1

        return image, target_one_hot

    def __len__(self):
        return len(self.dataframe)
