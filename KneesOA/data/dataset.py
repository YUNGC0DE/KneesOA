import numpy as np
import pandas as pd
import torch
from PIL import Image

from KneesOA.config import transform, augmentation
from torch.utils.data import Dataset


class KneeOADataset(Dataset):
    def __init__(self, split_path: str, test: bool = False, return_orig_image: bool = False):
        super().__init__()
        self.dataframe = pd.read_csv(split_path)
        self.targets = self.dataframe.iloc[:, -1].to_list()
        self.test = test
        self.return_orig_image = return_orig_image

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx, :]
        image_path, target = row.values
        image = transform(Image.open(image_path))
        if not self.test:
            image = augmentation(image)

        target_one_hot = torch.zeros(5)
        target_one_hot[target] = 1

        if self.return_orig_image:
            return image, target_one_hot, np.array(Image.open(image_path)) / 255

        return image, target_one_hot

    def __len__(self):
        return len(self.dataframe)
