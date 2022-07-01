from dataclasses import dataclass
import os.path as osp

import torchvision.transforms as transforms

CURRENT_PATH = osp.dirname(osp.realpath(__file__))


@dataclass
class DataConfig:
    data_dir: str = "/home/evgenii/Desktop/KneesOA/data/"
    models_dir: str = osp.join(CURRENT_PATH, '../models')
    mean: float = 0.6078
    std: float = 0.1933


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(DataConfig.mean, DataConfig.std)
])

augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
])
