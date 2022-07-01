import os
import random
from typing import Optional

import numpy as np
import torch

from KneesOA.model.backbone import ResNet, Bottleneck


def load_network(model_weights: Optional[str] = None):
    """
    Load network
    :param model_weights: Path to model weights
    :return: model instance
    """
    resnest50 = ResNet(Bottleneck, [3, 4, 6, 3], radix=2, groups=1,
                       bottleneck_width=64, stem_width=32, avg_down=True,
                       avd=True, avd_first=False)

    if model_weights is not None:
        model_dict = torch.load(model_weights)["net_state"]
        resnest50.load_state_dict(model_dict)

    return resnest50


def fix_seeds(random_state: int = 17) -> None:
    """
    FIX model random sid
    :param random_state:
    :return: None
    """
    random.seed(random_state)
    os.environ["PYTHONHASHSEED"] = str(random_state)
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    torch.cuda.manual_seed(random_state)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(_):
    # https://pytorch.org/docs/stable/data.html#data-loading-randomness
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
