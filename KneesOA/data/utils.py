import torch
from torch.utils.data import DataLoader

from KneesOA.data.dataset import KneeOADataset
from KneesOA.model.utils import seed_worker


def create_loaders(args):
    train_dataset = KneeOADataset(args.train_file)
    test_dataset = KneeOADataset(args.test_file)
    val_dataset = KneeOADataset(args.val_file)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=torch.default_generator,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=torch.default_generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=torch.default_generator,
    )

    return train_loader, test_loader, val_loader
