import torch
from torch.utils.data import DataLoader
from pytorch_metric_learning import samplers

from KneesOA.data.dataset import KneeOADataset
from KneesOA.model.utils import seed_worker


def create_loaders(args):
    train_dataset = KneeOADataset(args.train_file)
    test_dataset = KneeOADataset(args.test_file)
    val_dataset = KneeOADataset(args.val_file)

    train_loader = DataLoader(
        train_dataset,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        batch_size=args.batch_size,
        generator=torch.default_generator,
        sampler=samplers.MPerClassSampler(train_dataset.targets, 3, None, len(train_dataset))
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=torch.default_generator,
        sampler=samplers.MPerClassSampler(test_dataset.targets, 3, None, len(test_dataset))
    )

    val_loader = DataLoader(
        val_dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        worker_init_fn=seed_worker,
        generator=torch.default_generator,
        sampler=samplers.MPerClassSampler(val_dataset.targets, 3, None, len(val_dataset))
    )

    return train_loader, test_loader, val_loader
