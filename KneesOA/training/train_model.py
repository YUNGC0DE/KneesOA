import os

import torch
from timm.optim import RAdam
from torch.utils.tensorboard import SummaryWriter
from clearml import Task

from KneesOA.config import DataConfig
from KneesOA.data.scheduler import CustomScheduler
from KneesOA.data.utils import create_loaders
from KneesOA.model.utils import load_network, fix_seeds
from KneesOA.training.train_loop import train_net, eval_net


def train_model(args):
    if args.writer == "runs":
        task = Task.init(
            project_name=args.project_name,
            task_name=args.task_name,
            output_uri=None,
        )
        task.set_initial_iteration(0)

    writer = SummaryWriter(args.writer)
    fix_seeds(args.random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, val_loader = create_loaders(args)
    net = load_network()
    net.to(device)
    os.makedirs(os.path.join(DataConfig.models_dir, args.task_name), exist_ok=True)

    loss = torch.nn.CrossEntropyLoss(weight=torch.tensor([1, 1, 1, 1.2, 1.6], device=device))
    optimizer = RAdam(net.parameters(), lr=args.lr)
    lr_scheduler = CustomScheduler(optimizer, mode="max", factor=args.gamma_factor, patience=args.patience)
    train_net(net, train_loader, val_loader, optimizer, loss, lr_scheduler, writer, device, args)
    balanced_accuracy = eval_net(net, test_loader, device)
    writer.add_scalar("TEST_Balanced_accuracy", balanced_accuracy, 0)
