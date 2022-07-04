import os

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix

from KneesOA.config import DataConfig


def eval_net(model, test_loader, device, logger, it):
    model.eval()
    targets_all = []
    predicts_all = []
    for i, (images, targets) in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            images = images.to(device)
            targets = torch.argmax(targets, dim=1)
            outputs = torch.argmax(model(images), dim=1)
            targets_all.extend(targets)
            predicts_all.extend(outputs.cpu().numpy())
    cm = confusion_matrix(targets_all, predicts_all, labels=[0, 1, 2, 3, 4])
    logger.report_matrix(
        "confusion",
        "ignored",
        iteration=it,
        matrix=cm,
        xaxis="predicted",
        yaxis="targets",
    )

    balanced_accuracy = balanced_accuracy_score(targets_all, predicts_all)
    return balanced_accuracy


def train_net(model, train_loader, val_loader, optimizer, loss, lr_scheduler, writer, device, args, logger):
    step = 0
    best_metric = -1
    writer_losses = []
    for epoch in range(args.epochs):
        model.train()
        for i, (images, targets) in enumerate(tqdm(train_loader)):
            model.train()
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            l = loss(outputs, targets)
            l.backward()
            optimizer.step()
            step += 1
            writer_losses.append(l.detach().cpu().numpy())
            if step % args.log_freq == 0:
                writer.add_scalar("Train_loss", np.mean(writer_losses), step)
                writer_losses.clear()

        state = {
            "net": model.__class__.__name__,
            "net_state": model.state_dict(),
        }

        eval_metric = eval_net(model, val_loader, device, logger, epoch)
        if eval_metric > best_metric:
            best_metric = eval_metric
            state["best_metric"] = eval_metric
            torch.save(
                state,
                os.path.join(DataConfig.models_dir, args.task_name, f"model_best.pkl")
            )
            writer.add_text("CNN", f"Model is saved on val iteration {epoch}", 0)

        writer.add_scalar("EVAL_Balanced_accuracy", eval_metric, epoch)
        reduced = lr_scheduler.step(eval_metric)
        writer.add_scalar(
            "Learning rate",
            optimizer.param_groups[0]["lr"],
            global_step=epoch,
        )
        if reduced:
            model_ = torch.load(os.path.join(DataConfig.models_dir, args.task_name, f"model_best.pkl"))
            model.load_state_dict(model_["net_state"])
