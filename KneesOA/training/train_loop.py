import os

import numpy as np
import torch
from tqdm import tqdm


def eval_net(model, test_loader, device):
    model.eval()
    targets_all = []
    predicts_all = []
    for i, (images, targets) in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            images = torch.stack(images).to(device)
            targets = torch.stack(targets).float().numpy().flatten()
            outputs = torch.sigmoid(model(backbone(images)))
            targets_all.extend(targets)
            predicts_all.extend(outputs.cpu().numpy().flatten())

    auc = roc_auc_score(targets_all, predicts_all)
    tnrs = [0]
    for i in range(100):
        k = i / 100
        predicts = np.array(copy.deepcopy(predicts_all))
        predicts[predicts >= k] = 1
        predicts[predicts < k] = 0
        tn, fp, fn, tp = confusion_matrix(np.array(targets_all), predicts).ravel()
        npv = tn / (tn + fn)
        if npv >= args.npv:
            tnrs.append(round(tn / (tn + fp) * 100, 2))

    return auc, max(tnrs)


def train_net(model, train_loader, val_loader, optimizer, loss, lr_scheduler, writer, device):
    step = 0
    best_metric = -1
    writer_losses = []
    for epoch in range(50):

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
            print(l)
            step += 1
            writer_losses.append(l.detach().cpu().numpy())
            if step % 50 == 0:
                writer.add_scalar("Train_loss", np.mean(writer_losses), step)
                writer_losses.clear()

        state = {
            "net": model.__class__.__name__,
            "net_state": model.state_dict(),
        }

        eval_metric, tnr = eval_net(model, val_loader, device)
        if eval_metric > best_metric:
            best_metric = eval_metric
            torch.save(
                state,
                os.path.join(model_save_path, f"model_best.pkl")
            )
            writer.add_text("CNN", f"Model is saved on val iteration {epoch}", 0)

        writer.add_scalar("EVAL_AUC", eval_metric, epoch)
        writer.add_scalar(f"EVAL_TNR_X_NPV:{args.npv}", tnr, epoch)
        reduced = lr_scheduler.step(eval_metric)
        writer.add_scalar(
            "Learning rate",
            optimizer.param_groups[0]["lr"],
            global_step=epoch,
        )
        if reduced:
            model_ = torch.load(os.path.join(model_save_path, f"model_best.pkl"))
            model.load_state_dict(model_["net_state"])
