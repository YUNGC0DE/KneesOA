import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_score, recall_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from clearml import Task
from tqdm import tqdm
from pytorch_grad_cam import ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from KneesOA.data.dataset import KneeOADataset
from KneesOA.model.utils import load_network, fix_seeds


def save_grad_cam(model, test_loader, device):
    target_layers = [model.layer4[-1]]
    cam = ScoreCAM(model=model, target_layers=target_layers, use_cuda=True)
    images, targets, original_images = next(test_loader.__iter__())
    images.to(device)
    targets = torch.argmax(targets, dim=1)
    grayscale_cam = cam(input_tensor=images, targets=None)
    images_rgb = torch.stack([original_images, original_images, original_images], dim=1).permute(0, 2, 3, 1)
    for idx, image in enumerate(images_rgb):
        visualization = show_cam_on_image(np.array(image), grayscale_cam[idx], use_rgb=True)
        plt.imsave(f"docs/images/Grad_Cam_{int(targets[idx])}.png", visualization)


def get_metrics(model, test_loader, device, logger, writer, it=1):
    model.eval()
    targets_all = []
    predicts_all = []
    for i, (images, targets, _) in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            images = images.to(device)
            targets = torch.argmax(targets, dim=1)
            outputs = model(images)
            outputs = torch.argmax(outputs, dim=1)
            targets_all.extend(targets)
            predicts_all.extend(outputs.cpu().numpy())
    cm = confusion_matrix(targets_all, predicts_all, labels=[0, 1, 2, 3, 4])
    precision = precision_score(targets_all, predicts_all, average=None)
    recall = recall_score(targets_all, predicts_all, average=None)

    for i in range(5):
        writer.add_scalar(f"Precision/{i}", precision[i], it)
        writer.add_scalar(f"Recall/{i}", recall[i], it)

    logger.report_matrix(
        "confusion",
        "ignored",
        iteration=it,
        matrix=cm,
        xaxis="predicted",
        yaxis="targets",
    )

    balanced_accuracy = balanced_accuracy_score(targets_all, predicts_all)
    writer.add_scalar("TEST Balanced Accuracy", balanced_accuracy, it)
    return balanced_accuracy


def _test_model(args):
    logger = None
    if args.writer == "runs":
        task = Task.init(
            project_name=args.project_name,
            task_name=args.task_name,
            output_uri=None,
        )
        task.set_initial_iteration(0)
        logger = task.get_logger().current_logger()

    writer = SummaryWriter(args.writer)
    fix_seeds(args.random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = KneeOADataset(args.test_file, test=True, return_orig_image=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    net = load_network(args.model_weights)
    net.to(device)
    net.eval()
    get_metrics(net, test_loader, device, logger, writer)
    save_grad_cam(net, test_loader, device)
