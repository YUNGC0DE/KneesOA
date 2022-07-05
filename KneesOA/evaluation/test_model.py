import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_score, recall_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from clearml import Task
from tqdm import tqdm

from KneesOA.data.dataset import KneeOADataset
from KneesOA.model.utils import load_network, fix_seeds


def get_metrics(model, test_loader, device, logger, writer, it=1):
    model.eval()
    targets_all = []
    predicts_all = []
    for i, (images, targets) in enumerate(tqdm(test_loader)):
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
    test_dataset = KneeOADataset(args.test_file, test=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    net = load_network(args.model_weights)
    net.to(device)
    net.eval()
    get_metrics(net, test_loader, device, logger, writer)
