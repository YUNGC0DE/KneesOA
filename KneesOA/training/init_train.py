import argparse
from KneesOA.config import DataConfig
from KneesOA.training.train_model import train_model


def create_parser():
    parser = argparse.ArgumentParser(description="Train Network")

    # Arguments for stats and models saving
    parser.add_argument("--project_name", type=str, default="OAKnees", help="project name for ClearML server")
    parser.add_argument("--task_name", type=str, help="experiment name for ClearML server")
    parser.add_argument(
        "--output_uri",
        type=str,
        default=".",
        help="AWS path for saving models; NO AWS ANYMORE :(",
    )
    parser.add_argument("--writer", type=str, default="runs", help="path to the dir where runs gonna be saved")

    # Arguments for data loading and preparation
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DataConfig.data_dir,
        help="path to a folder with data",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=DataConfig.models_dir,
        help="path to a folder with models",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        help="path to a train set markup file",
    )
    parser.add_argument(
        "--val_file",
        type=str,
        help="path to a val set markup file",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        help="path to a test set markup file",
    )
    parser.add_argument(
        "--reopen_if_saved",
        action="store_true",
        help="whether to save preprocessed files during first epoch and read them on subsequent epochs",
    )
    parser.add_argument("--random_state", type=int, default=17, help="random state for random generators")

    # training params
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=1,
        help="number of gradient accumulation steps",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--num_workers", type=int, default=1, help="num workers for data loader")
    parser.add_argument("--epochs", type=int, default=300, help="number of epochs")
    parser.add_argument("--log_freq", type=int, default=50, help="frequency of logging training loss")
    parser.add_argument("--lr", type=float, default=0.001, help="optimizer learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="L2 regularization")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum for SGD optimizer")
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="patience for RediceLROnPlateau LR-scheduler",
    )
    parser.add_argument(
        "--gamma_factor",
        type=float,
        default=0.5,
        help="gamma factor for RediceLROnPlateau LR-scheduler",
    )

    return parser


if __name__ == "__main__":
    args = create_parser().parse_args()
    train_model(args)
