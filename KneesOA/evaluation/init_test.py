import argparse
from KneesOA.config import DataConfig
from KneesOA.evaluation.test_model import _test_model


def create_parser():
    parser = argparse.ArgumentParser(description="Test Network")

    # Arguments for stats and models saving
    parser.add_argument("--model_weights", type=str, help="path to model_weights")
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
        "--test_file",
        type=str,
        help="path to a test set markup file",
    )
    parser.add_argument("--random_state", type=int, default=17, help="random state for random generators")
    parser.add_argument(
        "--reopen_if_saved",
        action="store_true",
        help="whether to save preprocessed files during first epoch and read them on subsequent epochs",
    )

    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--num_workers", type=int, default=1, help="num workers for data loader")

    return parser


if __name__ == "__main__":
    args = create_parser().parse_args()
    _test_model(args)
