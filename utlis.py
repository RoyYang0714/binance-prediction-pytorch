"""Utlis."""
import argparse
import random

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="FinTech Final")

    # Data
    parser.add_argument("--data_dir", metavar="DIR", help="Path to dataset")
    parser.add_argument("--ticker", type=str, help="Binance name")
    parser.add_argument("--inter", type=str, help="time intervals")
    parser.add_argument("--ema", type=str, default="EMA150", help="EMA for label")
    parser.add_argument(
        "--shift_time", type=int, default=30, help="time shift for label"
    )

    # Misc
    parser.add_argument(
        "--seed", type=int, default=42, metavar="S", help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--epochs", type=int, default=150, metavar="S", help="Training epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-2, metavar="S", help="Training batch size"
    )
    parser.add_argument(
        "--bs_trn", type=int, default=1024, metavar="S", help="Training batch size"
    )
    parser.add_argument(
        "--ckpt", type=str, default=None, metavar="S", help="Model weights path",
    )
    parser.add_argument(
        "--eval", action="store_true", help="Whether do the evaluation",
    )

    # Agent
    parser.add_argument(
        "--init_money", type=int, default=10000, metavar="S", help="Initial money"
    )
    parser.add_argument(
        "--sell_rate", type=float, default=0.015, metavar="S", help="selling point"
    )
    parser.add_argument(
        "--fee", type=float, default=0.001, metavar="S", help="trading fee"
    )
    parser.add_argument(
        "--thres", type=int, default=10, metavar="S", help="mini buy in price"
    )

    return parser.parse_args()


def set_random_seed(seed, deterministic=False):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class AverageMeter(object):
    """Compute Average."""

    def __init__(self):
        """Init."""
        self.reset()

    def reset(self):
        """Reset."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update"""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_checkpoint(model, ckpt_path):
    """Load model weights."""
    checkpoint = torch.load(f"{ckpt_path}", map_location=torch.device("cpu"))

    model.load_state_dict(checkpoint["state_dict"])
    return model


def save_checkpoint(model, optim, epoch, file_name):
    """Save checkpoint."""
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optim.state_dict(),
        },
        f"{file_name}",
    )
