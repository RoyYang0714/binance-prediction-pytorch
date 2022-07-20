"""Prediction agent."""
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import build_dataset, predictors_list
from model import Buy_Model
from utlis import (
    AverageMeter,
    load_checkpoint,
    parse_args,
    save_checkpoint,
    set_random_seed,
)
from vis import plot_prediction


def train(args, model, train_set, train_loader, writer):
    """Train the model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    train_main_loss = AverageMeter()

    model.train()
    print("=" * 50)
    print("Start training...")
    for epoch in range(1, args.epochs + 1):
        print("Epoch #", epoch)
        for i, data in enumerate(train_loader):
            curr_iter = i + (epoch - 1) * len(train_set)
            x, y = data
            x, y = x.to(args.device), y.to(args.device)

            y_pred = model(x)

            loss = loss_fn(y_pred, y)
            train_main_loss.update(loss.item())

            acc = accuracy_score(
                y.detach().cpu().numpy(),
                np.argmax(y_pred.detach().cpu().numpy(), axis=1),
            )

            # tensorboard
            writer.add_scalar("Loss/train", loss.item(), curr_iter)
            writer.add_scalar("Accuracy/train", acc, curr_iter)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"The loss calculated: {train_main_loss.avg:0.6f}")

        save_checkpoint(model, optimizer, epoch, args.ckpt)


def predict(args, df, model):
    """Run on unseen data."""
    df["Buy"] = df["target_cls"]
    df["Sell"] = df["target_cls"]
    asset = {"money": args.init_money, "coins": 0}
    num_trades = 0
    num_buy = 0
    num_sell = 0

    buy_time = 0
    for i in tqdm(range(len(df))):
        X_cls_valid = [[df[p][i] for p in predictors_list]]

        x_test = torch.Tensor(X_cls_valid).type(torch.Tensor).to(args.device)
        prediction = np.argmax(model(x_test).detach().cpu().numpy(), axis=1)
        sell = 0

        # Agent
        if (
            i == buy_time + args.shift_time - 1
            and asset["coins"] != 0
        ):
            asset["money"] += asset["coins"] * df.Close[i] * (1 - args.fee)
            asset["coins"] = 0
            prediction = 0
            sell = 1
            num_sell += 1
        elif prediction:
            if asset["money"] > args.thres * (1 + args.fee):
                buy_in = asset["money"] / (df.Close[i] * (1 + args.fee))
                asset["coins"] += buy_in
                asset["money"] = 0
                num_buy += 1
                buy_time = i
            else:
                prediction = 0

        df.loc[i, "Buy"] = prediction
        df.loc[i, "Sell"] = sell

    asset["money"] += asset["coins"] * df.Close[i] * (1 - args.fee)
    asset["coins"] = 0
    num_trades = num_buy + num_sell + 1

    print(f"ROI: {((asset['money'] - args.init_money) / args.init_money)*100:.2f}")
    print(f"Initial Capital: {args.init_money}")
    print(f"Final Capital: {asset['money']}")
    print(f"Numbers of trades: {num_trades}")
    return df


def main():
    """Main."""
    args = parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ticker = f"{args.ticker}_{args.inter}"

    if args.ckpt is None:
        args.ckpt = f"checkpoints/{ticker}.ckpt"

    torch.backends.cudnn.benchmark = True
    set_random_seed(args.seed)

    test_start_time_str = "01/01/21 00:00:00"
    test_end_time_str = "01/12/21 00:00:00"
    test_start_time = datetime.strptime(test_start_time_str, "%d/%m/%y %H:%M:%S")
    test_end_time = datetime.strptime(test_end_time_str, "%d/%m/%y %H:%M:%S")

    # Build dataset
    train_set, train_loader, test_df, x_test, y_test = build_dataset(
        args, ticker, test_start_time, test_end_time
    )

    model = Buy_Model(len(predictors_list))
    model = model.to(args.device)

    if args.eval:
        model = load_checkpoint(model, args.ckpt)
    else:
        writer = SummaryWriter(log_dir=f"./log_dir/{ticker}")
        train(args, model, train_set, train_loader, writer)

    x_test = torch.from_numpy(x_test).type(torch.Tensor).to(args.device)

    model.eval()
    pred = model(x_test)

    acc = accuracy_score(y_test, np.argmax(pred.detach().cpu().numpy(), axis=1))

    print("=" * 50)
    print(f"Test on the unseen data: {test_start_time} to {test_end_time}")

    test_df = predict(args, test_df, model)
    plot_prediction(test_df, ticker)
    print(f"The testing accuracy is {acc * 100:.2f} %")
    print("=" * 50)


if __name__ == "__main__":
    main()
