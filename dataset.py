"""Dataset for Final."""
import os
from datetime import datetime

import numpy as np
import pandas as pd
import pandas_ta as ta
import torch
from torch.utils.data import DataLoader, Dataset

from vis import plot_price

predictors_list = [
    "aboveSAR",
    "aboveUpperBB",
    "belowLowerBB",
    "normRSI",
    "oversoldRSI",
    "overboughtRSI",
    "aboveEMA5",
    "aboveEMA10",
    "aboveEMA15",
    "aboveEMA20",
    "aboveEMA30",
    "aboveEMA40",
    "aboveEMA50",
    "aboveEMA60",
    "aboveEMA70",
    "aboveEMA80",
    "aboveEMA90",
    "aboveEMA100",
]


class BinanceDataset(Dataset):
    """Binance public dataset pytorch wrapper.

    ref: https://github.com/binance/binance-public-data
    """

    def __init__(self, data, labels):
        """Init dataset."""
        self.data = torch.from_numpy(data).type(torch.Tensor)
        self.labels = torch.from_numpy(labels).type(torch.long)

    def __len__(self):
        """Dataset length."""
        return len(self.data)

    def __getitem__(self, idx):
        """Get single data."""
        return self.data[idx], self.labels[idx]


def parse_data(data_dir, ticker):
    """Load and parse data."""
    data = pd.DataFrame(columns=["Open time", "High", "Low", "Close",])
    for csv_file in sorted(os.listdir(data_dir)):
        if "csv" not in csv_file:
            continue
        df = pd.read_csv(
            f"{data_dir}/{csv_file}",
            usecols=[0, 2, 3, 4],
            names=["Open time", "High", "Low", "Close",],
        )

        tmp = df["Open time"].copy()
        for i in range(len(df.index)):
            tmp.loc[i] = datetime.fromtimestamp(df["Open time"][i] // 1000)
        df["Open time"] = tmp

        data = pd.concat([data, df], ignore_index=True)
    plot_price(ticker, data)
    return data


def compute_technical_indicators(df):
    """Compute technical indicators."""
    df["EMA5"] = ta.ema(df["Close"], length=5)
    df["EMA10"] = ta.ema(df["Close"], length=10)
    df["EMA15"] = ta.ema(df["Close"], length=15)
    df["EMA20"] = ta.ema(df["Close"], length=10)
    df["EMA30"] = ta.ema(df["Close"], length=30)
    df["EMA40"] = ta.ema(df["Close"], length=40)
    df["EMA50"] = ta.ema(df["Close"], length=50)

    df["EMA60"] = ta.ema(df["Close"], length=60)
    df["EMA70"] = ta.ema(df["Close"], length=70)
    df["EMA80"] = ta.ema(df["Close"], length=80)
    df["EMA90"] = ta.ema(df["Close"], length=90)

    df["EMA100"] = ta.ema(df["Close"], length=100)
    df["EMA150"] = ta.ema(df["Close"], length=150)
    df["EMA200"] = ta.ema(df["Close"], length=200)

    bbandsdf = ta.bbands(df["Close"], length=20)

    psardf = ta.psar(df["High"], df["Low"])
    df["SAR"] = psardf["PSARs_0.02_0.2"]

    df["upperBB"] = bbandsdf["BBU_20_2.0"]
    df["middleBB"] = bbandsdf["BBM_20_2.0"]
    df["lowerBB"] = bbandsdf["BBL_20_2.0"]

    df["RSI"] = ta.rsi(df["Close"], length=14)

    df["normRSI"] = df["RSI"] / 100.0
    return df


def compute_features(df):
    """computes features for forest decisions."""
    df["aboveEMA5"] = np.where(df["Close"] > df["EMA5"], 1, 0)
    df["aboveEMA10"] = np.where(df["Close"] > df["EMA10"], 1, 0)
    df["aboveEMA15"] = np.where(df["Close"] > df["EMA15"], 1, 0)
    df["aboveEMA20"] = np.where(df["Close"] > df["EMA20"], 1, 0)
    df["aboveEMA30"] = np.where(df["Close"] > df["EMA30"], 1, 0)
    df["aboveEMA40"] = np.where(df["Close"] > df["EMA40"], 1, 0)

    df["aboveEMA50"] = np.where(df["Close"] > df["EMA50"], 1, 0)
    df["aboveEMA60"] = np.where(df["Close"] > df["EMA60"], 1, 0)
    df["aboveEMA70"] = np.where(df["Close"] > df["EMA70"], 1, 0)
    df["aboveEMA80"] = np.where(df["Close"] > df["EMA80"], 1, 0)
    df["aboveEMA90"] = np.where(df["Close"] > df["EMA90"], 1, 0)

    df["aboveEMA100"] = np.where(df["Close"] > df["EMA100"], 1, 0)
    df["aboveEMA150"] = np.where(df["Close"] > df["EMA150"], 1, 0)
    df["aboveEMA200"] = np.where(df["Close"] > df["EMA200"], 1, 0)

    df["aboveUpperBB"] = np.where(df["Close"] > df["upperBB"], 1, 0)
    df["belowLowerBB"] = np.where(df["Close"] < df["lowerBB"], 1, 0)

    df["aboveSAR"] = np.where(df["Close"] > df["SAR"], 1, 0)

    df["oversoldRSI"] = np.where(df["RSI"] < 30, 1, 0)
    df["overboughtRSI"] = np.where(df["RSI"] > 70, 1, 0)

    # cleanup NaN values
    df = df.fillna(0).copy()

    return df


def define_target_condition(df, ema, shift_time):
    """Buy in condition."""
    df["target_cls"] = np.where(
        df["Close"].shift(-shift_time) > df[f"{ema}"].shift(-shift_time), 1, 0
    )

    # remove NaN values
    df = df.fillna(0).copy()

    return df


def extract_features(df, ema, shift_time):
    """Extracting features."""
    df = compute_technical_indicators(df)
    df = compute_features(df)
    df = define_target_condition(df, ema, shift_time)

    x = df[predictors_list].fillna(0).to_numpy()
    y = df.target_cls.fillna(0).to_numpy()
    return df, x, y


def build_dataset(args, ticker, test_start_time, test_end_time):
    """Build training and testing data."""
    df = parse_data(args.data_dir, ticker)

    test_split_num = 0
    end = len(df)
    for i in range(len(df)):
        if df["Open time"][i] >= test_start_time and test_split_num == 0:
            test_split_num = i
        elif df["Open time"][i] >= test_end_time:
            end = i
            break

    train_df = df.iloc[:test_split_num, :].copy()
    test_df = df.iloc[test_split_num:end, :].copy().reset_index()

    train_df, x_train, y_train = extract_features(train_df, args.ema, args.shift_time)
    test_df, x_test, y_test = extract_features(test_df, args.ema, args.shift_time)

    print("=" * 50)
    print(f"Total data: {len(df)}")
    print(f"Training split: {x_train.shape[0]}")
    print(f"Test split: {x_test.shape[0]}")
    print(f"Num of Features: {len(predictors_list)}")

    train_set = BinanceDataset(x_train, y_train)
    train_loader = DataLoader(
        train_set, batch_size=args.bs_trn, shuffle=True, drop_last=True
    )

    return train_set, train_loader, test_df, x_test, y_test
