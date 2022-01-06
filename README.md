# Binance Prediction Pytorch Model

## Main Results
### ETHUSDT from 2021-01-01 00:00:00 to 2021-12-01 00:00:00
|Time interval|   ROI   |
|:-----------:|:-------:|
|  1d (Human) |   2.74% |
|  1d (Model) | 125.05% |
|  4h (Human) |  36.86% |
|  4h (Model) | 300.37% |
|  1h (Human) |  37.55% |
|  1h (Model) | 393.66% |

### BTCUSDT from 2021-01-01 00:00:00 to 2021-12-01 00:00:00
|Time interval|   ROI   |
|:-----------:|:-------:|
|  1d (Human) |   3.11% |
|  1d (Model) |  30.08% |
|  4h (Human) |  18.30% |
|  4h (Model) |  30.67% |
|  1h (Human) |  19.79% |
|  1h (Model) |  32.07% |

## Getting started
### Environment
- Test OS: Ubuntu 16.04 LTS
- Python version: 3.8

### Preparation
- Create folders.
```bash
mkdir images
mkdir checkpoints
```
- Please run ``pip install –r requirements.txt`` to install the needed libraries.

## Dataset
### [Binance Public Data](https://github.com/binance/binance-public-data)
- Clone the repo.
- Follow the [instruction](https://github.com/binance/binance-public-data/tree/master/python) to download required data.
```bash
# ETHUSDT
python download-kline.py -s ETHUSDT -startDate 2017-08-01 -endDate 2021-12-01

# BTCUSDT
python download-kline.py -s BTCUSDT -startDate 2017-08-01 -endDate 2021-12-01
```

- It will download the required data as below. Unzip the zip files under the ``1h``, ``4h`` and ``1d`` directories.
```bash
binance_prediction_pytorch
    `-- binance-public-data
        `-- data
            `-- data
                `-- spot
                    |-- daily
                    `-- monthly
                        `-- klines
                            |-- ETHUSDT
                            `-- BTCUSDT
```
- Then soft link the data directory to the repo root as below.
```bash
binance_prediction_pytorch
    |-- binance-public-data
    `-- data
        `-- spot
            |-- daily
            `-- monthly
                `-- klines
                    |-- ETHUSDT
                    `-- BTCUSDT
```

## Experiments
### Training
- Run training and evaluation on ETHUSDT. It will store the checkpoints under ``checkpoints`` with ticker name and time interval if don't specify the checkpoint path with ``--ckpt``.
```bash
# 1d
./run.sh ETHUSDT 1d

# 4h
./run.sh ETHUSDT 4h --sell_rate 0.03

# 1h
./run.sh ETHUSDT 1h --sell_rate 0.03
```

- Run training and evaluation on BTCUSDT
```bash
# 1d
./run.sh BTCUSDT 1d

# 4h
./run.sh BTCUSDT 4h --sell_rate 0.03

# 1h
./run.sh BTCUSDT 1h --sell_rate 0.03
```

### Inference
- Specify the checkpoint path with ``eval`` mode to only do the inference.
```bash
./run.sh ETHUSDT 1h --sell_rate 0.03 --ckpt ${YOUR_CHECKPOINT_PATH} --eval
```

### Jupyter Notebook
- Directly try with jupyter notebook file [Final.ipynb](./Final.ipynb).