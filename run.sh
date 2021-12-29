#!/usr/bin/env bash
set -x
# ----configs----
ticker=$1
inter=$2
PY_ARGS=${@:3}

# Run the model
python trainer.py --data_dir data/spot/monthly/klines/$ticker/$inter/2017-08-01_2021-12-01 \
--ticker $ticker \
--inter $inter \
${PY_ARGS}
