#!/bin/bash

ROOT_DIR="/scratch/da33/trang/masked-lm"
SRC_PATH=$ROOT_DIR'/bert'
DATA_ROOT=$ROOT_DIR'/train'

module load python/3.6.2
module load cuda/10.0
module load cudnn/7.6.5-cuda10.1
source $ROOT_DIR/env/bin/activate

model_prefix=$1
set -x
MASK_STRATEGY=pos
MODEL_DIR=$ROOT_DIR/models/pubmed
CONFIG_JSON=$SRC_PATH/experiments/config
cd $SRC_PATH && python3 run_pretraining.py \
--data-dir=$MODEL_DIR \
--hparams=$CONFIG_JSON/pubmed_base_$MASK_STRATEGY.json \
--model-name='pubmed-da-'$model_prefix'-'$MASK_STRATEGY