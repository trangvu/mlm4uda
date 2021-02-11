#!/bin/bash

ROOT_DIR="/scratch/da33/trang/masked-lm"
SRC_PATH=$ROOT_DIR'/bert'
DATA_ROOT=$ROOT_DIR'/train'

module load python/3.6.2
module load cuda/10.0
module load cudnn/7.6.5-cuda10.1
source $ROOT_DIR/env/bin/activate

set -x
MASK_STRATEGY=$1
CHECKPOINT=$2
model_prefix=$3
DATA_DIR=$ROOT_DIR/data/blue/bert_data
CONFIG_JSON=$SRC_PATH/experiments/config/blue

cd $SRC_PATH && python3 run_finetuning.py \
--data-dir=$DATA_DIR \
--hparams=$CONFIG_JSON/finetune_biosses.json \
--model-name='biosses'$model_prefix'-'$MASK_STRATEGY \
--init-checkpoint=$CHECKPOINT
