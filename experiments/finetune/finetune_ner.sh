#!/bin/bash

ROOT_DIR="/scratch/da33/trang/masked-lm"
SRC_PATH=$ROOT_DIR'/advLM/adv'
DATA_ROOT='/projects/da33/data_nlp/named_entity_recognition'

module load python/3.6.2
module load cuda/10.0
module load cudnn/7.6.5-cuda10.1
source $ROOT_DIR/env/bin/activate

TASKS=( 'conll2003' 'wnut2016' 'fin' 'jnlpba' 'bc2gm' 'bionlp09' 'bionlp11epi' )

TASK_IDX=$1
DATA_DIR=$ROOT_DIR/data/blue/bert_data
CONFIG_JSON=$SRC_PATH/experiments/config
BERT_BASE_CASED=$ROOT_DIR'/models/bert_base_cased'

TASK_NAME=${TASKS[$TASK_IDX]}
CONFIG_FILE=$CONFIG_JSON/${TASK_NAME}_finetune.json
MODEL_NAME=${TASK_NAME}_base_cased

echo '**************************************'
echo 'RUN NER TASKS'
echo 'TASK      : '$TASK_NAME
echo 'CHECKPOINT: '$BERT_BASE_CASED
echo 'CONFIG    : '$CONFIG_FILE
echo 'MODEL_NAME: '$MODEL_NAME
echo 'DATA_DIR  : '$DATA_ROOT
echo '**************************************'

cd $SRC_PATH && python3 run_finetuning.py \
--data-dir=$DATA_ROOT \
--hparams=$CONFIG_FILE \
--model-name=$MODEL_NAME \
--init-checkpoint=$BERT_BASE_CASED