#!/bin/bash

ROOT_DIR="/scratch/da33/trang/masked-lm"
SRC_PATH=$ROOT_DIR'/advLM/adv'
DATA_ROOT='/projects/da33/data_nlp/named_entity_recognition'

module load python/3.6.2
module load cuda/10.0
module load cudnn/7.6.5-cuda10.1
source $ROOT_DIR/env/bin/activate

#DATASETS=( 'wnut2016' 'fin' 'jnlpba' 'bc2gm' 'bionlp09' 'bionlp11epi' 'sentiment140' 'pubmed' )
DATASETS=( 'wnut2016' 'fin' 'jnlpba' 'bc2gm' 'bionlp09' 'bionlp11epi' )
STRATEGIES=( 'rand' 'pos' 'entropy' 'adv' )
#STRATEGIES=( 'rand' 'pos' 'entropy' 'adv' 'mix' )
TASK_NAME=ner_span

#DATASET_IDX=$1
#STRATEGY_ID=$2
#DATASET_NAME=${DATASETS[DATASET_IDX]}
#STRATEGY_NAME=${STRATEGIES[STRATEGY_ID]}
PREFIX=$1
CONFIG_JSON=$SRC_PATH/experiments/config

for DATASET_NAME in "${DATASETS[@]}"; do
  for STRATEGY_NAME in "${STRATEGIES[@]}"; do
    CONFIG_FILE=$CONFIG_JSON/ner_span_uda.json
    MODEL_NAME=${DATASET_NAME}_${STRATEGY_NAME}_${TASK_NAME}_base_cased$PREFIX
    INIT_MODEL_NAME=$DATASET_NAME"_conll2003_noext_base_"$STRATEGY_NAME
    INIT_MODEL=/scratch/da33/trang/masked-lm/models/${DATASET_NAME}/models/$INIT_MODEL_NAME
    DATA_DIR=$DATA_ROOT/conll2003_${DATASET_NAME}_span

    echo '**************************************'
    echo 'TASK TUNING NER SPAN TASKS'
    echo 'TASK         : '$TASK_NAME
    echo 'DATASET_NAME : '$DATASET_NAME
    echo 'CHECKPOINT   : '$INIT_MODEL
    echo 'CONFIG       : '$CONFIG_FILE
    echo 'MODEL_NAME   : '$MODEL_NAME
    echo 'DATA_DIR     : '$DATA_DIR
    echo '**************************************'

    cd $SRC_PATH && python3 run_finetuning.py \
    --data-dir=$DATA_DIR \
    --hparams=$CONFIG_FILE \
    --model-name=$MODEL_NAME \
    --init-checkpoint=$INIT_MODEL
  done
done