#!/bin/bash

ROOT_DIR="/scratch/da33/trang/masked-lm"
SRC_PATH=$ROOT_DIR'/advLM/adv'
DATA_ROOT='/projects/da33/data_nlp/named_entity_recognition'

module load python/3.6.2
module load cuda/10.0
module load cudnn/7.6.5-cuda10.1
source $ROOT_DIR/env/bin/activate

DATASETS=( 'wnut2016' 'fin' 'jnlpba' 'bc2gm' 'bionlp09' 'bionlp11epi')
STRATEGIES=( 'rand' 'pos' 'entropy' 'adv' 'mix_pos' 'mix_entropy' 'mix_adv' "rw")
DATASET_IDX=$1
STRATEGY_ID=$2


DATASET_NAME=${DATASETS[DATASET_IDX]}
STRATEGY_NAME=${STRATEGIES[STRATEGY_ID]}
CONFIG_FILE=$SRC_PATH/experiments/config/domain-tuning-external-2M/$DATASET_NAME'_ext_2M_'$STRATEGY_NAME'.json'
MODEL_NAME=$DATASET_NAME"_conll2003_ext_2M_50K_base_"$STRATEGY_NAME
MODEL_DIR=$ROOT_DIR/models/$DATASET_NAME
echo '**************************************'
echo 'CONTINUE PRETRAINING'
echo 'DATASET_NAME      : '$DATASET_NAME
echo 'CONFIG    : '$CONFIG_FILE
echo 'MODEL_NAME: '$MODEL_NAME
echo 'MODEL_DIR  : '$MODEL_DIR
echo '**************************************'

cd $SRC_PATH && python3 run_pretraining.py \
  --data-dir=$MODEL_DIR \
  --hparams=$CONFIG_FILE \
  --model-name=$MODEL_NAME

echo "Step 2: Start task tuning on CoNLL2003 span"
CONFIG_JSON=$SRC_PATH/experiments/config
TASK_NAME=ner_span
TASK_TUNING_CONFIG_FILE=$CONFIG_JSON/ner_span_uda.json
TASK_TUNING_MODEL_NAME=${DATASET_NAME}'_ext_2M_50K_'${STRATEGY_NAME}_${TASK_NAME}_base_cased$PREFIX
INIT_MODEL_NAME=$MODEL_NAME
INIT_MODEL=${MODEL_DIR}/models/$INIT_MODEL_NAME
DATA_DIR=$DATA_ROOT/conll2003_${DATASET_NAME}_span

    echo '**************************************'
    echo 'TASK TUNING NER SPAN TASKS'
    echo 'TASK         : '$TASK_NAME
    echo 'DATASET_NAME : '$DATASET_NAME
    echo 'CHECKPOINT   : '$INIT_MODEL
    echo 'CONFIG       : '$TASK_TUNING_CONFIG_FILE
    echo 'MODEL_NAME   : '$TASK_TUNING_MODEL_NAME
    echo 'DATA_DIR     : '$DATA_DIR
    echo '**************************************'

    cd $SRC_PATH && python3 run_finetuning.py \
    --data-dir=$DATA_DIR \
    --hparams=$TASK_TUNING_CONFIG_FILE \
    --model-name=$TASK_TUNING_MODEL_NAME \
    --init-checkpoint=$INIT_MODEL

echo "Cleaning model"
rm -r -f $DATA_DIR/models/$TASK_TUNING_MODEL_NAME/finetuning_models

echo "Get result"
cd $DATA_DIR/models/$TASK_TUNING_MODEL_NAME/result; sed -n 'p;n' *_results.txt > results_dev.txt;sed -n 'n;p' *_results.txt > results_test.txt;cat results_test.txt | cut -d' ' -f9