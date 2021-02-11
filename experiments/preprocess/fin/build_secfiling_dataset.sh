#!/bin/bash

ROOT_DIR="/scratch/da33/trang/masked-lm"
SRC_PATH=$ROOT_DIR'/advLM/adv'
DATA_ROOT='/projects/da33/data_nlp/named_entity_recognition'

module load python/3.6.2
module load cuda/10.0
module load cudnn/7.6.5-cuda10.1
source $ROOT_DIR/env/bin/activate

CONFIG_JSON=$SRC_PATH/experiments/config
BERT_BASE_CASED=$ROOT_DIR'/models/bert_base_cased'

#SIZES=( '100k' '500k' '1M' )
SIZES=( '2M' )
DATASET_NAME=secfiling
VOCAB_FILE=$BERT_BASE_CASED'/vocab.txt'
MAX_SEQ_LEN=128

for SIZE in "${SIZES[@]}"; do
  CORPUS_DIR=$DATA_ROOT/$DATASET_NAME'/raw-'$SIZE
  OUT_DIR=$DATA_ROOT/$DATASET_NAME'/tfrecord-'$SIZE
  NUM_OUT_FILES=100
  echo '**************************************'
  echo 'BUILD PRETRAINING DATASET'
  echo 'DATASET      : '$DATASET_NAME
  echo 'VOCAB        : '$VOCAB_FILE
  echo 'MAX SEQ LEN  : '$MAX_SEQ_LEN
  echo 'CORPUS DIR   : '$CORPUS_DIR
  echo 'OUTPUT DIR   : '$OUT_DIR
  echo 'NUM_OUT_FILES: '$NUM_OUT_FILES
  echo '**************************************'


  cd $SRC_PATH && python3 build_pretraining_dataset.py \
      --corpus-dir=$CORPUS_DIR \
      --vocab-file=$VOCAB_FILE \
      --output-dir=$OUT_DIR \
      --max-seq-length=$MAX_SEQ_LEN \
      --num-processes=1 \
      --blanks-separate-docs=False \
      --num-out-files=$NUM_OUT_FILES
done