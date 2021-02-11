#!/bin/bash

ROOT_DIR="/scratch/da33/trang/masked-lm"
SRC_PATH=$ROOT_DIR'/advLM/adv'
DATA_ROOT='/projects/da33/data_nlp/named_entity_recognition'

module load python/3.6.2
module load cuda/10.0
module load cudnn/7.6.5-cuda10.1
source $ROOT_DIR/env/bin/activate

DATASETS=( 'conll2003' 'wnut2016' 'fin' 'jnlpba' 'bc2gm' 'bionlp09' 'bionlp11epi' 'sentiment140' 'pubmed' )

DATASET_IDX=$1
CONFIG_JSON=$SRC_PATH/experiments/config
BERT_BASE_CASED=$ROOT_DIR'/models/bert_base_cased'

DATASET_NAME=${DATASETS[DATASET_IDX]}
VOCAB_FILE=$BERT_BASE_CASED'/vocab.txt'
MAX_SEQ_LEN=128
CORPUS_DIR=$DATA_ROOT/$DATASET_NAME'/raw'
OUT_DIR=$DATA_ROOT/$DATASET_NAME'/tfrecord'
NUM_OUT_FILES=50
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