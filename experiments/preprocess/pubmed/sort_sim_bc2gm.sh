#!/bin/bash
ROOT_DIR="/scratch/da33/trang/masked-lm"
SRC_PATH=$ROOT_DIR'/advLM/adv/experiments/preprocess'
DATA_ROOT='/projects/da33/data_nlp/named_entity_recognition'

module load python/3.6.2
module load cuda/10.0
module load cudnn/7.6.5-cuda10.1
source $ROOT_DIR/env/bin/activate
set -x
TASK_NAME=bc2gm
INFILE=/scratch/da33/trang/masked-lm/data/pubmed/$TASK_NAME/pubmed.txt
OUTFILE=/projects/da33/data_nlp/named_entity_recognition/$TASK_NAME/pubmed.$TASK_NAME.sim.sorted.txt
GOLDFILE=/projects/da33/data_nlp/named_entity_recognition/$TASK_NAME/train.sent.txt

cd $SRC_PATH && python ngram_similarity.py --input-file=$INFILE --output-file=$OUTFILE --gold-file=$GOLDFILE
