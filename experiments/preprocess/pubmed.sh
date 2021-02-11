#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --ntasks=12
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4096
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=vuth0001@student.monash.edu
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err


ROOT_DIR="/home/xvuthith/da33_scratch/trang/masked-lm"
DATE=`date '+%Y%m%d-%H%M%S'`
SRC_PATH=$ROOT_DIR'/bert'
DATA_ROOT="/home/xvuthith/da33_scratch/trang/masked-lm"
DATA_DIR=$DATA_ROOT"/data"
#
module load python/3.6.2
source $ROOT_DIR/env/bin/activate
VOCAB_FILE="/home/xvuthith/da33_scratch/trang/masked-lm/models/bert_base_uncased/vocab.txt"
OUT_DIR="${DATA_ROOT}/train/pubmed-128"

cd $SRC_PATH && python3 build_pretraining_dataset.py \
    --corpus-dir="/home/xvuthith/da33_scratch/trang/masked-lm/data/pubmed/sim/corpus" \
    --vocab-file=$VOCAB_FILE \
    --output-dir=$OUT_DIR \
    --max-seq-length=128 \
    --num-processes=12 \
    --blanks-separate-docs=False \
    --do-lower-case \
    --num-out-files=500
