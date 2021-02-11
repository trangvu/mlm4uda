#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --ntasks=12
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --partition=short
#SBATCH --mem-per-cpu=4096
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=vuth0001@student.monash.edu
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err


ROOT_DIR="/mnt/lustre/projects/da33/trang/masked-lm"
DATE=`date '+%Y%m%d-%H%M%S'`
SRC_PATH=$ROOT_DIR'/bert'
DATA_ROOT="/mnt/lustre/projects/da33/trang/masked-lm"
DATA_DIR=$DATA_ROOT"/data"
#
module load python/3.6.2
source $ROOT_DIR/env/bin/activate

INPUT="wiki_tok"
DATASET="wikidump-en"
VOCAB_FILE="/mnt/lustre/projects/da33/trang/masked-lm/pretrained/cased_L-12_H-768_A-12/vocab.txt"
OUT_DIR="${DATA_ROOT}/train/wikibook-128"

cd $SRC_PATH && python3 build_pretraining_dataset.py \
    --corpus-dir="${DATA_ROOT}/wikidump/en-tok" \
    --vocab-file=$VOCAB_FILE \
    --output-dir=$OUT_DIR \
    --max-seq-length=128 \
    --num-processes=12 \
    --blanks-separate-docs=True \
    --do-lower-case \
    --num-out-files=50
