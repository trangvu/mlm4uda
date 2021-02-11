import collections

from tqdm import tqdm
import argparse
from itertools import chain
import re
import numpy as np

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--log_file", help="log file of masked out sentences", required=True)
    # parser.add_argument("--src_model", help="source model path", required=True)
    # parser.add_argument("--tgt_model", help="tgt model path", required=True)
    # parser.add_argument("--vocab_file", help="vocab file", required=True)
    # args = parser.parse_args()
    # log_file = args.log_file
    # src_model = args.src_model
    # tgt_model = args.tgt_model
    # vocab_file = args.vocab_file
    src_model = "/home/trang/workspace/repo/advLM/adv/experiments/analysis/density_ratio/unigram.wordpiece.conll2003.tsv"
    tgt_model = "/home/trang/workspace/repo/advLM/adv/experiments/analysis/density_ratio/unigram.wordpiece.wnut2016.tsv"
    vocab_file = "/home/trang/workspace/mlm/data/models/bert_base_cased/vocab.txt"
    counter = collections.Counter()
    total_count = 0
    src_prob = {}
    with tqdm(open(src_model, "r"), desc=f"loading {src_model}") as f:
        for line in f:
            line = line.strip()
            if line:
                tokens = line.split()
                src_prob[tokens[0]] = float(tokens[1])

    TASK_NAMES = [ "wnut2016", "fin", "jnlpba", "bc2gm", "bionlp09", "bionlp11epi" ]
    for task in TASK_NAMES:
        tgt_prob = {}
        tgt_model = "/home/trang/workspace/repo/advLM/adv/experiments/analysis/density_ratio/unigram.wordpiece.{}.tsv".format(task)
        with tqdm(open(tgt_model, "r"), desc=f"loading {tgt_model}") as f:
            for line in f:
                line = line.strip()
                if line:
                    tokens = line.split()
                    tgt_prob[tokens[0]] = float(tokens[1])
        with open("/home/trang/workspace/repo/advLM/adv/experiments/analysis/density_ratio/rw_{}.txt".format(task), "w") as fout:
            for key in tgt_prob:
                r = 1 - (src_prob[key]/tgt_prob[key])
                if r < 0:
                    r = 1e-5
                fout.write("{}\t{:10.10f}\n".format(key,r))