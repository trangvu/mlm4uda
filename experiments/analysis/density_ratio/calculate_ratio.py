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
    # log_file = "/media/trang/data/1Working/emnlp2020/dt/dt-wnut2016-rand.log"
    src_model = "/home/trang/workspace/repo/advLM/adv/experiments/analysis/density_ratio/unigram.wordpiece.conll2003.tsv"
    # tgt_model = "/home/trang/workspace/repo/advLM/adv/experiments/analysis/density_ratio/unigram.wordpiece.wnut2016.tsv"
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
    STRATEGIES = [ "rand", "pos", "entropy", "adv" ]
    for task in TASK_NAMES:
        tgt_prob = {}
        tgt_model = "/home/trang/workspace/repo/advLM/adv/experiments/analysis/density_ratio/unigram.wordpiece.{}.tsv".format(task)
        with tqdm(open(tgt_model, "r"), desc=f"loading {tgt_model}") as f:
            for line in f:
                line = line.strip()
                if line:
                    tokens = line.split()
                    tgt_prob[tokens[0]] = float(tokens[1])

        for stragtegy in STRATEGIES:
            masked_outs = []
            sentences = []
            masked_positions = []
            pos_tag_regex = r'^_.*_$'
            log_file = "/media/trang/data/1Working/emnlp2020/dt/dt-{}-{}.log".format(task, stragtegy)
            fratio = open("/home/trang/workspace/repo/advLM/adv/experiments/analysis/density_ratio/ratio_{}_{}.txt".
                          format(task,stragtegy),"w")
            frank = open("/home/trang/workspace/repo/advLM/adv/experiments/analysis/density_ratio/rank_{}_{}.txt".
                         format(task, stragtegy), "w")
            ratios = []
            ranks = []
            with tqdm(open(log_file, "r"), desc=f"loading {log_file}") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        masked_out = []
                        sentence = []
                        masked_position = []
                        i = 0
                        tokens= line.split()
                        seq_len = len(tokens)
                        while i < seq_len:
                            if re.match(pos_tag_regex, tokens[i+1]):
                                sentence.append(tokens[i])
                                i += 2
                            elif (i+2 <seq_len) and (re.match(pos_tag_regex, tokens[i+2])):
                                replaced = tokens[i]
                                original = tokens[i+1][1:-1]
                                pos_tag = tokens[i+2][1:-1]
                                sentence.append(original)
                                masked_position.append(len(sentence) - 1)
                                masked_out.append((original, pos_tag))
                                i += 3
                            else:
                                break
                            #     # print(line)
                            #     raise('Wrong log')
                        p_src = [src_prob[token] if token in src_prob else 1e-5 for token in sentence]
                        p_tgt = [tgt_prob[token] if token in tgt_prob else 0.0 for token in sentence]
                        p_src = np.asarray(p_src)
                        p_tgt = np.asarray(p_tgt)
                        ratio = 1.0 - (p_src / p_tgt)
                        ratio[ratio < 0] = 0.0
                        sorted_ratio = sorted(ratio, reverse=True)
                        rank = [sorted_ratio.index(x) for x in ratio]
                        rank = np.asarray(rank)
                        masked_ratio = ratio[masked_position]
                        masked_rank = rank[masked_position]
                        ratios.append(masked_ratio)
                        ranks.append(masked_rank)
            num_period = 10
            period_len = int(len(ratios) / num_period)
            i = -1
            for n in range(num_period):
                sub_ratio = []
                sub_rank = []
                for k in range(period_len):
                    i += 1
                    sub_ratio.extend(ratios[i])
                    sub_rank.extend(ranks[i])
                fratio.write("{:10.10f}\n".format(np.average(sub_ratio)))
                frank.write("{:10.10f}\n".format(np.average(sub_rank)))