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
    counter = collections.Counter()
    total_count = 0

    TASK_NAMES = [ "wnut2016", "fin", "jnlpba", "bc2gm", "bionlp09", "bionlp11epi" ]
    STRATEGIES = [ "rand", "pos", "entropy", "adv" ]
    for task in TASK_NAMES:
        PREFER_TAGS = ['ADJ', 'VERB', 'NOUN', 'PRON', 'ADV']
        for stragtegy in STRATEGIES:
            masked_outs = []
            sentences = []
            masked_positions = []
            pos_tag_regex = r'^_.*_$'
            log_file = "/media/trang/data/1Working/emnlp2020/dt/dt-{}-{}.log".format(task, stragtegy)
            fratio = open("/home/trang/workspace/repo/advLM/adv/experiments/analysis/syntactic_ratio/pos_{}_{}.txt".
                          format(task,stragtegy),"w")
            totals = []
            nouns = []
            verbs = []
            adjs = []
            advs = []
            prons = []
            nones = []
            with tqdm(open(log_file, "r"), desc=f"loading {log_file}") as f:
                for line in f:
                    line = line.strip()
                    total = 0
                    noun = 0
                    verb = 0
                    adj = 0
                    adv = 0
                    pron = 0
                    none = 0
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
                                total += 1
                                #['ADJ', 'VERB', 'NOUN', 'PRON', 'ADV']
                                if pos_tag == PREFER_TAGS[0]:
                                    adj += 1
                                elif pos_tag == PREFER_TAGS[1]:
                                    verb += 1
                                elif pos_tag == PREFER_TAGS[2]:
                                    noun += 1
                                elif pos_tag == PREFER_TAGS[3]:
                                    pron += 1
                                elif pos_tag == PREFER_TAGS[4]:
                                    adv += 1
                                else:
                                    none += 1
                                i += 3
                            else:
                                break
                            #     # print(line)
                            #     raise('Wrong log')

                        totals.append(total)
                        adjs.append(adj)
                        verbs.append(verb)
                        nouns.append(noun)
                        prons.append(pron)
                        advs.append(adv)
                        nones.append(none)
            num_period = 10
            period_len = int(len(totals) / num_period)
            i = -1
            for n in range(num_period):
                sub_totals = 0
                sub_nouns = 0
                sub_verbs = 0
                sub_adjs = 0
                sub_advs = 0
                sub_prons = 0
                sub_nones = 0
                for k in range(period_len):
                    i += 1
                    sub_totals += totals[i]
                    sub_nouns += nouns[i]
                    sub_verbs += verbs[i]
                    sub_adjs += adjs[i]
                    sub_advs += advs[i]
                    sub_prons += prons[i]
                    sub_nones += nones[i]
                # ['ADJ', 'VERB', 'NOUN', 'PRON', 'ADV']
                a = "{} {} {} {} {} {} {}\n".format(sub_adjs, sub_verbs, sub_nouns,
                       sub_prons, sub_advs, sub_nones, sub_totals)
                fratio.write(a)