import collections
import os
from collections import namedtuple, Set

additional_data="/scratch/da33/trang/masked-lm/data/pubmed/sim/pubmed-02"
origin_data="/home/xvuthith/da33_scratch/trang/masked-lm/data/blue/data/BIOSSES/train_da.txt"
output_data="/home/xvuthith/da33_scratch/trang/masked-lm/data/blue/data/BIOSSES/sim_da.txt"
rest_data="/home/xvuthith/da33_scratch/trang/masked-lm/data/blue/data/BIOSSES/rand_da.txt"
num_sent=10000

def _read_words(filename):
  with open(filename, "r") as f:
      return f.read().replace("\n", "").lower().split(' ')

def build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    print("* Vocab size {}".format(len(words)))
    return word_to_id

def read_and_extract_ngrams(filename, word_to_id):
    unigrams = set()
    bigrams = set()
    three_grams = set()
    four_grams = set()
    with open(filename, 'r') as fin:
        for line in fin:
            token = line.strip().lower().split(' ')
            n = len(token)
            for i in range(n):
                unigrams.add(token[i])
                if i < n - 1:
                    bigrams.add(','.join(token[i:i+2]))
                if i < n - 2:
                    three_grams.add(','.join(token[i:i+3]))
                if i < n - 3:
                    four_grams.add(','.join(token[i:i+4]))
    return unigrams, bigrams, three_grams, four_grams

def compute_jaccard_score(ngram_doc, ngram_sent):
    intersection_cardinality = len(set.intersection(*[ngram_doc, ngram_sent]))
    union_cardinality = len(set.union(*[ngram_doc, ngram_sent]))
    return intersection_cardinality / float(union_cardinality)


# vocabs = build_vocab(origin_data)
unigrams, bigrams, three_grams, four_grams = read_and_extract_ngrams(origin_data, None)

additional_sents = []
with open(additional_data, 'r') as fin:
    for line in fin:
        token = line.strip().lower().split(' ')
        additional_sents.append(token)

print("Number of candidate sentences: {}".format(len(additional_sents)))
scores = []
for sent in additional_sents:
    n = len(sent)
    sent_unigram = set()
    sent_bigram = set()
    sent_3gram = set()
    sent_4gram = set()
    for i in range(n):
        sent_unigram.add(sent[i])
        if i < n - 1:
            sent_bigram.add(','.join(sent[i:i+2]))
        if i < n - 2:
            sent_3gram.add(','.join(sent[i:i+3]))
        if i < n - 3:
            sent_4gram.add(','.join(sent[i:i+4]))

    score_1 =  compute_jaccard_score(unigrams, sent_unigram)
    score_2 = compute_jaccard_score(bigrams, sent_bigram)
    score_3 = compute_jaccard_score(three_grams, sent_3gram)
    score_4 = compute_jaccard_score(four_grams, sent_4gram)
    avg_score = (score_1 + score_2 + score_3 + score_4)/4
    scores.append(avg_score)
    print(avg_score)

indices=sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
selected_indices = indices[0:num_sent]
rest_indices = indices[num_sent:-1]
with open(output_data, 'w') as fout:
    for i in selected_indices:
        tokens = additional_sents[i]
        fout.write("{}\n".format(' '.join(tokens)))

with open(rest_data, 'w') as fout:
    for i in rest_indices:
        tokens = additional_sents[i]
        fout.write("{}\n".format(' '.join(tokens)))