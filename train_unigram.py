import collections

from tqdm import tqdm
from itertools import chain
import argparse
from model import tokenization

def tokenize_and_align(tokenizer, words):
  """Splits up words into subword-level tokens."""
  basic_tokenizer = tokenizer.basic_tokenizer
  tokenized_words = []
  for word in words:
    word = tokenization.convert_to_unicode(word)
    word = basic_tokenizer._clean_text(word)
    if word == "[CLS]" or word == "[SEP]":
      word_toks = [word]
    else:
      word_toks = basic_tokenizer._run_split_on_punc(word)
    tokenized_word = []
    for word_tok in word_toks:
      tokenized_word += tokenizer.wordpiece_tokenizer.tokenize(word_tok)
    tokenized_words.append(tokenized_word)
  assert len(tokenized_words) == len(words)
  flatten = list(chain.from_iterable(tokenized_words))
  return flatten

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", help="corpus file", required=True)
    parser.add_argument("--output_file", help="output model file", required=True)
    parser.add_argument("--vocab_file", help="vocab file", required=True)
    args = parser.parse_args()
    corpus_file = args.corpus
    output_file = args.output_file
    vocab_file = args.vocab_file
    print(corpus_file)
    counter = collections.Counter()
    total_count = 0

    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file,
                                           do_lower_case=False)
    with tqdm(open(corpus_file, "r"), desc=f"loading {corpus_file}") as f:
        for line in f:
            line = line.strip()
            if line:
                line = "[CLS] {} [SEP]".format(line)
                tokens = line.split()
                pieces = tokenize_and_align(tokenizer, tokens)
                total_count += len(pieces)
                counter.update(list(pieces))

    with open(output_file, "w") as fout:
        for key in tokenizer.vocab:
            if key in counter:
                p = float(counter[key])/total_count
            else:
                p = 1e-10
            fout.write("{}\t{:10.10f}\n".format(key,p))