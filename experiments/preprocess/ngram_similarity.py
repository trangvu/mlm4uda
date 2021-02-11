# author: trangvu
# Sort sentence by ngram similarity
import argparse

def read_and_extract_ngrams(filename):
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

def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input-file", required=True,
                      help="Input file")
  parser.add_argument("--output-file", required=True,
                      help="Output file")
  parser.add_argument("--gold-file", required=True,
                      help="Gold file")
  args = parser.parse_args()

  output_file = args.output_file
  additional_data = args.input_file
  origin_data = args.gold_file
  unigrams, bigrams, three_grams, four_grams = read_and_extract_ngrams(origin_data)

  additional_sents = []
  with open(additional_data, 'r') as fin:
      for line in fin:
          token = line.strip().split(' ')
          additional_sents.append(token)

  print("Number of candidate sentences: {}".format(len(additional_sents)))
  scores = []
  count = 0
  fout = open(output_file + ".score", 'w')
  for sent in additional_sents:
      n = len(sent)
      sent_unigram = set()
      sent_bigram = set()
      sent_3gram = set()
      sent_4gram = set()
      for i in range(n):
          sent_unigram.add(sent[i].lower())
          if i < n - 1:
              sent_bigram.add("{},{}".format(sent[i].lower(),sent[i + 1].lower()))
          if i < n - 2:
              sent_3gram.add("{},{},{}".format(sent[i].lower(),sent[i + 1].lower(),sent[i + 2].lower()))
          if i < n - 3:
              sent_4gram.add("{},{},{},{}".format(sent[i].lower(),sent[i + 1].lower(),sent[i + 2].lower(),sent[i + 3].lower()))

      score_1 = compute_jaccard_score(unigrams, sent_unigram)
      score_2 = compute_jaccard_score(bigrams, sent_bigram)
      score_3 = compute_jaccard_score(three_grams, sent_3gram)
      score_4 = compute_jaccard_score(four_grams, sent_4gram)
      avg_score = (score_1 + score_2 + score_3 + score_4) / 4
      fout.write("{:10.10f}\n".format(avg_score))
      scores.append(avg_score)
      count += 1
      if count % 1000 == 0:
          print("...processing {} setences so far".format(count))
  print("Finish calculate score")
  fout.close()
  indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
  with open(output_file, 'w') as fout:
      for i in indices:
          tokens = additional_sents[i]
          fout.write("{}\n".format(' '.join(tokens)))

if __name__ == "__main__":
  main()
