import argparse

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input-file", required=True,
                      help="Input file")
  parser.add_argument("--output-file", required=True,
                      help="Output file")
  args = parser.parse_args()

  output_file = args.output_file
  input_file = args.input_file
  fout = open(output_file, 'w')
  num_sent = 0
  num_discard = 0
  num_clean_sent = 0
  with open(input_file, 'r') as fin:
      for line in fin:
          line = line.strip()
          sentences = sent_tokenize(line)
          for sent in sentences:
              num_sent +=1
              tokens = word_tokenize(sent)
              if len(tokens) < 5:
                  num_discard += 1
              else:
                  num_clean_sent += 1
                  fout.write("{}\n".format(' '.join(tokens)))
  fout.close()
  print("Total sentence: {}".format(num_sent))
  print("Total discard: {}".format(num_discard))
  print("Clean sentence: {}".format(num_clean_sent))

if __name__ == "__main__":
  main()
