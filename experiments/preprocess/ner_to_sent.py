# author: trangvu
# Convert NER format to sentence format
import argparse


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input-dir", required=True,
                      help="Input dir.")
  parser.add_argument("--output-dir", required=True,
                      help="Output dir.")
  parser.add_argument("--input-file", required=True,
                      help="Input file")
  parser.add_argument("--output-file", required=True,
                      help="Output file")
  args = parser.parse_args()

  output_file = "{}/{}".format(args.output_dir, args.output_file)
  input_file = "{}/{}".format(args.input_dir, args.input_file)
  token_count = 0
  sentences = []
  sent_len = []
  with open(input_file, 'r') as fin:
      sentence = []
      for line in fin:
          line = line.strip().split()
          if not line:
              if sentence:
                  sent_len.append(len(sentence))
                  sentences.append(sentence)
                  sentence = []
              continue
          if line[0] == "-DOCSTART-":
              continue
          token_count += 1
          word= line[0]
          sentence.append(word)
  num_sent = len(sentences)
  print("Num tokens: {}".format(token_count))
  print("Num sentences: {}".format(num_sent))
  print("Average sentence len: {}".format(sum(sent_len)/len(sent_len)))
  print("Max sentence len: {}".format(max(sent_len)))
  print("Min sentence len: {}".format(min(sent_len)))

  with open(output_file, 'w') as fout:
      for sent in sentences:
          sent_print = ' '.join(sent)
          fout.write("{}\n".format(sent_print))



if __name__ == "__main__":
  main()
