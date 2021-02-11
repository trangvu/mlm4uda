from tqdm import tqdm
import json
from sklearn.feature_extraction.text import CountVectorizer
from typing import List
import itertools
import numpy as np
import argparse


def load_data(data_path: str) -> List[str]:
    examples = []
    with tqdm(open(data_path, "r"), desc=f"loading {data_path}") as f:
        for line in f:
            line = line.strip()
            if line:
                if data_path.endswith(".jsonl") or data_path.endswith(".json"):
                    example = json.loads(line)
                else:
                    example = {"text": line}
                text = example['text']
                examples.append(text)
    return examples


def load_vocab(file):
    text = load_data(file)
    count_vectorizer = CountVectorizer(min_df=1)
    count_vectorizer.fit(tqdm(text))
    vocab = set(count_vectorizer.vocabulary_.keys())
    return vocab


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--files", nargs="+", help="files containing tokenized text from each domain", required=True)
    # parser.add_argument("--output_file", help="path to save heatmap", required=True)
    #
    # args = parser.parse_args()
    # files = args.files
    vocabs = {}
    SOURCE_TASK="conll2003"
    TASK_NAMES = ["wnut2016", "fin", "jnlpba", "bc2gm", "bionlp09", "bionlp11epi"]
    source_vocab = load_vocab("/home/trang/workspace/mlm/data/conll2003/train.sent.txt")
    data_dir = "/home/trang/workspace/mlm/data"
    for task in TASK_NAMES:
        file = "{}/{}/test.sent.txt".format(data_dir, task)
        vocabs[task] = load_vocab(file)
        differences = vocabs[task] - source_vocab
        string = "\n".join(differences)
        with open("/home/trang/workspace/repo/advLM/adv/experiments/analysis/oov/{}.txt".format(task), "w+") as f:
            f.write(string)
