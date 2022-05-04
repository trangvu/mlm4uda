# mlm4uda
Source code for paper [Effective Unsupervised Domain Adaptation with Adversarially Trained Language Models](https://www.aclweb.org/anthology/2020.emnlp-main.497)


## Dataset
- WNUT2016 [Data](https://github.com/aritter/twitter_nlp/tree/master/data/annotated/wnut16)
- CoNLL2003 [Data](https://github.com/synalp/NER/tree/master/corpus/CoNLL-2003) [Other](https://nlp.stanford.edu/projects/project-ner.shtml)
    (eng.testa - dev, eng.testb - test)
    Base on Reuter newswire data (English and German)`
    Labels: ORG, PER, LOC, MISC
    Format: <Token> <POS tag> <chunk tag> <NER tag>
- Financial NER [Data](http://people.eng.unimelb.edu.au/tbaldwin/resources/finance-sec)
    FIN3: 3 financial agreement documents
    FIN5: 5 financial agreement documents
- WNUT2017 [Data](https://noisy-text.github.io/2017/emerging-rare-entities.html)
- BioNER [Data](https://github.com/cambridgeltl/MTL-Bioinformatics-2016)
    + Anatomy: AnatEM, CRAFT-anatomy
    + Gene/Protein: BC2GM, BioNLP09, BioNLP11EPI, BioNLP13GE, Ex-PTM, JNPBA
    + Chemical: BC4CHEMD, BC5CDR-chem, CRAFT, BIONLP13CG      

## Experiments
All experiment scripts can be found under `experiments` directory
### Preprocess
Build pretraining dataset
```bash
CORPUS_DIR=
VOCAB_FILE=$BERT_BASE_CASED/vocab.txt
OUT_DIR=
MAX_SEQ_LEN=128
NUM_OUT_FILES=50
python3 build_pretraining_dataset.py \
    --corpus-dir=$CORPUS_DIR \
    --vocab-file=$VOCAB_FILE \
    --output-dir=$OUT_DIR \
    --max-seq-length=$MAX_SEQ_LEN \
    --num-processes=1 \
    --blanks-separate-docs=False \
    --num-out-files=$NUM_OUT_FILES
```
### Domain tuning
Task config json files can be found under `experiments/config/domain-tuning` directory

```bash
DATA_DIR=
CONFIG_FILE=fin_noext_adv.json
MODEL_NAME=
python3 run_pretraining.py \
  --data-dir=$DATA_DIR \
  --hparams=$CONFIG_FILE \
  --model-name=$MODEL_NAME
```
### Run NER span dectection task
Task config json files can be found under `experiments/config/` directory
```bash
MODEL_CONFIG=fin_span_finetune.json
CHECKPOINT=
python3 run_finetuning.py \
    --data-dir=$DATA_DIR \
    --hparams=$MODEL_CONFIG \
    --model-name='ner' \
    --init-checkpoint=$CHECKPOINT
```

## References
Please cite the following paper if you found the resources in this repository useful.
```
@inproceedings{vu-etal-2020-effective,
    title = "Effective Unsupervised Domain Adaptation with Adversarially Trained Language Models",
    author = "Vu, Thuy-Trang  and
      Phung, Dinh  and
      Haffari, Gholamreza",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.emnlp-main.497",
    doi = "10.18653/v1/2020.emnlp-main.497",
    pages = "6163--6173"
}
```


## Acknowledgement
This project is implemented based on [electra](https://github.com/google-research/electra) source code
