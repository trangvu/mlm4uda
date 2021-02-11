

TASK_NAMES = [ "wnut2016", "fin", "jnlpba", "bc2gm", "bionlp09", "bionlp11epi" ]
task = TASK_NAMES[5]
TEST=[1,2,3,4,5]
ground_truth = "/home/trang/workspace/mlm/data/{}/test.sent.txt".format(task)
oov = "/home/trang/workspace/repo/advLM/adv/experiments/analysis/oov/{}.txt".format(task)
testset = []
with open(ground_truth) as fin:
    for line in fin:
        line = line.strip().lower()
        if line:
            testset.append(line.split())
vocab = []
with open(oov) as fin:
    for line in fin:
        line = line.strip()
        if line:
            vocab.append(line)

vocab = set(vocab)
fout = open("/home/trang/workspace/repo/advLM/adv/experiments/analysis/oov/acc.txt","w")
# EXPS = ["wnut2016_ext_adv_ner_span_base_cased", "wnut2016_ext_mix_adv_ner_span_base_cased"]
# EXPS = ["fin_ext_entropy_ner_span_base_cased", "fin_ext_adv_ner_span_base_cased", "fin_entropy_ner_span_base_cased"]
# EXPS = ["jnlpba_ext_mix_adv_ner_span_base_cased", "jnlpba_ext_adv_ner_span_base_cased"]
# EXPS = ["bc2gm_ext_mix_adv_ner_span_base_cased", "bc2gm_ext_adv_ner_span_base_cased", "bc2gm_adv_ner_span_base_cased"]
# EXPS = ["bionlp09_ext_mix_adv_ner_span_base_cased", "bionlp09_ext_adv_ner_span_base_cased", "bionlp09_adv_ner_span_base_cased"]
EXPS = ["bionlp11epi_rand_ner_span_base_cased", "bionlp11epi_pos_ner_span_base_cased", "bionlp11epi_ext_mix_adv_ner_span_base_cased", "bionlp11epi_entropy_ner_span_base_cased", "bionlp11epi_adv_ner_span_base_cased"]
for exp in EXPS:
    data_dir = "/home/trang/workspace/repo/emnlp2020/test/{}/test_predictions".format(exp)
    for t in TEST:
        label_file = "{}/ner_span_test_{}_predictions.pkl_label.txt".format(data_dir,t)
        prediction_file = "{}/ner_span_test_{}_predictions.pkl_pred.txt".format(data_dir, t)

        labelset = []
        with open(label_file) as fin:
            for line in fin:
                line = line.strip()
                if line:
                    labelset.append(line.split())

        predset = []
        with open(prediction_file) as fin:
            for line in fin:
                line = line.strip()
                if line:
                    predset.append(line.split())

        total = 0
        acc = 0
        total_oov = 0
        total_non_oov = 0
        oov_acc = 0
        non_oov_acc = 0
        for pred, label, raw in zip(predset, labelset, testset):
            sent_len = len(pred)
            for i in range(sent_len):
                if label[i] != "0":
                    total += 1
                    if label[i] == pred[i]:
                        acc += 1
                        if i >= len(raw):
                            non_oov_acc += 1
                            total_non_oov += 1
                        elif raw[i] in vocab:
                            oov_acc += 1
                            total_oov += 1
                        else:
                            non_oov_acc += 1
                            total_non_oov += 1
                    else:
                        if i >= len(raw):
                            total_non_oov += 1
                        elif raw[i] in vocab:
                            total_oov += 1
                        else:
                            total_non_oov += 1
        assert total == total_oov + total_non_oov
        assert  acc == oov_acc + non_oov_acc
        accuracy = 100 * float(acc)/float(total)
        oov_accuracy = 100 *  float(oov_acc) / float(total_oov)
        non_oov_accuracy = 100 *  float(non_oov_acc) / float(total_non_oov)
        fout.write("{}-{}: acc={:10.2f} oov={:10.2f} non_oov={:10.2f}\n".format(exp,t, accuracy, oov_accuracy, non_oov_accuracy))
fout.close()