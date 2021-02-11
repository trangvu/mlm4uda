# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fine-tunes an ELECTRA model on a downstream task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import json

import tensorflow
import tensorflow.compat.v1 as tf

import configure_finetuning
from finetune import preprocessing
from finetune import task_builder
from model import modeling
from model import optimization
from util import training_utils
from util import utils
import numpy as np


class FinetuningModel(object):
  """Finetuning model with support for multi-task training."""

  def __init__(self, config: configure_finetuning.FinetuningConfig, tasks,
               is_training, features, num_train_steps):
    # Create a shared transformer encoder
    bert_config = training_utils.get_bert_config(config)
    self.bert_config = bert_config
    if config.debug:
      bert_config.num_hidden_layers = 3
      bert_config.hidden_size = 144
      bert_config.intermediate_size = 144 * 4
      bert_config.num_attention_heads = 4
    assert config.max_seq_length <= bert_config.max_position_embeddings

    embedding_size = (
      self.bert_config.hidden_size if config.embedding_size is None else
      config.embedding_size)

    bert_model = modeling.BertModel(
        bert_config=bert_config,
        is_training=is_training,
        input_ids=features["input_ids"],
        input_mask=features["input_mask"],
        token_type_ids=features["segment_ids"],
        use_one_hot_embeddings=config.use_tpu,
        embedding_size=embedding_size)
    percent_done = (tf.cast(tf.train.get_or_create_global_step(), tf.float32) /
                    tf.cast(num_train_steps, tf.float32))

    # Add specific tasksimport logging

    self.outputs = {"task_id": features["task_id"]}
    losses = []
    for task in tasks:
      with tf.variable_scope("task_specific/" + task.name):
        task_losses, task_outputs = task.get_prediction_module(
            bert_model, features, is_training, percent_done)
        losses.append(task_losses)
        self.outputs[task.name] = task_outputs
    self.loss = tf.reduce_sum(
        tf.stack(losses, -1) *
        tf.one_hot(features["task_id"], len(config.task_names)))


def model_fn_builder(config: configure_finetuning.FinetuningConfig, tasks,
                     num_train_steps, pretraining_config=None):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):
    """The `model_fn` for TPUEstimator."""
    utils.log("Building model...")
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    model = FinetuningModel(
        config, tasks, is_training, features, num_train_steps)

    # Load pre-trained weights from checkpoint
    init_checkpoint = config.init_checkpoint
    if pretraining_config is not None:
      init_checkpoint = tf.train.latest_checkpoint(pretraining_config.model_dir)
      utils.log("Using checkpoint", init_checkpoint)
    tvars = tf.trainable_variables()
    scaffold_fn = None
    initialized_variable_names = {}
    if init_checkpoint:
      utils.log("Using checkpoint", init_checkpoint)
      assignment_map, initialized_variable_names = modeling.get_assignment_map_from_checkpoint(
          tvars, init_checkpoint)
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    utils.log("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      utils.logerr("  name = %s, shape = %s%s", var.name, var.shape,
                init_string)

    # Build model for training or prediction
    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = optimization.create_optimizer(
          model.loss, config.learning_rate, num_train_steps,
          weight_decay_rate=config.weight_decay_rate,
          warmup_proportion=config.warmup_proportion,
          n_transformer_layers=model.bert_config.num_hidden_layers
      )
      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=model.loss,
          train_op=train_op,
          training_hooks=[training_utils.ETAHook(
              {} if config.use_tpu else dict(loss=model.loss),
              num_train_steps, config.iterations_per_loop, config.use_tpu, 10)])
    else:
      assert mode == tf.estimator.ModeKeys.PREDICT
      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          predictions=utils.flatten_dict(model.outputs))

    utils.log("Building complete")
    return output_spec

  return model_fn


class ModelRunner(object):
  """Fine-tunes a model on a supervised task."""

  def __init__(self, config: configure_finetuning.FinetuningConfig, tasks,
               pretraining_config=None):
    self._config = config
    self._tasks = tasks
    self._preprocessor = preprocessing.Preprocessor(config, self._tasks)

    num_gpus = utils.get_available_gpus()
    utils.log("Found {} gpus".format(len(num_gpus)))

    if num_gpus == 1:
      session_config = tf.ConfigProto(
        log_device_placement=True,
        allow_soft_placement=True,
        gpu_options=tf.GPUOptions(allow_growth=True))

      run_config = tf.estimator.RunConfig(
        model_dir=config.model_dir,
        save_checkpoints_steps=config.save_checkpoints_steps,
        # save_checkpoints_secs=3600,
        # tf_random_seed=FLAGS.seed,
        session_config=session_config,
        # keep_checkpoint_max=0,
        log_step_count_steps=100
      )
    else:
      train_distribution_strategy = tf.distribute.MirroredStrategy(
        devices=None,
        cross_device_ops=tensorflow.contrib.distribute.AllReduceCrossDeviceOps('nccl', num_packs=len(num_gpus)))
      eval_distribution_strategy = tf.distribute.MirroredStrategy(devices=None)

      session_config = tf.ConfigProto(
        # log_device_placement=True,
        inter_op_parallelism_threads=0,
        intra_op_parallelism_threads=0,
        allow_soft_placement=True,
        gpu_options=tf.GPUOptions(allow_growth=True))

      run_config = tf.estimator.RunConfig(
        model_dir=config.model_dir,
        save_checkpoints_steps=config.save_checkpoints_steps,
        train_distribute=train_distribution_strategy,
        eval_distribute=eval_distribution_strategy,
        # save_checkpoints_secs=3600,
        # tf_random_seed=FLAGS.seed,
        session_config=session_config,
        # keep_checkpoint_max=0,
        log_step_count_steps=100
      )

    if self._config.do_train:
      (self._train_input_fn,
       self.train_steps) = self._preprocessor.prepare_train()
    else:
      self._train_input_fn, self.train_steps = None, 0

    model_fn = model_fn_builder(
        config=config,
        tasks=self._tasks,
        num_train_steps=self.train_steps,
        pretraining_config=pretraining_config)
    self._estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      config=run_config,
      params={'train_batch_size': config.train_batch_size,
              'eval_batch_size': config.eval_batch_size})

  def train(self):
    utils.log("Training for {:} steps".format(self.train_steps))
    self._estimator.train(
        input_fn=self._train_input_fn, max_steps=self.train_steps)

  def evaluate(self):
    return {task.name: self.evaluate_task(task) for task in self._tasks}

  def test(self):
    return {task.name: self.evaluate_task(task, split="test") for task in self._tasks}

  def evaluate_task(self, task, split="dev", return_results=True):
    """Evaluate the current model."""
    utils.log("Evaluating", task.name)
    eval_input_fn, _ = self._preprocessor.prepare_predict([task], split)
    results = self._estimator.predict(input_fn=eval_input_fn,
                                      yield_single_examples=True)
    scorer = task.get_scorer()
    for r in results:
      if r["task_id"] != len(self._tasks):  # ignore padding examples
        r = utils.nest_dict(r, self._config.task_names)
        scorer.update(r[task.name])
    if return_results:
      utils.log(task.name + '-' + split + ": " + scorer.results_str())
      utils.log()
      return dict(scorer.get_results())
    else:
      return scorer

  def write_classification_outputs(self, tasks, trial, split):
    """Write classification predictions to disk."""
    utils.log("Writing out predictions for", tasks, split)
    predict_input_fn, _ = self._preprocessor.prepare_predict(tasks, split)
    results = self._estimator.predict(input_fn=predict_input_fn,
                                      yield_single_examples=True)
    # task name -> eid -> model-logits
    logits = collections.defaultdict(dict)
    for r in results:
      if r["task_id"] != len(self._tasks):
        r = utils.nest_dict(r, self._config.task_names)
        task_name = self._config.task_names[r["task_id"]]
        logits[task_name][r[task_name]["eid"]] = (
            r[task_name]["logits"] if "logits" in r[task_name]
            else r[task_name]["predictions"])
    for task_name in logits:
      utils.log("Pickling predictions for {:} {:} examples ({:})".format(
          len(logits[task_name]), task_name, split))
      if trial <= self._config.n_writes_test:
        utils.write_pickle(logits[task_name], self._config.test_predictions(
            task_name, split, trial))

  def write_tagging_outputs(self, tasks, trial, split):
    """Write classification predictions to disk."""
    utils.log("Writing out predictions for", tasks, split)
    predict_input_fn, _ = self._preprocessor.prepare_predict(tasks, split)
    results = self._estimator.predict(input_fn=predict_input_fn,
                                      yield_single_examples=True)

    # task name -> eid -> model-logits
    labels = collections.defaultdict(dict)
    predictions = collections.defaultdict(dict)
    length = collections.defaultdict(dict)
    for r in results:
      if r["task_id"] != len(self._tasks):
        r = utils.nest_dict(r, self._config.task_names)
        task_name = self._config.task_names[r["task_id"]]
        predictions[task_name][r[task_name]["eid"]] = r[task_name]["predictions"]
        labels[task_name][r[task_name]["eid"]] = r[task_name]["labels"]
        length[task_name][r[task_name]["eid"]] = np.sum(r[task_name]["labels_mask"])
    for task_name in predictions:
      utils.log("Pickling predictions for {:} {:} examples ({:})".format(
          len(predictions[task_name]), task_name, split))
      if trial <= self._config.n_writes_test:
        preds_file = self._config.test_predictions(
          task_name, split, trial) + "_pred.txt"
        label_file = self._config.test_predictions(
          task_name, split, trial) + "_label.txt"
        task_preds = predictions[task_name]
        task_labels = labels[task_name]
        task_length = length[task_name]
        num_ex = len(task_preds)
        if "/" in preds_file:
          tf.io.gfile.makedirs(preds_file.rsplit("/", 1)[0])
        with tf.io.gfile.GFile(preds_file, "w") as fpred, tf.io.gfile.GFile(label_file, "w") as flabel:
          for i in range(num_ex):
            l = int(task_length[i])
            p = np.array(map(str,task_preds[i][0:l])).tolist()
            l = np.array(map(str,task_labels[i][0:l])).tolist()
            fpred.write("{}\n".format(' '.join(p)))
            flabel.write("{}\n".format(' '.join(l)))


def write_results(config: configure_finetuning.FinetuningConfig, results, split="dev"):
  """Write evaluation metrics to disk."""
  utils.log("Writing results to", config.results_txt)
  utils.mkdir(config.results_txt.rsplit("/", 1)[0])
  utils.write_pickle(results, config.results_pkl)
  with tf.io.gfile.GFile(config.results_txt, "w") as f:
    results_str = ""
    for trial_results in results:
      for task_name, task_results in trial_results.items():
        if task_name == "time" or task_name == "global_step":
          continue
        results_str += task_name + '-' + split + ": " + " - ".join(
            ["{:}: {:.2f}".format(k, v)
             for k, v in task_results.items()]) + "\n"
    f.write(results_str)
  utils.write_pickle(results, config.results_pkl)


def run_finetuning(config: configure_finetuning.FinetuningConfig):
  """Run finetuning."""

  # Setup for training
  results = []
  trial = 1
  heading_info = "model={:}, trial {:}/{:}".format(
      config.model_name, trial, config.num_trials)
  heading = lambda msg: utils.heading(msg + ": " + heading_info)
  heading("Config")
  utils.log_config(config)
  generic_model_dir = config.model_dir
  tasks = task_builder.get_tasks(config)

  # Train and evaluate num_trials models with different random seeds
  while config.num_trials < 0 or trial <= config.num_trials:
    heading_info = "model={:}, trial {:}/{:}".format(
      config.model_name, trial, config.num_trials)
    heading = lambda msg: utils.heading(msg + ": " + heading_info)
    config.model_dir = generic_model_dir + "_" + str(trial)
    if config.do_train:
      utils.rmkdir(config.model_dir)

    model_runner = ModelRunner(config, tasks)
    if config.do_train:
      heading("Start training")
      model_runner.train()
      utils.log()

    if config.do_eval:
      heading("Run dev set evaluation")
      results.append(model_runner.evaluate())
      write_results(config, results, "dev")
      if config.write_test_outputs and trial <= config.n_writes_test:
        heading("Running on the test set and writing the predictions")
        for task in tasks:
          # Currently only writing preds for GLUE and SQuAD 2.0 is supported
          if task.name in ["cola", "mrpc", "mnli", "sst", "rte", "qnli", "qqp",
                           "sts"]:
            for split in task.get_test_splits():
              model_runner.write_classification_outputs([task], trial, split)
          # elif task.name in ["biosses", "ddi", "chemprot", "hoc", "chem", "disease"]:
          #   results.append(model_runner.test())
          #   write_results(config, results, "test")
          elif task.name == "squad":
            scorer = model_runner.evaluate_task(task, "test", False)
            scorer.write_predictions()
            preds = utils.load_json(config.qa_preds_file("squad"))
            null_odds = utils.load_json(config.qa_na_file("squad"))
            for q, _ in preds.items():
              if null_odds[q] > config.qa_na_threshold:
                preds[q] = ""
            utils.write_json(preds, config.test_predictions(
                task.name, "test", trial))
          else:
            results.append(model_runner.test())
            write_results(config, results, "test")
            for split in task.get_test_splits():
              model_runner.write_tagging_outputs([task], trial, split)
            # utils.log("Skipping task", task.name,
            #           "- writing predictions is not supported for this task")

    if trial != config.num_trials and (not config.keep_all_models):
      utils.rmrf(config.model_dir)
    trial += 1


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--data-dir", required=True,
                      help="Location of data files (model weights, etc).")
  parser.add_argument("--model-name", required=True,
                      help="The name of the model being fine-tuned.")
  parser.add_argument("--init-checkpoint", required=False,
                      help="Init checkpoint.")
  parser.add_argument("--hparams", default="{}",
                      help="JSON dict of model hyperparameters.")
  args = parser.parse_args()
  if args.hparams.endswith(".json"):
    hparams = utils.load_json(args.hparams)
  else:
    hparams = json.loads(args.hparams)
  tf.logging.set_verbosity(tf.logging.ERROR)
  run_finetuning(configure_finetuning.FinetuningConfig(
      args.model_name, args.data_dir, args.init_checkpoint, **hparams))


if __name__ == "__main__":
  main()
