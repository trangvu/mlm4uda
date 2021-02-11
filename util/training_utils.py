# coding=utf-8
# Adapt from electra

"""Utilities for training the models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import re
import time
import tensorflow.compat.v1 as tf

from model import modeling
from util import utils


class ETAHook(tf.estimator.SessionRunHook):
  """Print out the time remaining during training/evaluation."""

  def __init__(self, to_log, n_steps, iterations_per_loop, on_tpu,
               log_every=1, is_training=True):
    self._to_log = to_log
    self._n_steps = n_steps
    self._iterations_per_loop = iterations_per_loop
    self._on_tpu = on_tpu
    self._log_every = log_every
    self._is_training = is_training
    self._steps_run_so_far = 0
    self._global_step = None
    self._global_step_tensor = None
    self._start_step = None
    self._start_time = None

  def begin(self):
    self._global_step_tensor = tf.train.get_or_create_global_step()

  def before_run(self, run_context):
    if self._start_time is None:
      self._start_time = time.time()
    return tf.estimator.SessionRunArgs(self._to_log)

  def after_run(self, run_context, run_values):
    self._global_step = run_context.session.run(self._global_step_tensor)
    self._steps_run_so_far += self._iterations_per_loop if self._on_tpu else 1
    if self._start_step is None:
      self._start_step = self._global_step - (self._iterations_per_loop
                                              if self._on_tpu else 1)
    self.log(run_values)

  def end(self, session):
    self._global_step = session.run(self._global_step_tensor)
    self.log()

  def log(self, run_values=None):
    step = self._global_step if self._is_training else self._steps_run_so_far
    if step % self._log_every != 0:
      return
    msg = "{:}/{:} = {:.1f}%".format(step, self._n_steps,
                                     100.0 * step / self._n_steps)
    time_elapsed = time.time() - self._start_time
    time_per_step = time_elapsed / (
        (step - self._start_step) if self._is_training else step)
    msg += ", SPS: {:.1f}".format(1 / time_per_step)
    msg += ", ELAP: " + secs_to_str(time_elapsed)
    msg += ", ETA: " + secs_to_str(
        (self._n_steps - step) * time_per_step)
    if run_values is not None:
      for tag, value in run_values.results.items():
        msg += " - " + str(tag) + (": {:.4f}".format(value))
    utils.log(msg)


def secs_to_str(secs):
  s = str(datetime.timedelta(seconds=int(round(secs))))
  s = re.sub("^0:", "", s)
  s = re.sub("^0", "", s)
  s = re.sub("^0:", "", s)
  s = re.sub("^0", "", s)
  return s


def get_bert_config(config):
  """Get model hyperparameters based on a pretraining/finetuning config"""
  if config.model_size == "large":
    args = {"hidden_size": 1024, "num_hidden_layers": 24}
  elif config.model_size == "base":
    args = {"hidden_size": 768, "num_hidden_layers": 12}
  elif config.model_size == "small":
    args = {"hidden_size": 256, "num_hidden_layers": 12}
  elif config.model_size == "mini":
    args = {"hidden_size": 256, "initializer_range": 0.02, "hidden_dropout_prob": 0.1, "num_attention_heads": 4,
            "num_hidden_layers": 4,
            "intermediate_size": 1024,
            "attention_probs_dropout_prob": 0.1}
  else:
    raise ValueError("Unknown model size", config.model_size)
  args["vocab_size"] = config.vocab_size
  args.update(**config.model_hparam_overrides)
  # by default the ff size and num attn heads are determined by the hidden size
  args["num_attention_heads"] = max(1, args["hidden_size"] // 64)
  args["intermediate_size"] = 4 * args["hidden_size"]
  args.update(**config.model_hparam_overrides)
  return modeling.BertConfig.from_dict(args)

def get_teacher_config(config):
  """Get model hyperparameters based on a pretraining/finetuning config"""
  if config.teacher_size == "large":
    args = {"hidden_size": 256, "num_hidden_layers": 2, "hidden_dropout_prob": 0.1}
  elif config.teacher_size == "base":
    args = {"hidden_size": 256, "num_hidden_layers": 2, "hidden_dropout_prob": 0.1}
  elif config.teacher_size == "small":
    args = {"hidden_size": 256, "num_hidden_layers": 2, "hidden_dropout_prob": 0.1}
  elif config.model_size == "mini":
    args = {"hidden_size": 128, "initializer_range": 0.02, "hidden_dropout_prob": 0.1, "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "intermediate_size": 1024,
            "attention_probs_dropout_prob": 0.1}
  else:
    raise ValueError("Unknown model size", config.model_size)
  args["vocab_size"] = config.vocab_size
  args.update(**config.model_hparam_overrides)
  # by default the ff size and num attn heads are determined by the hidden size
  args["num_attention_heads"] = max(1, args["hidden_size"] // 64)
  args["intermediate_size"] = 4 * args["hidden_size"]
  args.update(**config.model_hparam_overrides)
  return modeling.BertConfig.from_dict(args)
