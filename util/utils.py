# coding=utf-8
# Adapt from electra

"""A collection of general utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import pickle
import sys

import tensorflow.compat.v1 as tf


def load_json(path):
  with tf.io.gfile.GFile(path, "r") as f:
    return json.load(f)


def write_json(o, path):
  if "/" in path:
    tf.io.gfile.makedirs(path.rsplit("/", 1)[0])
  with tf.io.gfile.GFile(path, "w") as f:
    json.dump(o, f)


def load_pickle(path):
  with tf.io.gfile.GFile(path, "rb") as f:
    return pickle.load(f)


def write_pickle(o, path):
  if "/" in path:
    tf.io.gfile.makedirs(path.rsplit("/", 1)[0])
  with tf.io.gfile.GFile(path, "wb") as f:
    pickle.dump(o, f, -1)


def mkdir(path):
  if not tf.io.gfile.exists(path):
    tf.io.gfile.makedirs(path)


def rmrf(path):
  if tf.io.gfile.exists(path):
    tf.io.gfile.rmtree(path)


def rmkdir(path):
  rmrf(path)
  mkdir(path)


def log(*args):
  msg = " ".join(map(str, args))
  sys.stdout.write(msg + "\n")
  sys.stdout.flush()

def logerr(*args):
  msg = " ".join(map(str, args))
  sys.stderr.write(msg + "\n")
  sys.stderr.flush()

def log_config(config):
  for key, value in sorted(config.__dict__.items()):
    log(key, value)
  log()


def heading(*args):
  log(80 * "=")
  log(*args)
  log(80 * "=")


def nest_dict(d, prefixes, delim="_"):
  """Go from {prefix_key: value} to {prefix: {key: value}}."""
  nested = {}
  for k, v in d.items():
    for prefix in prefixes:
      if k.startswith(prefix + delim):
        if prefix not in nested:
          nested[prefix] = {}
        nested[prefix][k.split(prefix + delim, 1)[1]] = v
      else:
        nested[k] = v
  return nested


def flatten_dict(d, delim="_"):
  """Go from {prefix: {key: value}} to {prefix_key: value}."""
  flattened = {}
  for k, v in d.items():
    if isinstance(v, dict):
      for k2, v2 in v.items():
        flattened[k + delim + k2] = v2
    else:
      flattened[k] = v
  return flattened

def get_available_gpus():
  from tensorflow.python.client import device_lib
  gpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
  if len(gpus) == 0: return ["/cpu:0"]
  tf.logging.info('Availble GPUs: {}'.format(', '.join(gpus)))
  return gpus