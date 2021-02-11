# coding=utf-8
# @author: trangvu

"""Pre-trains dynamic and adversarial BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import json

import tensorflow
import tensorflow.compat.v1 as tf

import configure_pretraining
from model import modeling, gumbel, tokenization
from model import optimization
from pretrain import pretrain_data
from pretrain import pretrain_helpers
from pretrain.pretrain_data import Inputs
from pretrain.pretrain_helpers import scatter_update
from util import training_utils
from util import utils

from model.modeling import create_initializer, get_shape_list, dropout, layer_norm, get_activation


class PretrainingModel(object):
  """Transformer pre-training using the replaced-token-detection task."""

  def __init__(self, config: configure_pretraining.PretrainingConfig,
               features, is_training):
    # Set up model config
    self._config = config
    self._bert_config = training_utils.get_bert_config(config)
    # if config.debug:
    #   self._bert_config.num_hidden_layers = 3
    #   self._bert_config.hidden_size = 144
    #   self._bert_config.intermediate_size = 144 * 4
    #   self._bert_config.num_attention_heads = 4

    embedding_size = (
      self._bert_config.hidden_size if config.embedding_size is None else
      config.embedding_size)

    # Mask the input
    inputs = pretrain_data.features_to_inputs(features)
    proposal_distribution = 1.0
    if config.masking_strategy == pretrain_helpers.ENTROPY_STRATEGY:
      old_model = self._build_transformer(
        inputs, is_training,
        embedding_size = embedding_size)
      entropy_output = self._get_entropy_output(inputs, old_model)
      proposal_distribution = entropy_output.entropy

    masked_inputs = pretrain_helpers.mask(
      config, pretrain_data.features_to_inputs(features), config.mask_prob,
      proposal_distribution=proposal_distribution)

    # BERT model
    model = self._build_transformer(
        masked_inputs, is_training, reuse=tf.AUTO_REUSE,
        embedding_size = embedding_size)
    mlm_output = self._get_masked_lm_output(masked_inputs, model)
    self.total_loss = mlm_output.loss

    # Evaluation
    eval_fn_inputs = {
        "input_ids": masked_inputs.input_ids,
        "masked_lm_preds": mlm_output.preds,
        "mlm_loss": mlm_output.per_example_loss,
        "masked_lm_ids": masked_inputs.masked_lm_ids,
        "masked_lm_weights": masked_inputs.masked_lm_weights,
        "input_mask": masked_inputs.input_mask
    }
    eval_fn_keys = eval_fn_inputs.keys()
    eval_fn_values = [eval_fn_inputs[k] for k in eval_fn_keys]

    """Computes the loss and accuracy of the model."""
    d = {k: arg for k, arg in zip(eval_fn_keys, eval_fn_values)}
    metrics = dict()
    metrics["masked_lm_accuracy"] = tf.metrics.accuracy(
        labels=tf.reshape(d["masked_lm_ids"], [-1]),
        predictions=tf.reshape(d["masked_lm_preds"], [-1]),
        weights=tf.reshape(d["masked_lm_weights"], [-1]))
    metrics["masked_lm_loss"] = tf.metrics.mean(
        values=tf.reshape(d["mlm_loss"], [-1]),
        weights=tf.reshape(d["masked_lm_weights"], [-1]))
    self.eval_metrics = metrics

  def _get_entropy_output(self, inputs: pretrain_data.Inputs, model):
    """Masked language modeling softmax layer."""
    with tf.variable_scope("cls/predictions", reuse=tf.AUTO_REUSE):
      hidden = tf.layers.dense(
        model.get_sequence_output(),
        units=modeling.get_shape_list(model.get_embedding_table())[-1],
        activation=modeling.get_activation(self._bert_config.hidden_act),
        kernel_initializer=modeling.create_initializer(
          self._bert_config.initializer_range))
      hidden = modeling.layer_norm(hidden)
      output_bias = tf.get_variable(
        "output_bias",
        shape=[self._bert_config.vocab_size],
        initializer=tf.zeros_initializer())
      logits = tf.matmul(hidden, model.get_embedding_table(),
                         transpose_b=True)
      logits = tf.nn.bias_add(logits, output_bias)

      probs = tf.nn.softmax(logits)
      log_probs = tf.nn.log_softmax(logits)
      entropy = -tf.reduce_sum(log_probs * probs, axis=[2])

      EntropyOutput = collections.namedtuple(
        "EntropyOutput", ["logits", "probs", "log_probs", "entropy"])
      return EntropyOutput(
        logits=logits, probs=probs, log_probs=log_probs, entropy=entropy)

  def _get_masked_lm_output(self, inputs: pretrain_data.Inputs, model):
    """Masked language modeling softmax layer."""
    masked_lm_weights = inputs.masked_lm_weights
    with tf.variable_scope("cls/predictions", reuse=tf.AUTO_REUSE):
      relevant_hidden = pretrain_helpers.gather_positions(
          model.get_sequence_output(), inputs.masked_lm_positions)
      hidden = tf.layers.dense(
          relevant_hidden,
          units=modeling.get_shape_list(model.get_embedding_table())[-1],
          activation=modeling.get_activation(self._bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              self._bert_config.initializer_range))
      hidden = modeling.layer_norm(hidden)
      output_bias = tf.get_variable(
          "output_bias",
          shape=[self._bert_config.vocab_size],
          initializer=tf.zeros_initializer())
      logits = tf.matmul(hidden, model.get_embedding_table(),
                         transpose_b=True)
      logits = tf.nn.bias_add(logits, output_bias)

      oh_labels = tf.one_hot(
          inputs.masked_lm_ids, depth=self._bert_config.vocab_size,
          dtype=tf.float32)

      probs = tf.nn.softmax(logits)
      log_probs = tf.nn.log_softmax(logits)
      label_log_probs = -tf.reduce_sum(log_probs * oh_labels, axis=-1)

      numerator = tf.reduce_sum(inputs.masked_lm_weights * label_log_probs)
      denominator = tf.reduce_sum(masked_lm_weights) + 1e-6
      loss = numerator / denominator
      preds = tf.argmax(log_probs, axis=-1, output_type=tf.int32)

      MLMOutput = collections.namedtuple(
          "MLMOutput", ["logits", "probs", "loss", "per_example_loss", "preds"])
      return MLMOutput(
          logits=logits, probs=probs, per_example_loss=label_log_probs,
          loss=loss, preds=preds)

  def _build_transformer(self, inputs: pretrain_data.Inputs, is_training,
                         bert_config=None, name="bert", reuse=False, **kwargs):
    """Build a transformer encoder network."""
    if bert_config is None:
      bert_config = self._bert_config
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
      return modeling.BertModel(
          bert_config=bert_config,
          is_training=is_training,
          input_ids=inputs.input_ids,
          input_mask=inputs.input_mask,
          token_type_ids=inputs.segment_ids,
          use_one_hot_embeddings=self._config.use_tpu,
          scope=name,
          **kwargs)

class AdversarialPretrainingModel(PretrainingModel):

  def __init__(self, config: configure_pretraining.PretrainingConfig,
               features, is_training):
    # Set up model config
    self._config = config
    self._bert_config = training_utils.get_bert_config(config)
    self._teacher_config = training_utils.get_teacher_config(config)

    embedding_size = (
      self._bert_config.hidden_size if config.embedding_size is None else
      config.embedding_size)

    tokenizer = tokenization.FullTokenizer(
      config.vocab_file, do_lower_case=config.do_lower_case)
    self._vocab = tokenizer.vocab
    self._inv_vocab = tokenizer.inv_vocab

    # Mask the input
    inputs = pretrain_data.features_to_inputs(features)
    old_model = self._build_transformer(
        inputs, is_training,
        embedding_size=embedding_size)
    input_states = old_model.get_sequence_output()
    input_states = tf.stop_gradient(input_states)

    teacher_output = self._build_teacher(input_states, inputs, is_training, embedding_size=embedding_size)
      # calculate the proposal distribution

    action_prob = teacher_output.action_probs #pi(x_i)

    coin_toss = tf.random.uniform([])
    log_q, masked_inputs = self._sample_masking_subset(inputs, action_prob)
    if config.masking_strategy == pretrain_helpers.MIX_ADV_STRATEGY:
      random_masked_input = pretrain_helpers.mask(
        config, pretrain_data.features_to_inputs(features), config.mask_prob)
      B, L = modeling.get_shape_list(inputs.input_ids)
      N = config.max_predictions_per_seq
      strategy_prob = tf.random.uniform([B])
      strategy_prob = tf.expand_dims(tf.cast(tf.greater(strategy_prob, 0.5), tf.int32), 1)
      l_strategy_prob = tf.tile(strategy_prob, [1, L])
      n_strategy_prob = tf.tile(strategy_prob, [1, N])
      mix_input_ids = masked_inputs.input_ids * l_strategy_prob + random_masked_input.input_ids * (1 - l_strategy_prob)
      mix_masked_lm_positions = masked_inputs.masked_lm_positions * n_strategy_prob + random_masked_input.masked_lm_positions * (1-n_strategy_prob)
      mix_masked_lm_ids = masked_inputs.masked_lm_ids * n_strategy_prob + random_masked_input.masked_lm_ids * (1-n_strategy_prob)
      n_strategy_prob = tf.cast(n_strategy_prob, tf.float32)
      mix_masked_lm_weights = masked_inputs.masked_lm_weights * n_strategy_prob + random_masked_input.masked_lm_weights * (1-n_strategy_prob)
      mix_masked_inputs = pretrain_data.get_updated_inputs(
        inputs,
        input_ids=tf.stop_gradient(mix_input_ids),
        masked_lm_positions=mix_masked_lm_positions,
        masked_lm_ids=mix_masked_lm_ids,
        masked_lm_weights=mix_masked_lm_weights,
        tag_ids = inputs.tag_ids
      )
      masked_inputs = mix_masked_inputs

    # BERT model
    model = self._build_transformer(
      masked_inputs, is_training, reuse=tf.AUTO_REUSE,
      embedding_size=embedding_size)
    mlm_output = self._get_masked_lm_output(masked_inputs, model)
    self.total_loss = mlm_output.loss

    # Teacher reward is the -log p(x_S|x;B)
    reward = tf.stop_gradient(tf.reduce_mean(mlm_output.per_example_loss, 1))
    self._baseline = tf.reduce_mean(reward, -1)
    self._std = tf.math.reduce_std(reward, -1)

    # Calculate teacher loss
    def compute_teacher_loss(log_q, reward, baseline, std):
      advantage = tf.abs((reward - baseline) / std)
      advantage = tf.stop_gradient(advantage)
      log_q = tf.Print(log_q, [log_q], "log_q: ")
      teacher_loss = tf.reduce_mean(- log_q * advantage)
      return teacher_loss

    teacher_loss = tf.cond(coin_toss < 0.1, lambda: compute_teacher_loss(log_q, reward, self._baseline, self._std), lambda: tf.constant(0.0))
    self.total_loss = mlm_output.loss + teacher_loss
    self.teacher_loss = teacher_loss
    self.mlm_loss = mlm_output.loss

    # Evaluation`
    eval_fn_inputs = {
      "input_ids": masked_inputs.input_ids,
      "masked_lm_preds": mlm_output.preds,
      "mlm_loss": mlm_output.per_example_loss,
      "masked_lm_ids": masked_inputs.masked_lm_ids,
      "masked_lm_weights": masked_inputs.masked_lm_weights,
      "input_mask": masked_inputs.input_mask
    }
    eval_fn_keys = eval_fn_inputs.keys()
    eval_fn_values = [eval_fn_inputs[k] for k in eval_fn_keys]

    """Computes the loss and accuracy of the model."""
    d = {k: arg for k, arg in zip(eval_fn_keys, eval_fn_values)}
    metrics = dict()
    metrics["masked_lm_accuracy"] = tf.metrics.accuracy(
      labels=tf.reshape(d["masked_lm_ids"], [-1]),
      predictions=tf.reshape(d["masked_lm_preds"], [-1]),
      weights=tf.reshape(d["masked_lm_weights"], [-1]))
    metrics["masked_lm_loss"] = tf.metrics.mean(
      values=tf.reshape(d["mlm_loss"], [-1]),
      weights=tf.reshape(d["masked_lm_weights"], [-1]))
    self.eval_metrics = metrics

  def _build_teacher(self, states, inputs: pretrain_data.Inputs, is_training, name="teacher", reuse=False, **kwargs):
    """Build teacher network to estimate token score."""
    input_shape = get_shape_list(states, expected_rank=3)
    prev_output = states
    hidden_size = self._teacher_config.hidden_size
    num_hidden_layers = self._teacher_config.num_hidden_layers
    if is_training:
      hidden_dropout_prob = self._teacher_config.hidden_dropout_prob
    else:
      hidden_dropout_prob = 0.0
    with tf.variable_scope("teacher", reuse=reuse):
      for layer_idx in range(num_hidden_layers):
        with tf.variable_scope("layer_%d" % layer_idx):
          layer_input = prev_output
          layer_output = tf.layers.dense(
            layer_input,
            hidden_size,
            activation=get_activation("gelu"),
            kernel_initializer=create_initializer(self._teacher_config.initializer_range))
          layer_output = dropout(layer_output, hidden_dropout_prob)
          layer_output = layer_norm(layer_output)
          prev_output = layer_output

      sequence_output = prev_output
      with tf.variable_scope("bernoulli"):
        with tf.variable_scope("transform"):
          logits = tf.layers.dense(
            sequence_output,
            units=1,
            kernel_initializer=create_initializer(
              self._teacher_config.initializer_range))
          action_probs = tf.nn.sigmoid(logits)
          action_probs = tf.squeeze(action_probs)

      TeacherOutput = collections.namedtuple(
        "TeacherOutput", ["action_probs"])
    return TeacherOutput(
        action_probs=action_probs)

  def _sample_masking_subset(self, inputs: pretrain_data.Inputs, action_probs):
    #calculate shifted action_probs
    input_mask = inputs.input_mask
    segment_ids = inputs.segment_ids
    input_ids = inputs.input_ids

    shape = modeling.get_shape_list(input_ids, expected_rank=2)
    batch_size = shape[0]
    max_seq_len = shape[1]

    def _remove_special_token(elems):
      action_prob = tf.cast(elems[0], tf.float32)
      segment = tf.cast(elems[1], tf.int32)
      input = tf.cast(elems[2], tf.int32)
      mask = tf.cast(elems[3], tf.int32)

      seq_len = tf.reduce_sum(mask)
      seg1_len = seq_len - tf.reduce_sum(segment)
      seq1_idx = tf.range(start=1, limit=seg1_len-1, dtype = tf.int32)
      seq2_limit = tf.math.maximum(seg1_len, seq_len - 1)
      seq2_idx = tf.range(start=seg1_len, limit=seq2_limit, dtype = tf.int32)
      mask_idx = tf.range(start=seq_len, limit=max_seq_len, dtype = tf.int32)
      index_tensor = tf.concat([seq1_idx, seq2_idx, mask_idx], axis = 0)

      seq1_prob = action_prob[1:seg1_len - 1]
      seq2_prob = action_prob[seg1_len:seq2_limit]
      mask_prob = tf.ones_like(mask_idx, dtype=tf.float32) * 1e-20
      cleaned_action_prob = tf.concat([seq1_prob, seq2_prob, mask_prob], axis = 0)
      cleaned_mask = tf.concat([mask[1:seg1_len -1], mask[seg1_len:seq_len - 1],
                                mask[seq_len:max_seq_len]], axis = 0)

      cleaned_input = tf.concat([input[1:seg1_len -1], input[seg1_len:seq_len - 1],
                                 input[seq_len:max_seq_len ]], axis = 0)

      cleaned_action_prob = cleaned_action_prob[0:max_seq_len - 3]
      index_tensor = index_tensor[0:max_seq_len - 3]
      cleaned_input = cleaned_input[0:max_seq_len - 3]
      cleaned_mask = cleaned_mask[0:max_seq_len - 3]

      return (cleaned_action_prob, index_tensor, cleaned_input, cleaned_mask)


    # Remove CLS and SEP action probs
    elems = tf.stack([action_probs, tf.cast(segment_ids, tf.float32),
                      tf.cast(input_ids, tf.float32), tf.cast(input_mask, tf.float32)], 1)
    cleaned_action_probs, index_tensors, cleaned_inputs, cleaned_input_mask = tf.map_fn(_remove_special_token,
                                                                                        elems,
                                                              dtype=(tf.float32, tf.int32, tf.int32, tf.int32),
                                                              parallel_iterations=1)
    logZ, log_prob = self._calculate_partition_table(cleaned_input_mask, cleaned_action_probs,
                                               self._config.max_predictions_per_seq)

    samples, log_q = self._sampling_a_subset(logZ, log_prob, self._config.max_predictions_per_seq)

    # Collect masked_lm_ids and masked_lm_positions
    zero_values = tf.zeros_like(index_tensors, tf.int32)
    selected_position = tf.where(tf.equal(samples, 1), index_tensors, zero_values)
    masked_lm_positions, _ = tf.nn.top_k(selected_position, self._config.max_predictions_per_seq, sorted=False)

    # Get the ids of the masked-out tokens
    shift = tf.expand_dims(max_seq_len * tf.range(batch_size), -1)
    flat_positions = tf.reshape(masked_lm_positions + shift, [-1, 1])
    masked_lm_ids = tf.gather_nd(tf.reshape(input_ids, [-1]),
                                 flat_positions)
    masked_lm_ids = tf.reshape(masked_lm_ids, [batch_size, -1])

    # Update the input ids
    replaced_prob = tf.random.uniform([batch_size, self._config.max_predictions_per_seq])
    replace_with_mask_positions = masked_lm_positions * tf.cast(
      tf.less(replaced_prob, 0.85), tf.int32)
    inputs_ids, _ = scatter_update(
      inputs.input_ids, tf.fill([batch_size, self._config.max_predictions_per_seq], self._vocab["[MASK]"]),
      replace_with_mask_positions)

    # Replace with random tokens
    replace_with_random_positions = masked_lm_positions * tf.cast(
      tf.greater(replaced_prob, 0.925), tf.int32)
    random_tokens = tf.random.uniform([batch_size, self._config.max_predictions_per_seq],
                                      minval=0, maxval=len(self._vocab), dtype=tf.int32)

    inputs_ids, _ = scatter_update(
      inputs_ids, random_tokens,
      replace_with_random_positions)

    masked_lm_weights = tf.ones_like(masked_lm_ids, tf.float32)
    inv_vocab = self._inv_vocab
    # Apply mask on input
    if self._config.debug:
      def pretty_print(inputs_ids, masked_lm_ids, masked_lm_positions, masked_lm_weights, tag_ids):
        debug_inputs = Inputs(
          input_ids=inputs_ids,
          input_mask=None,
          segment_ids=None,
          masked_lm_positions=masked_lm_positions,
          masked_lm_ids=masked_lm_ids,
          masked_lm_weights=masked_lm_weights,
          tag_ids=tag_ids)
        pretrain_data.print_tokens(debug_inputs, inv_vocab)

        ## TODO: save to the mask choice
        return inputs_ids, masked_lm_ids, masked_lm_positions, masked_lm_weights

      mask_shape = masked_lm_ids.get_shape()
      inputs_ids, masked_lm_ids, masked_lm_positions, masked_lm_weights = \
        tf.py_func(pretty_print, [inputs_ids, masked_lm_ids, masked_lm_positions, masked_lm_weights, inputs.tag_ids],
                   (tf.int32, tf.int32, tf.int32, tf.float32))
      inputs_ids.set_shape(inputs.input_ids.get_shape())
      masked_lm_ids.set_shape(mask_shape)
      masked_lm_positions.set_shape(mask_shape)
      masked_lm_weights.set_shape(mask_shape)

    masked_input = pretrain_data.get_updated_inputs(
      inputs,
      input_ids=tf.stop_gradient(input_ids),
      masked_lm_positions=tf.stop_gradient(masked_lm_positions),
      masked_lm_ids=tf.stop_gradient(masked_lm_ids),
      masked_lm_weights=tf.stop_gradient(masked_lm_weights),
      tag_ids=inputs.tag_ids)

    return log_q, masked_input

  def _calculate_partition_table(self, input_mask, action_prob, max_predictions_per_seq):
    shape = modeling.get_shape_list(action_prob, expected_rank=2)
    seq_len = shape[1]

    with tf.variable_scope("teacher/dp"):
      '''
      Calculate DP table: aims to calculate logZ[0,K]
      # We add an extra row so that when we calculate log_q_yes, we don't have out of bound error
      # Z[b,N+1,k] = log 0 - we do not allow to choose anything
      # logZ size batch_size x N+1 x K+1
      '''
      initZ = tf.TensorArray(tf.float32, size=max_predictions_per_seq + 1)
      logZ_0 = tf.zeros_like(input_mask, dtype=tf.float32)  # size b x N
      logZ_0 = tf.pad(logZ_0, [[0, 0], [0, 1]], "CONSTANT")  # size b x N+1
      initZ = initZ.write(tf.constant(0), logZ_0)

      # mask logp
      action_prob = tf.cast(input_mask, dtype=tf.float32) * action_prob
      action_prob = tf.clip_by_value(action_prob, 1e-20, 1.0)
      logp = tf.log(action_prob)
      logp_no = tf.log(1-action_prob)
      accum_logp = tf.cumsum(logp, axis=1, reverse=True)

      def accum_cond(j, logZ_j, logb, loga):
        return tf.greater(j, -1)

      def accum_body(j, logZ_j, logb, loga):
        # logb: log_yes = logp[j] + logZ[j+1, k-1] -- already compute
        # loga: log_no = log(1-p[j]) + logZ[j+1, k]
        logb_j = tf.squeeze(logb[:, j])
        log_one_minus_p_j = tf.squeeze(logp_no[:, j])
        loga = loga + log_one_minus_p_j
        next_logZ_j = tf.math.reduce_logsumexp(tf.stack([loga, logb_j]), 0)
        logZ_j = logZ_j.write(j, next_logZ_j)
        return [tf.subtract(j, 1), logZ_j, logb, next_logZ_j]

      def dp_loop_cond(k, logZ, lastZ):
        return tf.less(k, max_predictions_per_seq + 1)

      def dp_body(k, logZ, lastZ):
        '''
        case j < N-k + 1:
          logZ[j,k] = log_sum( log(1-pi(j)) + logZ[j+1,k], logp(j) + logZ[j+1,k-1])
        case j = N-k + 1
          logZ[j,k] = accum_logp[j]
        case j > N-k + 1
          logZ[j,k] = 0
        '''

        # shift lastZ one step
        shifted_lastZ = tf.roll(lastZ[:, :-1], shift=1, axis=1) #logZ[j+1,k-1]
        log_yes = logp + shifted_lastZ  # b x N
        logZ_j = tf.TensorArray(tf.float32, size=seq_len + 1)
        init_value = accum_logp[:, seq_len - k]
        logZ_j = logZ_j.write(seq_len - k, init_value)
        _, logZ_j, logb, loga = tf.while_loop(accum_cond, accum_body, [seq_len - k - 1, logZ_j, log_yes, init_value])
        logZ_j = logZ_j.stack()  # N x b
        logZ_j = tf.transpose(logZ_j, [1, 0])  # b x N
        logZ = logZ.write(k, logZ_j)
        return [tf.add(k, 1), logZ, logZ_j]

      k = tf.constant(1)
      _, logZ, lastZ = tf.while_loop(dp_loop_cond, dp_body,
                                     [k, initZ, logZ_0],
                                     shape_invariants=[k.get_shape(), tf.TensorShape([]),
                                                       tf.TensorShape([None, None])])
      logZ = logZ.stack()  # N x b x N
      logZ = tf.transpose(logZ, [1, 2, 0])
    return logZ, logp

  def _sampling_a_subset(self,logZ, logp, max_predictions_per_seq):
    shape = modeling.get_shape_list(logp, expected_rank=2)
    seq_len = shape[1]

    def gather_z_indexes(sequence_tensor, positions):
      """Gathers the vectors at the specific positions over a minibatch."""
      # set negative indices to zeros
      mask = tf.zeros_like(positions, dtype=tf.int32)
      masked_position = tf.reduce_max(tf.stack([positions, mask]), 0)

      index = tf.reshape(tf.cast(tf.where(tf.equal(mask, 0)), dtype=tf.int32), [-1])
      flat_offsets = index * (max_predictions_per_seq + 1)
      flat_positions = masked_position + flat_offsets
      flat_sequence_tensor = tf.reshape(sequence_tensor, [-1])
      output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
      return output_tensor

    def sampling_loop_cond(j, subset, count, left, log_q):
      # j < N and left > 0
      # we want to exclude last tokens, because it's always a special token [SEP]
      return tf.logical_or(tf.less(j, seq_len), tf.greater(tf.reduce_sum(left), 0))

    def sampling_body(j, subset, count, left, log_q):
      # calculate log_q_yes and log_q_no
      logp_j = logp[:, j]
      log_Z_total = gather_z_indexes(logZ[:, j, :], left)  # b
      log_Z_yes = gather_z_indexes(logZ[:, j + 1, :], left - 1)  # b
      logit_yes = logp_j + log_Z_yes - log_Z_total
      logit_no = tf.log(tf.clip_by_value(1 - tf.exp(logit_yes), 1e-20, 1.0))
      # draw 2 Gumbel noise and compute action by argmax
      logits = tf.transpose(tf.stack([logit_no, logit_yes]), [1, 0])
      actions = gumbel.gumbel_softmax(logits)
      action_mask = tf.cast(tf.argmax(actions, 1), dtype=tf.int32)
      no_left_mask = tf.where(tf.greater(left, 0), tf.ones_like(left, dtype=tf.int32),
                              tf.zeros_like(left, dtype=tf.int32))
      output = action_mask * no_left_mask
      actions = tf.reduce_max(actions, 1)
      log_actions = tf.log(actions)
      # compute log_q_j and update count and subset
      count = count + output
      left = left - output
      log_q = log_q + log_actions
      subset = subset.write(j, output)

      return [tf.add(j, 1), subset, count, left, log_q]

    with tf.variable_scope("teacher/sampling"):
      # Batch sampling
      subset = tf.TensorArray(tf.int32, size=seq_len)
      count = tf.zeros_like(logp[:, 0], dtype=tf.dtypes.int32)
      left = tf.ones_like(logp[:, 0], dtype=tf.dtypes.int32)
      left = left * max_predictions_per_seq
      log_q = tf.zeros_like(count, dtype=tf.dtypes.float32)

      _, subset, count, left, log_q = tf.while_loop(sampling_loop_cond, sampling_body,
                                                    [tf.constant(0), subset, count, left, log_q])

      subset = subset.stack()  # K x b x N
      subset = tf.transpose(subset, [1, 0])
    return subset, log_q

class RatioBasedPretrainingModel(AdversarialPretrainingModel):
  def __init__(self, config: configure_pretraining.PretrainingConfig,
               features, ratio, is_training):
    # Set up model config
    self._config = config
    self._bert_config = training_utils.get_bert_config(config)


    embedding_size = (
      self._bert_config.hidden_size if config.embedding_size is None else
      config.embedding_size)

    tokenizer = tokenization.FullTokenizer(
      config.vocab_file, do_lower_case=config.do_lower_case)
    self._vocab = tokenizer.vocab
    self._inv_vocab = tokenizer.inv_vocab

    # Mask the input
    inputs = pretrain_data.features_to_inputs(features)
    # Load ratio
    with tf.variable_scope("rw_masking"):
      with tf.variable_scope("ratio"):
        self.ratios = tf.constant(ratio)
        action_prob = tf.nn.embedding_lookup(self.ratios, inputs.input_ids)

    log_q, masked_inputs = self._sample_masking_subset(inputs, action_prob)

    # BERT model
    model = self._build_transformer(
      masked_inputs, is_training, reuse=tf.AUTO_REUSE,
      embedding_size=embedding_size)
    mlm_output = self._get_masked_lm_output(masked_inputs, model)
    self.total_loss = mlm_output.loss

    # Evaluation`
    eval_fn_inputs = {
      "input_ids": masked_inputs.input_ids,
      "masked_lm_preds": mlm_output.preds,
      "mlm_loss": mlm_output.per_example_loss,
      "masked_lm_ids": masked_inputs.masked_lm_ids,
      "masked_lm_weights": masked_inputs.masked_lm_weights,
      "input_mask": masked_inputs.input_mask
    }
    eval_fn_keys = eval_fn_inputs.keys()
    eval_fn_values = [eval_fn_inputs[k] for k in eval_fn_keys]

    """Computes the loss and accuracy of the model."""
    d = {k: arg for k, arg in zip(eval_fn_keys, eval_fn_values)}
    metrics = dict()
    metrics["masked_lm_accuracy"] = tf.metrics.accuracy(
      labels=tf.reshape(d["masked_lm_ids"], [-1]),
      predictions=tf.reshape(d["masked_lm_preds"], [-1]),
      weights=tf.reshape(d["masked_lm_weights"], [-1]))
    metrics["masked_lm_loss"] = tf.metrics.mean(
      values=tf.reshape(d["mlm_loss"], [-1]),
      weights=tf.reshape(d["masked_lm_weights"], [-1]))
    self.eval_metrics = metrics


def model_fn_builder(config: configure_pretraining.PretrainingConfig):
  """Build the model for training."""

  def model_fn(features, labels, mode, params):
    """Build the model for training."""
    if config.masking_strategy == pretrain_helpers.ADVERSARIAL_STRATEGY or config.masking_strategy == pretrain_helpers.MIX_ADV_STRATEGY:
      model = AdversarialPretrainingModel(config, features,
                               mode == tf.estimator.ModeKeys.TRAIN)
    elif config.masking_strategy == pretrain_helpers.RW_STRATEGY:
      ratio = []
      with open(config.ratio_file, "r") as fin:
        for line in fin:
          line = line.strip()
          if line:
            tok = line.split()
            ratio.append(float(tok[1]))
      model = RatioBasedPretrainingModel(config, features, ratio,
                                          mode == tf.estimator.ModeKeys.TRAIN)
    else:
      model = PretrainingModel(config, features,
                             mode == tf.estimator.ModeKeys.TRAIN)
    utils.log("Model is built!")

    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    if config.init_checkpoint:
      (assignment_map, initialized_variable_names
       ) = modeling.get_assignment_map_from_checkpoint(tvars, config.init_checkpoint)
      tf.train.init_from_checkpoint(config.init_checkpoint, assignment_map)

    utils.log("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      utils.log("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    if mode == tf.estimator.ModeKeys.TRAIN:
      if config.masking_strategy == pretrain_helpers.ADVERSARIAL_STRATEGY:
        student_train_op = optimization.create_optimizer(
          model.mlm_loss, config.learning_rate, config.num_train_steps,
          weight_decay_rate=config.weight_decay_rate,
          use_tpu=config.use_tpu,
          warmup_steps=config.num_warmup_steps,
          lr_decay_power=config.lr_decay_power)
        teacher_train_op = optimization.create_optimizer(
          model.teacher_loss, config.teacher_learning_rate, config.num_train_steps,
          lr_decay_power=config.lr_decay_power)
        train_op = tf.group(student_train_op, teacher_train_op)
        output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=model.total_loss,
          train_op=train_op,
          training_hooks=[training_utils.ETAHook(dict(loss=model.mlm_loss, teacher_loss=model.teacher_loss,
                                                      reward=model._baseline),
                                                 config.num_train_steps, config.iterations_per_loop,
                                                 config.use_tpu)])
      else:
        train_op = optimization.create_optimizer(
          model.total_loss, config.learning_rate, config.num_train_steps,
          weight_decay_rate=config.weight_decay_rate,
          use_tpu=config.use_tpu,
          warmup_steps=config.num_warmup_steps,
          lr_decay_power=config.lr_decay_power
        )
        output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=model.total_loss,
          train_op=train_op,
          training_hooks=[training_utils.ETAHook(dict(loss=model.total_loss),
              config.num_train_steps, config.iterations_per_loop,
              config.use_tpu)])
    elif mode == tf.estimator.ModeKeys.EVAL:
      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=model.total_loss,
          eval_metric_ops=model.eval_metrics,
          evaluation_hooks=[training_utils.ETAHook(dict(loss=model.total_loss),
              config.num_eval_steps, config.iterations_per_loop,
              config.use_tpu, is_training=False)])
    else:
      raise ValueError("Only TRAIN and EVAL modes are supported")
    return output_spec

  return model_fn


def train_or_eval(config: configure_pretraining.PretrainingConfig):
  """Run pre-training or evaluate the pre-trained model."""
  if config.do_train == config.do_eval:
    raise ValueError("Exactly one of `do_train` or `do_eval` must be True.")
  if config.debug:
    utils.rmkdir(config.model_dir)
  utils.heading("Config:")
  utils.log_config(config)

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
      gpu_options=tf.GPUOptions(allow_growth = True))

    run_config = tf.estimator.RunConfig(
      model_dir=config.model_dir,
      save_checkpoints_steps=config.save_checkpoints_steps,
      train_distribute=train_distribution_strategy,
      eval_distribute=eval_distribution_strategy,
      # save_checkpoints_secs=3600,
      # tf_random_seed=FLAGS.seed,
      session_config= session_config,
      # keep_checkpoint_max=0,
      log_step_count_steps=100
    )

  model_fn = model_fn_builder(config=config)
  estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      config=run_config,
      params={'train_batch_size': config.train_batch_size,
              'eval_batch_size': config.eval_batch_size})

  if config.do_train:
    utils.heading("Running training")
    estimator.train(input_fn=pretrain_data.get_input_fn(config, True),
                    max_steps=config.num_train_steps)
  if config.do_eval:
    utils.heading("Running evaluation")
    result = estimator.evaluate(
        input_fn=pretrain_data.get_input_fn(config, False),
        steps=config.num_eval_steps)
    for key in sorted(result.keys()):
      utils.log("  {:} = {:}".format(key, str(result[key])))
    return result


def train_one_step(config: configure_pretraining.PretrainingConfig):
  """Builds an ELECTRA model an trains it for one step; useful for debugging."""
  train_input_fn = pretrain_data.get_input_fn(config, True)
  features = tf.data.make_one_shot_iterator(train_input_fn(dict(
      batch_size=config.train_batch_size))).get_next()
  model = PretrainingModel(config, features, True)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    utils.log(sess.run(model.total_loss))

# def adversarial_train(config: configure_pretraining.PretrainingConfig, estimator):
#   # Training loop for adversarial MLM
#   num_step = 0
#   for step in range(config.num_train_steps):
#
#     estimator.train(input_fn=pretrain_data.get_input_fn(config, True),
#                   max_steps=config.num_train_steps)

def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--data-dir", required=True,
                      help="Location of models (model weights, etc).")
  parser.add_argument("--model-name", required=True,
                      help="The name of the model being fine-tuned.")
  parser.add_argument("--hparams", default="{}",
                      help="JSON dict of model hyperparameters.")
  args = parser.parse_args()
  if args.hparams.endswith(".json"):
    hparams = utils.load_json(args.hparams)
  else:
    hparams = json.loads(args.hparams)
  tf.logging.set_verbosity(tf.logging.ERROR)
  train_or_eval(configure_pretraining.PretrainingConfig(
      args.model_name, args.data_dir, **hparams))


if __name__ == "__main__":
  main()
