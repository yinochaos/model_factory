#!/bin/env python
# -*- coding: UTF-8 -*-
# Copyright 2020 yinochaos <pspcxl@163.com>. All Rights Reserved.
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
# ==============================================================================
from collections import namedtuple
import subprocess
import os
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.keras.saving.hdf5_format import preprocess_weights_for_loading
from tensorflow.python.keras.saving.hdf5_format import load_attributes_from_hdf5_group
from tensorflow.python.keras.saving.hdf5_format import _legacy_weights
from tensorflow.python.keras.saving import saving_utils
#from tensorflow.python.keras.engine.training import _detect_save_format



class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """

        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            ret = self.worst_score >= cur_score
            return ret


"""
def load_attributes_from_hdf5_group(group, name):
  Loads attributes of the specified name from the HDF5 group.
  This method deals with an inherent problem
  of HDF5 file which is not able to store
  data larger than HDF5_OBJECT_HEADER_LIMIT bytes.
  Args:
      group: A pointer to a HDF5 group.
      name: A name of the attributes to load.
  Returns:
      data: Attributes data.
  if name in group.attrs:
    data = [
        n.decode('utf8') if hasattr(n, 'decode') else n
        for n in group.attrs[name]
    ]
  else:
    data = []
    chunk_id = 0
    while '%s%d' % (name, chunk_id) in group.attrs:
      data.extend([
          n.decode('utf8') if hasattr(n, 'decode') else n
          for n in group.attrs['%s%d' % (name, chunk_id)]
      ])
      chunk_id += 1
  return data

def _legacy_weights(layer):
  For legacy reason, the layer.weights was in the order of
  [self.trainable_weights + self.non_trainable_weights], and this order was
  used for preserving the weights in h5 format. The new order of layer.weights
  are the same as layer.get_weights() which is more intuitive for user. To
  keep supporting the existing saved h5 file, this method should be used to
  save/load weights. In future version, we will delete this method and
  introduce a breaking change for h5 and stay with the new order for weights.
  Args:
    layer: a `tf.keras.Model` or `tf.keras.layers.Layer` instance.
  Returns:
    A list of variables with the order of trainable_weights, followed by
      non_trainable_weights.
  weights = layer.trainable_weights + layer.non_trainable_weights
  if any(not isinstance(w, variables_module.Variable) for w in weights):
    raise NotImplementedError(
        'Save or restore weights that is not an instance of `tf.Variable` is '
        'not supported in h5, use `save_format=\'tf\'` instead. Got a model '
        'or layer {} with weights {}'.format(layer.__class__.__name__, weights))
  return weights
"""

def load_cpkt_mapping_weights(model, checkpoint, name_mapping, skip_mismatch=False):
    def load_variable(checkpoint, name):
        if isinstance(checkpoint, dict):
            return checkpoint[name]
        else:
            return tf.train.load_variable(checkpoint, name)
    mapping = {}
    for k, v in name_mapping.items():
        if k in model.layers:
            mapping[k] = v
        else:
            logging.warning('layer ' + k + ' not in model')
    weight_value_pairs = []
    for layer, variables in mapping.items():
        layer = model.layers[layer]
        weights, values = [], []

        for w, v in zip(layer.trainable_weights, variables):  # 允许跳过不存在的权重
            try:
                values.append(load_variable(checkpoint, v))
                weights.append(w)
            except Exception as e:
                if skip_mismatch:
                    print('%s, but ignored.' % str(e))
                else:
                    raise e

        weight_value_pairs.extend(zip(weights, values))
    backend.batch_set_value(weight_value_pairs)


def load_h5_mapping_weights(model, filepath, name_mapping, skip_mismatch):

    #filepath, save_format = _detect_save_format(filepath)
    #if save_format == 'tf':
    #    raise NotImplementedError(
    #        'Weights may only be loaded based on topology into Models when '
    #        'loading TensorFlow-formatted weights (got by_name=True to '
    #        'load_weights).')
    if h5py is None:
        raise ImportError(
            '`load_weights` requires h5py when loading weights from HDF5.')
    if not model._is_graph_network and not model.built:
        raise ValueError(
            'Unable to load weights saved in HDF5 format into a subclassed '
            'Model which has not created its variables yet. Call the Model '
            'first, then load the weights.')
    model._assert_weights_created()
    with h5py.File(filepath, 'r') as f:
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']
    load_weights_from_hdf5_group_by_name_mapping(f, model.layers, name_mapping, skip_mismatch=skip_mismatch)


def load_weights_from_hdf5_group_by_name_mapping(f, layers, name_mapping, skip_mismatch=False):
    """Implements name-based weight loading.
    (instead of topological weight loading).
    Layers that have no matching name are skipped.
    Args:
        f: A pointer to a HDF5 group.
        layers: a list of target layers.
        name_mapping : name mapping dict
        skip_mismatch: Boolean, whether to skip loading of layers
            where there is a mismatch in the number of weights,
            or a mismatch in the shape of the weights.
    Raises:
        ValueError: in case of mismatch between provided layers
            and weights file and skip_match=False.
    """
    if 'keras_version' in f.attrs:
        original_keras_version = f.attrs['keras_version']
        if hasattr(original_keras_version, 'decode'):
            original_keras_version = original_keras_version.decode('utf8')
    else:
        original_keras_version = '1'
    if 'backend' in f.attrs:
        original_backend = f.attrs['backend']
        if hasattr(original_backend, 'decode'):
            original_backend = original_backend.decode('utf8')
    else:
        original_backend = None

    # New file format.
    layer_names = load_attributes_from_hdf5_group(f, 'layer_names')

    # Reverse index of layer name to list of layers with name.
    index = {}
    for layer in layers:
        if layer.name:
            index.setdefault(layer.name, []).append(layer)

    # We batch weight value assignments in a single backend call
    # which provides a speedup in TensorFlow.
    weight_value_tuples = []
    for k, name in enumerate(layer_names):
        g = f[name]
        weight_names = load_attributes_from_hdf5_group(g, 'weight_names')
        weight_values = [np.asarray(g[weight_name]) for weight_name in weight_names]

        for layer in index.get(name, []):
            symbolic_weights = _legacy_weights(layer)
            weight_values = preprocess_weights_for_loading(
                layer, weight_values, original_keras_version, original_backend)
            if len(weight_values) != len(symbolic_weights):
                if skip_mismatch:
                    logging.warning('Skipping loading of weights for '
                                    'layer {}'.format(layer.name) + ' due to mismatch '
                                    'in number of weights ({} vs {}).'.format(
                                        len(symbolic_weights), len(weight_values)))
                    continue
                raise ValueError('Layer #' + str(k) + ' (named "' + layer.name +
                                 '") expects ' + str(len(symbolic_weights)) +
                                 ' weight(s), but the saved weights' + ' have ' +
                                 str(len(weight_values)) + ' element(s).')
            # Set values.
            for i in range(len(weight_values)):
                if backend.int_shape(symbolic_weights[i]) != weight_values[i].shape:
                    if skip_mismatch:
                        logging.warning('Skipping loading of weights for '
                                        'layer {}'.format(layer.name) + ' due to '
                                        'mismatch in shape ({} vs {}).'.format(
                                            symbolic_weights[i].shape,
                                            weight_values[i].shape))
                        continue
                    raise ValueError('Layer #' + str(k) + ' (named "' + layer.name +
                                     '"), weight ' + str(symbolic_weights[i]) +
                                     ' has shape {}'.format(backend.int_shape(
                                         symbolic_weights[i])) +
                                     ', but the saved weight has shape ' +
                                     str(weight_values[i].shape) + '.')

                else:
                    weight_value_tuples.append((symbolic_weights[i], weight_values[i]))
    backend.batch_set_value(weight_value_tuples)
