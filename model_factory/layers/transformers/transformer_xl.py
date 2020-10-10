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

""" model interface
"""

from typing import Any, Tuple, List, Dict
import tensorflow as tf
from model_factory.layers.attentions.multihead_attention import RelativeMultiHeadAttention
from model_factory.layers.transformers.transformer import AddNorm


class TransformerXL(tf.keras.layers.Layer):
    def __init__(self, num_heads, head_size, units, activation, hidden_size, dropout_prob,
                 layer_norm_eps, kernel_initializer='glorot_uniform', **kwargs):
        super().__init__(**kwargs)
        # @TODO add MultiHeadAttention
        self.attention = RelativeMultiHeadAttention(num_heads=num_heads, head_size=head_size, name="attention")
        self.add_norm_attention = AddNorm(hidden_size, dropout_prob, layer_norm_eps, kernel_initializer)
        self.add_norm_out = AddNorm(hidden_size, dropout_prob, layer_norm_eps, kernel_initializer)
        self.forward = tf.keras.layers.Dense(units, activation=activation, kernel_initializer=kernel_initializer)

    def call(self, inputs, rel_pos_embedding, mems, attention_mask=None, is_output_attentions=False, training=False):
        # TODO add convert attention_mask from (B,Sq) --> (B,H,Sq,Sq)
        attention_outputs = self.attention(inputs, rel_pos_embedding, mems, attention_mask, is_output_attentions, training=training)
        attention_output = attention_outputs[0]
        attention_output = self.add_norm_attention(attention_output, inputs, training=training)
        intermediate_output = self.forward(attention_output)
        layer_output = self.add_norm_out(intermediate_output, attention_output, training=training)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs
