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
from model_factory.layers.transformers.transformer import Transformer
from model_factory.layers.embeddings.pos_embedding import TransformerEmbedding


class Bert(tf.keras.layers.Layer):
    def __init__(self, vocab_size, emb_hidden_size, max_position_embeddings, type_vocab_size, shared_hiden_size,
                 num_layers, num_heads, head_size, units, activation, hidden_size, dropout_prob,
                 layer_norm_eps, is_output_hidden_states, is_output_attentions, kernel_initializer='glorot_uniform', **kwargs):
        super().__init__(**kwargs)
        self.embedding_layer = TransformerEmbedding(vocab_size, emb_hidden_size, max_position_embeddings,
                                                    type_vocab_size, shared_hiden_size, layer_norm_eps, dropout_prob, kernel_initializer)
        self.transformer_layers = [
            Transformer(num_heads, head_size, units, activation, hidden_size, dropout_prob, layer_norm_eps, kernel_initializer,
                        name="layer_%d" % (i)) for i in range(num_layers)
        ]
        self.is_output_attentions = is_output_attentions
        self.is_output_hidden_states = is_output_hidden_states
        self.dense = tf.keras.layers.Dense(
            hidden_size,
            kernel_initializer=kernel_initializer,
            activation="tanh",
            name="dense",
        )

    def call(self, inputs, attention_mask=None, training=False):
        """
        if isinstance(inputs, (tuple, list)):
            input_ids = inputs[0]
            position_ids = inputs[1]
            token_type_ids = inputs[2] if len(inputs) > 2 else None
        else:
            raise NotImplementedError
        """
        hidden_states = self.embedding_layer(inputs, training=training)
        all_hidden_states = () if self.is_output_hidden_states else None
        all_attentions = () if self.is_output_attentions else None

        for layer in self.transformer_layers:
            if self.is_output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer(hidden_states, attention_mask, self.is_output_attentions, training=training)
            hidden_states = layer_outputs[0]

            if self.is_output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        # Add last layer
        if self.is_output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        first_token_tensor = hidden_states[:, 0]
        pooler_output = self.dense(first_token_tensor)
        return (hidden_states, pooler_output, all_hidden_states, all_attentions)


class Albert(tf.keras.layers.Layer):
    def __init__(self, vocab_size, emb_hidden_size, max_position_embeddings, type_vocab_size, shared_hiden_size,
                 num_layers, num_heads, head_size, units, activation, hidden_size, dropout_prob,
                 layer_norm_eps, is_output_hidden_states, is_output_attentions, kernel_initializer='glorot_uniform', **kwargs):
        super().__init__(**kwargs)
        self.embedding_layer = TransformerEmbedding(vocab_size, emb_hidden_size, max_position_embeddings,
                                                    type_vocab_size, shared_hiden_size, layer_norm_eps, dropout_prob, kernel_initializer)
        assert emb_hidden_size == hidden_size, "for shared transformer layers emb_hidden_size(%d) == hidden_size(%s)" % (emb_hidden_size, hidden_size)
        self.transformer_layer = Transformer(num_heads, head_size, units, activation, hidden_size, dropout_prob, layer_norm_eps, kernel_initializer)
        self.is_output_attentions = is_output_attentions
        self.is_output_hidden_states = is_output_hidden_states
        self.num_layers = num_layers
        self.dense = tf.keras.layers.Dense(
            hidden_size,
            kernel_initializer=kernel_initializer,
            activation="tanh",
            name="dense",
        )

    def call(self, inputs, attention_mask=None, training=False):
        """
        if isinstance(inputs, (tuple, list)):
            input_ids = inputs[0]
            position_ids = inputs[1]
            token_type_ids = inputs[2] if len(inputs) > 2 else None
        else:
            raise NotImplementedError
        """
        hidden_states = self.embedding_layer(inputs, training=training)
        all_hidden_states = () if self.is_output_hidden_states else None
        all_attentions = () if self.is_output_attentions else None

        for _ in range(self.num_layers):
            if self.is_output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = self.transformer_layer(hidden_states, attention_mask, self.is_output_attentions, training=training)
            hidden_states = layer_outputs[0]

            if self.is_output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        # Add last layer
        if self.is_output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        first_token_tensor = hidden_states[:, 0]
        pooler_output = self.dense(first_token_tensor)
        return (hidden_states, pooler_output, all_hidden_states, all_attentions)
