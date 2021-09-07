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


class TransformerEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, hidden_size, max_position_embeddings, type_vocab_size=None, shared_hiden_size=0,
                 layer_norm_eps=1e-6, dropout_prob=0.5, kernel_initializer='glorot_uniform', **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        if shared_hiden_size > 0:
            hidden_size = shared_hiden_size
        self.word_embeddings = tf.keras.layers.Embedding(
            vocab_size,
            hidden_size,
            embeddings_initializer=kernel_initializer,
            name="token_type_embeddings",
        )
        if shared_hiden_size > 0:
            self.shared_word_embeddings = tf.keras.layers.Dense(self.hidden_size, use_bias=False, kernel_initializer=kernel_initializer)
        else:
            self.shared_word_embeddings = None
        self.position_embeddings = tf.keras.layers.Embedding(
            max_position_embeddings,
            self.hidden_size,
            embeddings_initializer=kernel_initializer,
            name="position_embeddings",
        )
        if type_vocab_size is None:
            self.token_type_embeddings = None
        else:
            self.token_type_embeddings = tf.keras.layers.Embedding(
                type_vocab_size,
                self.hidden_size,
                embeddings_initializer=kernel_initializer,
                name="token_type_embeddings",
            )
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(dropout_prob)

    # @tf.function
    def call(self, inputs, training=False):
        if self.token_type_embeddings is not None:
            word_ids, pos_ids, type_ids = inputs
        else:
            word_ids, pos_ids = inputs
        if self.shared_word_embeddings is None:
            word_embeddings = self.word_embeddings(word_ids)
        else:
            word_embeddings = self.shared_word_embeddings(self.word_embeddings(word_ids))
        print('type_ids AAA ', type_ids)
        if self.token_type_embeddings is None:
            embeddings = word_embeddings + self.position_embeddings(pos_ids)
        else:
            embeddings = word_embeddings + self.position_embeddings(pos_ids)
            embeddings = embeddings + self.token_type_embeddings(type_ids)
        print('emb  ', embeddings)
        embeddings = self.LayerNorm(embeddings)
        print('emb1 ', embeddings)
        embeddings = self.dropout(embeddings, training=training)
        print('emb2  ', embeddings)
        return embeddings


class RelativePositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, demb, **kwargs):
        super().__init__(**kwargs)

        self.inv_freq = 1 / (10000 ** (tf.range(0, demb, 2.0) / demb))

    def call(self, pos_seq, batch_size=None):
        sinusoid_inp = tf.einsum("i,j->ij", pos_seq, self.inv_freq)
        pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], -1)

        if batch_size is not None:
            return tf.tile(pos_emb[:, None, :], [1, batch_size, 1])
        else:
            return pos_emb[:, None, :]
