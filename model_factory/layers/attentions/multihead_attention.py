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

import math
import tensorflow as tf


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads: int, head_size: int, key_size=None, use_bias=True,
                 dropout: float = 0.0, kernel_initializer='glorot_uniform', **kwargs):
        super().__init__(**kwargs)
        self.num_attention_heads = num_heads
        self.head_size = head_size
        self.key_size = key_size or head_size
        self.all_head_size = self.num_attention_heads * self.head_size
        self.query = tf.keras.layers.Dense(
            self.key_size * self.num_attention_heads, use_bias=use_bias, kernel_initializer=kernel_initializer, name="query"
        )
        self.key = tf.keras.layers.Dense(
            self.key_size * self.num_attention_heads, use_bias=use_bias, kernel_initializer=kernel_initializer, name="key"
        )
        self.value = tf.keras.layers.Dense(
            self.all_head_size, use_bias=use_bias, kernel_initializer=kernel_initializer, name="value"
        )
        self.dropout = tf.keras.layers.Dropout(dropout)
        #self.scale = 1 / tf.sqrt(self.key_size)
        self.scale = math.sqrt(self.key_size)

    # attention_mask : a binary Tensor of shape `[batch_size?, num_heads?, query_elements, key_elements]`
    def call(self, inputs, attention_mask=None, is_output_attentions=False, training=False):

        query = inputs[0]
        key = inputs[1]
        value = inputs[2] if len(inputs) > 2 else key

        # verify shapes
        if key.shape[-2] != value.shape[-2]:
            raise ValueError(
                "the number of elements in 'key' must be equal to the same as the number of elements in 'value'"
            )

        if attention_mask is not None:
            if len(attention_mask.shape) < 2:
                raise ValueError("'mask' must have atleast 2 dimensions")
            if query.shape[-2] != attention_mask.shape[-2]:
                raise ValueError(
                    "mask's second to last dimension must be equal to the number of elements in 'query'"
                )
            if key.shape[-2] != attention_mask.shape[-1]:
                raise ValueError(
                    "mask's last dimension must be equal to the number of elements in 'key'"
                )

        # Linear transformations
        batch_size = tf.shape(query)[0]
        qw = self.query(query)
        kw = self.key(key)
        vw = self.value(value)
        # (B, S, H*D) -> (B, S, H, D)
        qw = tf.reshape(qw, (batch_size, - 1, self.num_attention_heads, self.key_size))
        kw = tf.reshape(kw, (batch_size, -1, self.num_attention_heads, self.key_size))
        vw = tf.reshape(vw, (batch_size, -1, self.num_attention_heads, self.head_size))
        # compute Attention (B, H, Sq, Sk)
        attention_score = tf.einsum('bjhd,bkhd->bhjk', qw, kw)
        #depth = tf.constant(self.key_size, dtype=attention_score.dtype)
        #attention_score /= tf.sqrt(depth)
        attention_score *= self.scale

        # apply mask
        if attention_mask is not None:
            attention_mask = tf.cast(attention_mask, tf.float32)
            # possibly expand on the head dimension so broadcasting works
            if len(attention_mask.shape) != len(attention_score.shape):
                attention_mask = tf.expand_dims(attention_mask, -3)
            attention_score += -10e9 * (1.0 - attention_mask)

        # Normalize the attention scores to probabilities
        attention_prob = tf.nn.softmax(attention_score, axis=-1)
        attention_prob = self.dropout(attention_prob, training=training)
        output = tf.einsum('bhjk,bkhd->bjhd', attention_prob, vw)
        output = tf.reshape(output, [batch_size, -1, self.all_head_size])
        return (output, attention_prob) if is_output_attentions else (output,)


class RelativeMultiHeadAttention(tf.keras.layers.Layer):
    """
    Rel
    h = [m,h]
    q,k,v = h*Wq, h*Wk,e, h*Wv
    Att =qT*k + qT*Wk,r*R + uT*k + vT*Wk,r*R
    let Q = Wk,r*R

    """

    def __init__(self, num_heads: int, head_size: int, key_size=None, use_bias=True,
                 dropout: float = 0.0, kernel_initializer='glorot_uniform', **kwargs):
        super().__init__(**kwargs)
        self.num_attention_heads = num_heads
        self.head_size = head_size
        self.key_size = key_size or head_size
        self.all_head_size = self.num_attention_heads * self.head_size
        self.query = tf.keras.layers.Dense(
            self.key_size * self.num_attention_heads, use_bias=use_bias, kernel_initializer=kernel_initializer, name="query"
        )
        # Wk,e
        self.key = tf.keras.layers.Dense(
            self.key_size * self.num_attention_heads, use_bias=use_bias, kernel_initializer=kernel_initializer, name="key"
        )
        self.value = tf.keras.layers.Dense(
            self.all_head_size, use_bias=use_bias, kernel_initializer=kernel_initializer, name="value"
        )
        # Wk.r
        self.key_pos = tf.keras.layers.Dense(
            self.key_size * self.num_attention_heads, use_bias=use_bias, kernel_initializer=kernel_initializer, name="key"
        )
        self.r_v = self.add_weight(shape=(self.num_attention_heads, self.head_size), initializer="zeros", trainable=True, name="r_r_bias")
        self.r_u = self.add_weight(shape=(self.num_attention_heads, self.head_size), initializer="zeros", trainable=True, name="r_w_bias")
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.scale = 1 / tf.sqrt(self.key_size)

    def _rel_shift(self, x):
        x_size = x.shape

        x = tf.pad(x, [[0, 0], [1, 0], [0, 0], [0, 0]])
        x = tf.reshape(x, [x_size[1] + 1, x_size[0], x_size[2], x_size[3]])
        x = tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
        x = tf.reshape(x, x_size)

        return x

    # attention_mask : a binary Tensor of shape `[batch_size?, num_heads?, query_elements, key_elements]`
    def call(self, inputs, rel_pos_embedding, mems, attention_mask=None, is_output_attentions=False, training=False):
        """
        Sk,Sq for key_len,query_len; H for num_heads; Dh,Dk for head_size, key_size
        r_emb: [Sk, H, D], used for term B
        r_w_bias: [H, D], used for term C
        r_bias: [Sk, D], used for term D
        """
        query = inputs

        if attention_mask is not None:
            if len(attention_mask.shape) < 2:
                raise ValueError("'mask' must have atleast 2 dimensions")
            if query.shape[-2] != attention_mask.shape[-2]:
                raise ValueError(
                    "mask's second to last dimension must be equal to the number of elements in 'query'"
                )

        # Linear transformations
        batch_size = tf.shape(query)[0]
        if mems is not None:
            query = tf.concat([mems, query], 0)
        qw = self.query(query)
        kw = self.key(query)
        vw = self.value(query)
        # (B, S, H*D) -> (B, S, H, D)
        qw = tf.reshape(qw, (batch_size, - 1, self.num_attention_heads, self.key_size))
        kw = tf.reshape(kw, (batch_size, -1, self.num_attention_heads, self.key_size))
        vw = tf.reshape(vw, (batch_size, -1, self.num_attention_heads, self.head_size))
        # compute attention score :AC = (qT+ruT)*k
        w_q_ru = qw + self.r_u  # (B, Sq, H, D)
        AC = tf.einsum("bjhd,bkhd->bhjk", w_q_ru, kw)  # (B, H, Sq, Sk)
        Q = self.key_pos(rel_pos_embedding)  # Q =Wk,r*R
        # compute attention score :BD =  (qT + rvT)*Wk,r*R
        w_q_rv = qw + self.r_v
        BD = tf.einsum("bjhd,khd->jkbh", w_q_rv, Q)  # qlen x klen x bsz x n_head
        BD = self._rel_shift(BD)
        BD = tf.einsum("jkbh->bhjk", BD)
        attention_score = AC + BD
        attention_score *= self.scale

        # apply mask
        if attention_mask is not None:
            attention_mask = tf.cast(attention_mask, tf.float32)
            # possibly expand on the head dimension so broadcasting works
            if len(attention_mask.shape) != len(attention_score.shape):
                attention_mask = tf.expand_dims(attention_mask, -3)
            attention_score += -10e9 * (1.0 - attention_mask)

        # Normalize the attention scores to probabilities
        attention_prob = tf.nn.softmax(attention_score, axis=-1)
        attention_prob = self.dropout(attention_prob, training=training)
        output = tf.einsum('bhjk,bkhd->bjhd', attention_prob, vw)
        output = tf.reshape(output, [batch_size, -1, self.all_head_size])
        return (output, attention_prob) if is_output_attentions else (output,)
