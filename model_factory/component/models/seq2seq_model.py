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
import tqdm
from typing import Any, Tuple, List, Dict
import tensorflow as tf
import numpy as np
from model_factory.component.models.model import Model

""" model interface
"""


class Seq2seqModel(Model):
    def __init__(self, optimizer, loss, encoder, decoder, max_decoder_len=32):
        self.optimizer = optimizer
        self.loss = loss
        self.encoder = encoder
        self.decoder = decoder
        self.hidden_size = encoder.hidden_size
        self.max_decoder_len = max_decoder_len

    # def summary(self):

    def predict(self, inputs, bos_token_id=None, eos_token_id=None, token_dict=None):
        def id_to_tokens(ids, token_dict):
            return [token_dict[id] if id in token_dict else '<UNKOWN>' for id in ids]
        results = []
        attentions = []

        bos_token_id = 1 if bos_token_id is None else bos_token_id
        eos_token_id = 2 if eos_token_id is None else eos_token_id

        for input_seq in inputs:
            enc_hidden = tf.zeros((1, self.hidden_size))
            enc_output, enc_hidden = self.encoder(input_seq, enc_hidden)
            dec_hidden = enc_hidden

            attention_plot = []
            token_out = []

            dec_input = tf.expand_dims([bos_token_id], 0)

            for t in range(self.max_decoder_len):
                predictions, dec_hidden, att_weights = self.decoder(dec_input, dec_hidden, enc_output)
                # storing the attention weights to plot later on
                attention_weights = tf.reshape(att_weights, (-1,))
                attention_plot.append(attention_weights.numpy())

                next_tokens = tf.argmax(predictions[0]).numpy()
                token_out.append(next_tokens)
                if next_tokens == eos_token_id:
                    break
                dec_input = tf.expand_dims([next_tokens], 0)
            r = id_to_tokens(token_out, token_dict)
            results.append(r)
            attentions.append(attention_plot)
        return results, attentions

    # @tf.function
    def train_step(self, inputs, targets):
        loss = 0
        enc_hidden = tf.zeros((inputs.shape[0], self.hidden_size))

        with tf.GradientTape() as tape:
            enc_output, enc_hidden = self.encoder(inputs, enc_hidden)
            dec_hidden = enc_hidden
            bos_token_id = 1
            dec_input = tf.expand_dims([bos_token_id] * targets.shape[0], 1)

            # Teacher forcing - feeding the target as the next input
            for t in range(1, targets.shape[1]):
                # pass enc_output to the decoder
                predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
                loss += self.loss(targets[:, t], predictions)
                # using teacher forcing
                dec_input = tf.expand_dims(targets[:, t], 1)

        batch_loss = (loss / int(targets.shape[1]))
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss
