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


class pBLSTM(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(pBLSTM, self).__init__()
        self.units = units
        name = kwargs.pop("name", None)
        lstm = tf.keras.layers.LSTM(self.units, return_sequences=True, **kwargs)
        if name is None:
            self.bilstm = tf.keras.layers.Bidirectional(lstm)
        else:
            self.bilstm = tf.keras.layers.Bidirectional(lstm, name=name)

    @tf.function
    def call(self, inputs):
        inputs = self.bilstm(inputs)
        batch_size, time_step, dimension = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2]
        inputs = tf.concat([inputs, tf.zeros([batch_size, time_step % 2, dimension], dtype=tf.float32)], axis=1)
        time_step += time_step % 2
        return tf.reshape(inputs, [batch_size, time_step // 2, dimension * 2])
