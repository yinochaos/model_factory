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
import math
import sys

""" model interface
"""


class Model(object):
    def __init__(self, optimizer, loss, model):
        self.optimizer = optimizer
        self.loss = loss
        self.model = model
        self.model.compile(optimizer=self.optimizer, loss=self.loss)
        self.num_loss = 1

    def fit(self, train_dataset, batch_size, epochs=1, valid_dataset=None, step_per_epoch=None,
            callbacks: List[tf.keras.callbacks.Callback] = None, bar_step=1, train_dataset_len=None):
        if callbacks is None:
            callbacks = []
        history_callback = tf.keras.callbacks.History()
        callbacks.append(history_callback)

        for c in callbacks:
            c.set_model(self.model)
            c.on_train_begin()

        epochs_seq = [i for i in range(epochs)]
        if train_dataset_len is None:
            train_dataset_len = 10000
            train_dataset_len = self.run_epoch(train_dataset, 0, epochs, callbacks, train_dataset_len, bar_step, batch_size)
            epochs_seq = epochs_seq[1:]

        for epoch in epochs_seq:
            self.run_epoch(train_dataset, epoch, epochs, callbacks, train_dataset_len, bar_step, batch_size)
        return history_callback

    def run_epoch(self, train_dataset, epoch, epochs, callbacks, train_dataset_len, bar_step, batch_size):
        for c in callbacks:
            c.on_epoch_begin(epoch=epoch)
        total_loss = []
        total_batchs = 0

        with tqdm.tqdm(total=train_dataset_len) as p_bar:
            for (batchs, (inputs, targets)) in enumerate(train_dataset):
                #print('inputs', inputs)
                #print('targets', targets)
                if isinstance(inputs, (tuple, list)):
                    if inputs[0].shape[0] != batch_size:
                        continue
                else:
                    if inputs.shape[0] != batch_size:
                        continue
                # if targets.shape[0] != batch_size:
                #    continue
                batch_loss = self.train_step(inputs, targets)
                if not isinstance(batch_loss, float):
                    batch_loss = batch_loss.numpy()
                if math.isnan(batch_loss):
                    print("batch_loss is nan in batchs(%d)" % (batchs), file=sys.stderr)
                    continue
                # TODO add multi_loss
                total_loss.append(batch_loss)
                if batchs % bar_step == 0:
                    p_bar.update(bar_step)
                    info = f"Epoch {epoch + 1}/{epochs} | Epoch Loss: {np.mean(total_loss):.4f} " \
                        f"Batch Loss: {batch_loss:.4f}"
                    p_bar.set_description_str(info)
                    total_batchs = batchs
        logs = {'loss': np.mean(total_loss)}
        for c in callbacks:
            c.on_epoch_end(epoch=epoch, logs=logs)
        return total_batchs

    def predict(self, inputs):
        return self.model.predict(inputs)

    def train_step(self, inputs, targets):
        return self.model.train_on_batch(inputs, targets)
