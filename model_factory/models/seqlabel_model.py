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
from model_factory.models.model import Model


class SeqlabelModel(Model):
    def __init__(self, optimier, loss):
        self.optimier = optimier
        self.loss = loss

    def fit(self, train_dataset, batch_size, epochs=1, valid_dataset=None, step_per_epoch=None, callbacks: List[tf.keras.callbacks.Callback] = None):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def train_step(self):
        raise NotImplementedError
