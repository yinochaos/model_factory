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
import numpy as np
from model_factory.models.model import Model


class GANModel(Model):
    def __init__(self, optimizer, loss, generator, discriminator, discriminator_optimier=None, discriminator_loss=None, latent_dim=128):
        self.optimizer = optimizer
        self.loss = loss
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.discriminator = discriminator
        discriminator_optimier = optimizer if discriminator_optimier is None else discriminator_optimier
        discriminator_loss = optimizer if discriminator_loss is None else discriminator_loss
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # The generator takes noise as input and generates imgs
        z = tf.keras.layers.Input(shape=(self.latent_dim,))
        img = self.generator(z)
        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        validity = self.discriminator(img)
        self.combined = tf.keras.models.Model(z, validity)
        self.combined.compile(loss=loss, optimizer=optimizer)

    def predict(self, inputs):
        return self.combined.predict(inputs)

    def train_step(self, inputs, targets):
        # Select a random batch of images
        valid = np.ones((inputs.shape[0], 1))
        fake = np.zeros((inputs.shape[0], 1))

        imgs = inputs

        noise = np.random.normal(0, 1, (inputs.shape[0], self.latent_dim))

        # Generate a batch of new images
        gen_imgs = self.generator.predict(noise)
        # Train the discriminator
        d_loss_real = self.discriminator.train_on_batch(imgs, valid)
        d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        _ = d_loss

        # ---------------------
        #  Train Generator
        # ---------------------
        noise = np.random.normal(0, 1, (inputs.shape[0], self.latent_dim))

        # Train the generator (to have the discriminator label samples as valid)
        g_loss = self.combined.train_on_batch(noise, valid)
        # TODO add two loss
        return g_loss
