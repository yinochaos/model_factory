#!/usr/bin/env python

"""Tests for `datasets` package.
    ref : https://docs.python.org/zh-cn/3/library/unittest.html
"""

import unittest
import tensorflow as tf
from datasets import TextlineParser
from datasets import TFDataset
from datasets.utils import TokenDicts, DataSchema
from model_factory.models import ClassiferModel

import tensorflow as tf
import numpy as np


class TestClassifierModel (unittest.TestCase):
    """Tests for `datasets` package."""

    def setUp(self):
        if tf.__version__.startswith('1.'):
            tf.compat.v1.enable_eager_execution()
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""
        pass

    def create_model(self, query_vocab_size):
        query_input = tf.keras.Input(shape=(None,), name='query_input')
        query_emb = tf.keras.layers.Embedding(query_vocab_size, 64)(query_input)
        #query_state = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=100, return_sequences=False))(query_emb)
        #query_state = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=100, return_sequences=False))(query_emb)
        query_state = tf.keras.layers.Bidirectional(tf.compat.v1.keras.layers.CuDNNGRU(units=100, return_sequences=False))(query_emb)
        middle = tf.keras.layers.Dense(64, activation='relu', name='mid_out')(query_state)
        prob = tf.keras.layers.Dense(1, activation='sigmoid', name='prob')(middle)
        # return tf.keras.Model(inputs=query_input, ouputs=prob)
        return tf.keras.Model(query_input, prob)

    def test_text_seq2seq_model(self):
        """Test something."""
        # init token_dicts
        gpus = tf.config.experimental.list_physical_devices('GPU')
        print(gpus)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        token_dicts = TokenDicts('tests/data/dicts', {'query': 0})
        data_field_list = []
        #param = ["name", "processor", "type", "dtype", "shape", "max_len", "token_dict_name"]
        data_field_list.append(DataSchema(name='query', processor='to_tokenid',
                                          dtype='int32', shape=(None,), is_with_len=False, token_dict_name='query'))
        label_field = DataSchema(name='label', processor='to_np', dtype='float32', shape=(1,), is_with_len=False)
        parser = TextlineParser(token_dicts, data_field_list, label_field)
        generator = TFDataset(parser=parser, file_path='tests/data', file_suffix='text_classifier.input')
        dataset = generator.generate_dataset(batch_size=16, num_epochs=20, is_shuffle=False)
        for (batchs, (inputs, targets)) in enumerate(dataset):
            print('bacths', batchs, 'inputs', inputs, 'targets', targets)
            if batchs > 3:
                break
        query_vocab_size = token_dicts.dict_size_by_name('query')
        print('query_size', query_vocab_size)

        optimizer = tf.keras.optimizers.Adam()
        model = ClassiferModel(optimizer=optimizer, loss='binary_crossentropy', model=self.create_model(query_vocab_size))
        # model.summary()
        model.fit(dataset, 12, epochs=8, bar_step=10)
        model.model.save_weights('model.h5')


if __name__ == '__main__':
    unittest.main()
