#!/usr/bin/env python

"""Tests for `datasets` package.
    ref : https://docs.python.org/zh-cn/3/library/unittest.html
"""

import unittest
import tensorflow as tf
from datasets import TextlineParser
from datasets import TFDataset
from datasets.utils import TokenDicts, DataSchema
from model_factory.models import Seq2seqModel
from model_factory.losses import seq2seq_cross_entropy_loss

import tensorflow as tf
import numpy as np


class GRUDecoder(tf.keras.Model):
    def __init__(self,
                 hidden_size: int,
                 embedding_vocab_size: int,
                 vocab_size: int):
        super(GRUDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(embedding_vocab_size,
                                                   hidden_size,
                                                   trainable=True,
                                                   mask_zero=True)

        self.gru = tf.keras.layers.GRU(hidden_size,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, dec_input, dec_hidden, enc_output):
        # x 在通过嵌入层后的形状 == （批大小，1，嵌入维度）
        decoder_embedding = self.embedding(dec_input)

        s = self.gru(decoder_embedding, initial_state=dec_hidden)
        decoder_outputs, decoder_state = s

        # 输出的形状 == （批大小 * 1，隐藏层大小）
        output = tf.reshape(decoder_outputs, (-1, decoder_outputs.shape[2]))

        # 输出的形状 == （批大小，vocab）
        x = self.fc(output)
        return x, decoder_state, None


class GRUEncoder(tf.keras.Model):
    def __init__(self, hidden_size, embedding_vocab_size):
        super(GRUEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = tf.keras.layers.Embedding(embedding_vocab_size,
                                                   hidden_size,
                                                   trainable=True,
                                                   mask_zero=True)
        self.gru = tf.keras.layers.GRU(hidden_size,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x: np.ndarray, hidden: np.ndarray) -> np.ndarray:
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state


class TestDatasets(unittest.TestCase):
    """Tests for `datasets` package."""

    def setUp(self):
        if tf.__version__.startswith('1.'):
            tf.compat.v1.enable_eager_execution()
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""
        pass

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
        label_field = DataSchema(
            name='label', processor='to_tokenid', dtype='int32', shape=(None,), is_with_len=False, token_dict_name='query')
        parser = TextlineParser(token_dicts, data_field_list, label_field)
        generator = TFDataset(parser=parser, file_path='tests/data', file_suffix='text_seq2seq.input')
        dataset = generator.generate_dataset(batch_size=12, num_epochs=60, is_shuffle=False)
        for (batchs, (inputs, targets)) in enumerate(dataset):
            print('bacths', batchs, 'inputs', inputs, 'targets', targets)
            if batchs > 3:
                break
        query_vocab_size = token_dicts.dict_size_by_name('query')
        print('query_size', query_vocab_size)
        print('<s>', token_dicts.to_id('query', '<s>'))
        print(r'<\s>', token_dicts.to_id('query', r'<\s>'))
        encoder = GRUEncoder(64, query_vocab_size)
        decoder = GRUDecoder(64, query_vocab_size, query_vocab_size)

        optimizer = tf.keras.optimizers.Adam()
        model = Seq2seqModel(optimizer, seq2seq_cross_entropy_loss, encoder, decoder)
        # model.summary()
        model.fit(dataset, 12, epochs=8, bar_step=10)


if __name__ == '__main__':
    unittest.main()
