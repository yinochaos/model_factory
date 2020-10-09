#!/usr/bin/env python

"""Tests for `datasets` package.
    ref : https://docs.python.org/zh-cn/3/library/unittest.html
"""

import unittest
import tensorflow as tf
from tensorflow.keras.utils import plot_model
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
        self.embedding_vocab_size = embedding_vocab_size

    def call(self, inputs):
        dec_input, dec_hidden = inputs
        # x 在通过嵌入层后的形状 == （批大小，1，嵌入维度）
        decoder_embedding = self.embedding(dec_input)

        s = self.gru(decoder_embedding, initial_state=dec_hidden)
        decoder_outputs, decoder_state = s

        # 输出的形状 == （批大小 * 1，隐藏层大小）
        #output = tf.reshape(decoder_outputs, (-1, decoder_outputs.shape[2]))

        # 输出的形状 == （批大小，vocab）
        x = self.fc(decoder_outputs)
        return x, decoder_state

    def vocab_size(self):
        return self.embedding_vocab_size


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

    def call(self, x: np.ndarray) -> np.ndarray:
        x = self.embedding(x)
        # print('x',x)
        output, state = self.gru(x)
        return output, state

    def get_inputs(self):
        return tf.keras.layers.Input(shape=(None,), dtype=tf.int32)


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
            name='label', processor='to_tokenid', dtype='int32', shape=(None,), is_with_len=True, token_dict_name='query')
        parser = TextlineParser(token_dicts, data_field_list, label_field)
        generator = TFDataset(parser=parser, file_path='tests/data', file_suffix='text_seq2seq.input')
        dataset = generator.generate_dataset(batch_size=64, num_epochs=900, is_shuffle=False)
        # for (batchs, (inputs, targets)) in enumerate(dataset):
        #    #print('bacths', batchs, 'inputs', inputs, 'targets', targets)
        #    if batchs > 3:
        #        break
        query_vocab_size = token_dicts.dict_size_by_name('query')
        print('query_size', query_vocab_size)
        print('<s>', token_dicts.to_id('query', '<s>'))
        print(r'<\s>', token_dicts.to_id('query', r'<\s>'))
        encoder = GRUEncoder(64, query_vocab_size)
        decoder = GRUDecoder(64, query_vocab_size, query_vocab_size)

        optimizer = tf.keras.optimizers.Adam()
        model = Seq2seqModel(optimizer, seq2seq_cross_entropy_loss, encoder, decoder, feature_fields=data_field_list, label_fields=[label_field])
        # model.model.summary()
        #plot_model(model.model, 'model.png', show_shapes=True)
        model.model.load_weights('model.h5')
        #model.fit(dataset, 64, epochs=20, bar_step=20)
        #model.model.save_weights('model.h5')
        for (batchs, (inputs, targets)) in enumerate(dataset):
            #result = model.predict(inputs)
            result = model.predict_beam(inputs)
            print('target', targets, 'predicts', result)
            if batchs > 1:
                break


if __name__ == '__main__':
    unittest.main()
