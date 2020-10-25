#!/usr/bin/env python


import unittest
from model_factory.model_factory import DataSchema
from model_factory.model_factory import ModelBuilder


class TestModelFactory (unittest.TestCase):
    """Tests for `model_factory` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""
        pass

    def test_text_seq2seq_model(self):
        """Test something."""
        # init token_dicts
        inputs = [DataSchema('query', 'text', 'int', ['T_q', 'a']),
                  DataSchema('slots', 'text', 'int', ['b']),
                  DataSchema('time', 'num', 'float32', ['T_q', 'c'])]
        outputs = [DataSchema('domain', 'prob', 'int', ['C']),
                   DataSchema('intent', 'prob', 'int32', ['D'])]
        outputs = [DataSchema('domain', 'prob', 'int', ['C'])]
        builder = ModelBuilder(max_grow_models=100, max_models=1000)
        model_list = builder.build_model(inputs, outputs)
        for model, i in zip(model_list, range(len(model_list))):
            print('model', model)
            model.show('outputs/' + str(i))


if __name__ == '__main__':
    unittest.main()
