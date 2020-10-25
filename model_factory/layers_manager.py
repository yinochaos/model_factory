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

"""Main module."""

from typing import Any, Tuple, List, Dict
import tensorflow as tf
from model_factory.model_factory import GraphNode

# -*- coding: utf-8 -*-

from collections import namedtuple
from typing import List
from collections import defaultdict
import copy
import yaml

#param = ["name", "inputs", "outputs", "node_fn"]
"""
node_fn types:
- softmax :
- embeding :
- reduce_dimT :
- reduce_dimC :
- reduce_dim :
- trans_dimT:
- trans_dimC:
- trans_dim:
"""


class LayersManager(object):
    def __init__(self, conf_file='conf/layers.yaml'):
        self.layer_dicts = {
            'softmax': [],
            'embedding': [],
            'reduce_dimT': [],
            'reduce_dimC': [],
            'reduce_dim': [],
            'trans_dimT': [],
            'trans_dimC': [],
            'trans_dim': []}
        yaml_data = yaml.load(open(conf_file), Loader=yaml.FullLoader)
        for item in yaml_data:
            pass

    def make_node_layer_define(self, node):
        """
        负责根据node的信息，生成layer的定义和名称；这2个分开表示是为了方便做layer参数共享
        @param node :
        @return:
            layer_name :
            layer_define :
        """
        layer_define = ''
        layer_name = ''
        return layer_name, layer_define
