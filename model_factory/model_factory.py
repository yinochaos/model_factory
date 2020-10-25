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

# -*- coding: utf-8 -*-

from collections import namedtuple
from typing import List
from collections import defaultdict
import copy
from graphviz import Digraph

__all__ = ['DataSchema']
#
#shape : [12,10,12]
# type : 数据类型，e.g. text, image, audio, num
# dtype : tf.int32, tf.float32
# shape : 数据shape e.g. [a,b,C,D] a,b var_len C,D const_len
# token_dict_name :
param = ["name", "type", "dtype", "shape"]
DataSchema = namedtuple('DataSchema', field_names=param)
DataSchema.__new__.__defaults__ = tuple([None] * len(param))

param = ["name", "inputs", "outputs", "node_fn"]
GraphNode = namedtuple('GraphNode', field_names=param)
GraphNode.__new__.__defaults__ = tuple([None] * len(param))


class GraphModel(object):
    def __init__(self, inputs: List[DataSchema], outputs: List[DataSchema], nodes: List[GraphNode]):
        self.inputs = inputs
        self.outputs = outputs
        if nodes is None:
            nodes = []
        self.nodes = nodes  # nodes keep grow order
        self.score = None
        self.layer_names = {}
        for node in self.nodes:
            assert node.name not in self.layer_names
            self.layer_names[node.name] = 1
        self.node_names = {}
        for node in self.inputs:
            assert node.name not in self.node_names
            self.layer_names[node.name] = 1
        for node in self.outputs:
            assert node.name not in self.node_names
            self.layer_names[node.name] = 1

    def compute_score(self):
        if self.score is None:
            score = 1.0
            self.score = score
        return self.score

    def _node_to_label(self, node):
        shape = ','.join(node.shape)
        label = '%s|(%s)' % (node.name, shape)
        return label

    def show(self, outfile):
        def add_dot_node(dot, data, node_sets):
            if isinstance(data, DataSchema):
                shape = 'box'
                shape = 'record'
                label = self._node_to_label(data)
            elif isinstance(data, GraphNode):
                shape = 'ellipse'
                shape = 'octagon'
                label = None
            else:
                raise NotImplementedError
            if data.name not in node_sets:
                node_sets.add(data.name)
                dot.attr('node', shape=shape)
                if label is None:
                    dot.node(name=data.name)
                else:
                    dot.node(name=data.name, label=label)

        dot = Digraph(name="model_struct", comment="comment", format="png")
        node_sets = set()
        # for node in self.inputs:
        #    node_sets.add(node.name)
        for node in self.nodes:
            add_dot_node(dot, node, node_sets)
            for data in node.inputs:
                add_dot_node(dot, data, node_sets)
                dot.edge(data.name, node.name)
            for data in node.outputs:
                add_dot_node(dot, data, node_sets)
                dot.edge(node.name, data.name)
        dot.render(outfile)

    def add_node(self, node: GraphNode):
        outputs = []
        for output in self.outputs:
            in_flag = False
            for input in node.inputs:
                if input.name == output.name:
                    in_flag = True
            if not in_flag:
                outputs.append(output)
        self.outputs = outputs
        if node.name in self.layer_names:
            num = self.layer_names[node.name]
            self.layer_names[node.name] += 1
            node = node._replace(name=node.name + str(num))
        else:
            self.layer_names[node.name] = 1
        outputs = []
        for output in node.outputs:
            if output.name in self.node_names:
                num = self.node_names[output.name]
                self.node_names[output.name] += 1
                output = output._replace(name=output.name + str(num))
            else:
                self.node_names[output.name] = 1
            self.outputs.append(output)
            outputs.append(output)
        node = node._replace(outputs=outputs)
        self.nodes.append(node)

    # code_type: tf, tf.keras, pytorch 等等
    # return model_code
    def generator_code(self, code_type='tf.keras'):
        assert code_type == 'tf.keras', "only support tf.keras code generate"
        model_code = ''
        # TODO
        return model_code


class ModelBuilder(object):
    def __init__(self, max_grow_models=5, max_models=20):
        self.max_grow_models = max_grow_models
        self.max_models = max_models

    def build_model(self, inputs: List[DataSchema], outputs: List[DataSchema]) -> List[GraphModel]:
        print('inputs', inputs)
        graph_model = self.make_init_graphmodel(inputs)
        model_list = [graph_model]
        finish_model_list = []
        while True:
            grow_model_list = []
            for graph_model in model_list:
                if self.is_need_grow(graph_model, outputs):
                    grow_model_list.append(graph_model)
                else:
                    finish_model_list.append(graph_model)
            if len(finish_model_list) >= self.max_models:
                return sorted(finish_model_list, key=lambda x: x.compute_score(), reverse=True)[0:self.max_models]
            if len(grow_model_list) == 0:
                break
            if len(grow_model_list) > self.max_grow_models:
                grow_model_list = sorted(grow_model_list, key=lambda x: x.compute_score(), reverse=True)[0:self.max_grow_models]
            model_list = []
            if len(grow_model_list) == 0:
                break
            for graph_model in grow_model_list:
                models = self.grow(graph_model, outputs)
                model_list.extend(models)

        return finish_model_list

    def is_need_grow(self, graph_model: GraphModel, outputs: List[DataSchema]):
        """"""
        assert len(outputs) == 1, 'only support one outputs'
        assert len(outputs[0].shape) <= 1, 'only support one dim vector or scala'
        if len(graph_model.outputs) == 1 and len(graph_model.outputs[0].shape) == 1:
            graph_model.add_node(GraphNode(name='L-last', inputs=graph_model.outputs, outputs=outputs, node_fn='softmax'))
            return False
        else:
            return True

    # 初始化模型,对于基础的数据，进行简单处理，e.g. text需要进行embedding，etc
    def make_init_graphmodel(self, inputs: List[DataSchema]) -> GraphModel:
        graph_model = GraphModel(inputs=inputs, outputs=inputs, nodes=None)
        for data in inputs:
            if data.type == 'text' and data.dtype == 'int':
                output = DataSchema(name=data.name + '_emb', type=data.type, dtype='float32', shape=data.shape + ['E_' + data.name])
                node = GraphNode(name='L-emb' + data.name, inputs=[data], outputs=[output], node_fn='embeding')
                graph_model.add_node(node)
        graph_model.show('init')
        return graph_model

    def _find_same_dims_from_inputs(self, inputs: List[DataSchema]):
        dim_count = defaultdict(int)
        result = {}
        for input in inputs:
            for dim in input.shape:
                dim_count[dim] += 1
        for dim in dim_count:
            if dim_count[dim] > 1:
                result[dim] = dim_count[dim]
        # find common dim in inputs
        return result

    def _reduce_dim(self, data, exclude_dims={}):
        # @TODO add exclude_dims
        keep_dim = data.shape[:-2]
        shape_len = len(data.shape) - 1
        nodes = []
        for i in range(shape_len):
            dim = data.shape[shape_len - i - 1]
            if dim in exclude_dims:
                break
            if dim[0] == 'T':
                name = data.name.split('-')[0] + '-R' + dim
                node = GraphNode(name='L-RNN', inputs=[data],
                                 outputs=[DataSchema(name=name, type=data.type, dtype='float32', shape=data.shape[:shape_len - i - 1] + ['E_' + name])], node_fn='reduce_dimT' + str(i + 1))
                nodes.append(node)
                return nodes
            elif dim[0] == 'C':
                name = data.name.split('-')[0] + '-C' + dim
                node = GraphNode(name='L-CNN', inputs=[data],
                                 outputs=[DataSchema(name=name, type=data.type, dtype='float32', shape=data.shape[:shape_len - i - 1] + ['E_' + name])], node_fn='reduce_dimC' + str(i + 1))
                nodes.append(node)
                return nodes
            else:
                name = data.name.split('-')[0] + '-F' + dim
                node = GraphNode(name='L-FNN', inputs=[data],
                                 outputs=[DataSchema(name=name, type=data.type, dtype='float32', shape=data.shape[:shape_len - i])], node_fn='reduce_dim' + str(i + 1))
                nodes.append(node)
        return nodes

    def _concat(self, datas, name=None, type=None):
        assert len(datas) > 1
        dim = len(datas[0].shape)
        common_dim = datas[0].shape[:-1]
        for data in datas:
            assert len(data.shape) == dim
            assert data.shape[:-1] == common_dim
        if name is None:
            name = "concat_" + ''.join([x.name[0] for x in datas])
        if type is None:
            type = 'num'
        node = GraphNode(name="L-concat", inputs=datas, outputs=[DataSchema(name=name, type=type,
                                                                            dtype='float32', shape=common_dim + ['E_' + name])], node_fn='concat')
        return node

    def _get_concat_list(self, datas):
        for datai in datas:
            concat_list = []
            for dataj in datas:
                if datai.shape[:-1] == dataj.shape[:-1]:
                    concat_list.append(dataj)
            if len(concat_list) > 1:
                return concat_list
        return None

    # 每次生成一层，慢慢把模型生长起来
    def grow(self, cur_model: GraphModel, outputs: List[DataSchema]) -> List[GraphModel]:
        assert len(outputs) == 1, 'only support one outputs'
        assert len(outputs[0].shape) <= 1, 'only support one dim vector or scala'
        concat_list = self._get_concat_list(cur_model.outputs)
        if concat_list is not None:
            node = self._concat(concat_list)
            cur_model.add_node(node)
            return [cur_model]
        model_list = []
        exclude_dims = self._find_same_dims_from_inputs(cur_model.outputs)
        print('exclude_dims', exclude_dims)
        for output in cur_model.outputs:
            if len(output.shape) == 1:
                continue
            nodes = self._reduce_dim(output, exclude_dims)
            for node in nodes:
                model = copy.deepcopy(cur_model)
                model.add_node(node)
                model_list.append(model)
            if len(model_list) > 0:
                break
        return model_list
