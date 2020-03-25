# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=import-self, too-many-lines, len-as-condition, no-else-return, unused-variable, too-many-nested-blocks
# pylint: disable=consider-iterating-dictionary, invalid-name, unused-argument, unused-variable, broad-except
# pylint: disable=import-outside-toplevel, simplifiable-if-expression, unnecessary-comprehension
"""Chainer frontend."""
import collections
import heapq
import os

import numpy
import six

import chainer
from chainer import function
from chainer import function_node
from chainer import variable


import itertools
import logging
import sys

import numpy as np

import tvm
from tvm.ir import module as _module

from .. import analysis as _analysis
from .. import expr as _expr
from .. import op as _op
from .. import function as _function
from ..loops import while_loop
from .common import get_relay_op
from .common import infer_shape as _infer_shape
from .common import infer_value as _infer_value

__all__ = ["from_chainer"]

_function_types = (function.Function, function_node.FunctionNode)

# Chainer Op --> TVM Op Map
CHAINER_OP_TVM_OP_MAP = {
    "LinearFunction"                       : _none(),
    "Convolution2DFunction"                : _none(),
    "Deconvolution2DFunction"              : _none(),
    "AveragePooling2D"                     : _none(),
    "MaxPoolingND"                         : _none(),
    "LocalResponseNormalization"           : _none(),
    "ReLU"                                 : _none(),
    "LeakyReLU"                            : _none(),
    "Concat"                               : _none(),
    "Softmax"                              : _none(),
    "Sigmoid"                              : _none(),
    "Reshape"                              : _none(),
}

def trace_graph_funcs(outputs):
    fan_out = collections.defaultdict(int)
    cand_funcs = []

    def add_cand_to_check(cands):
        for cand in cands:
            x = cand.creator
            if x is None:
                continue
            if x not in fan_out:
                heapq.heappush(cand_funcs, (-x.rank, len(fan_out), x))
            fan_out[x] += 1

    add_cand_to_check(outputs)
    while cand_funcs:
        _, _, func = heapq.heappop(cand_funcs)
        assert isinstance(func, _function_types)
        add_cand_to_check(func.inputs)

    ret = []
    cand_funcs = []
    seen_set = set()

    def add_cand(cands):
        cands = [cand.creator for cand in cands if cand.creator is not None]
        for x in cands:
            if x in seen_set:
                continue
            order = 1
            if fan_out[x] == 1 and len(cands) == 1:
                order = -len(seen_set)
            heapq.heappush(cand_funcs, (order, -x.rank, -len(seen_set), x))
            seen_set.add(x)

    add_cand(outputs)
    while cand_funcs:
        _, _, _, func = heapq.heappop(cand_funcs)
        ret.append(func)
        add_cand(func.inputs)

    return ret[::-1]


def get_relay_input_vars(input_shapes):
    """ Return Relay vars from input shapes """
    return {iname: _expr.var(iname, shape=ishape)
            for iname, ishape in input_shapes.items()}

def convert_operators(func_list):
    """ Convert each Torch IR operators to Relay equivalent """
    for func in func_list:
        assert isinstance(func, _function_types)

        relay_op = CHAINER_OP_TVM_OP_MAP[func.label]
        relay_out = relay_op(func)

    return [relay_out]

# Chainer-TVM Bridge
class ChainerTVMBridge(object):
    """A helper class for handling relay functions from chainer model.
    """

    def __init__(self, model, shape, dtype='float32'):
        self._module = model
        self._shape = shape
        self._dtype = dtype
        self._sym_array = {}
        self._tvmparams = {}
        self._outs = []

    def from_chainer(self):
        """To convert the chainer symbol to relay functions."""

        # Form dummy input based on input shape and datatype provided
        #TODO: Make it multi-input later
        x = chainer.Variable(np.zeros(self._shape, dtype=np.float32))

        # Creates a context of Chainer with Computation Graph enabled
        with function.force_backprop_mode(), chainer.using_config('train', False):
            output = self._module(x)

        # Instance validation of output
        if isinstance(output, variable.Variable):
            output = [output]
        assert isinstance(output, (tuple, list))
        for i in output:
            assert isinstance(i, variable.Variable)

        # Dump the Chainer Graph
        func_list = trace_graph_funcs(outputs)
        
        # Prepare relay graphs
        input_vars = get_relay_input_vars(self._shape)
        tensors = {}
        self._tvmparams = {k: tvm.nd.array(v) for k, v in tensors.items()}

        outputs = convert_operators(func_list)
        outputs = outputs[0] if len(outputs) == 1 else _expr.Tuple(outputs)
        sym = _function.Function(analysis.free_vars(outputs), outputs)
        return IRModule.from_expr(sym), self._tvmparams


def from_chainer(model, input_shapes, dtype="float32"):
    """ Load Chainer model in the form of a chainer.Chain model and convert into relay.
    The corresponding parameters will be mapped automatically.

    Parameters
    ----------
    model : chainer.Chain object
        Chainer graph

    input_shapes : Dictionary of input dimensions
        Graph level input shape dictionary

    dtype : str or dict of str to str
        The input types to the graph

    Returns
    -------
    mod : tvm.relay.Module
        The module that optimizations will be performed on.

    params : dict of str to tvm.runtime.NDArray
        Dict of converted parameters stored in tvm.runtime.ndarray format
    """
    return ChainerTVMBridge(model, input_shapes, dtype).from_chainer()
