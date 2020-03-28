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
import numpy as np

import chainer
from chainer import function
from chainer import function_node


from tvm.ir import IRModule

from .. import analysis
from .. import expr as _expr
from .. import function as _function
from .. import op as _op
from ... import nd as _nd
from .common import new_var

__all__ = ["from_chainer"]

_function_types = (function.Function, function_node.FunctionNode)

def _none():
    def _impl(inputs, func):
        return None
    return _impl

def _relu():
    def _impl(inputs, func):
        data = inputs[0]
        return _op.nn.relu(data)
    return _impl

# Chainer Op --> TVM Op Map
CHAINER_OP_TVM_OP_MAP = {
    "LinearFunction"                       : _none(),
    "Convolution2DFunction"                : _none(),
    "Deconvolution2DFunction"              : _none(),
    "AveragePooling2D"                     : _none(),
    "MaxPoolingND"                         : _none(),
    "LocalResponseNormalization"           : _none(),
    "ReLU"                                 : _relu(),
    "LeakyReLU"                            : _none(),
    "Concat"                               : _none(),
    "Softmax"                              : _none(),
    "Sigmoid"                              : _none(),
    "Reshape"                              : _none(),
}

def get_array(parameter):
    """ A helper function to get underlying array.
    """
    if isinstance(parameter, chainer.Parameter):
        array = parameter.array
    elif isinstance(parameter, chainer.Variable):
        array = parameter.array
    elif isinstance(parameter, chainer.get_array_types()):
        array = parameter
    else:
        raise ValueError(
            'The type of parameter is unknown. It should be either Parameter '
            'or Variable or ndarray, but the type was {}.'.format(
                type(parameter)))
    return array

def canonicalize_param_names(name):
    """ A helper function to canonicalize param names.
    """
    return 'param' + name.replace('/', '_')

class VariableStore(object):
    """A helper class for handling Params and variables of Chainer Model.

        Parameters
    ----------
    model : chainer.Chain object
        The chainer graph
    """
    def __init__(self, model):
        self.name_list = dict()
        self.tvmparams = {}
        self.params = {}
        self.nodes = {}
        for name, param in model.namedparams():
            processed_name = canonicalize_param_names(name)
            self.set_name(param, processed_name)
            self.params[processed_name] = _nd.array(get_array(param))
            self.nodes[processed_name] = new_var(processed_name,
                                                 shape=param.shape, dtype=str(param.dtype))

    def get_name(self, var):
        """Get name."""
        str_id = id(var)
        if str_id in self.name_list:
            return self.name_list[str_id][0]
        else:
            new_name = 'var{}'.format(len(self.name_list))
            self.set_name(var, new_name)
            return new_name

    def set_name(self, var, name, pinned=False):
        """Set name."""
        str_id = id(var)
        assert str_id not in self.name_list or not self.name_list[str_id][1]
        self.name_list[str_id] = (name, pinned)
        if isinstance(var, (chainer.Variable, chainer.Parameter)):
            array_id = id(var.array)
            self.name_list[array_id] = (name, pinned)
            self.nodes[name] = new_var(name, shape=var.shape, dtype=str(var.dtype))

    def update_node(self, name, node):
        """Update node."""
        self.nodes[name] = node

def trace_graph_funcs(outputs):
    """ A helper function to trace all functions -> inputs -> outputs of Chainer Model.
    """
    cands = []
    function_nodes = collections.OrderedDict()
    push_count = [0]

    def add_cand(cand):
        heapq.heappush(cands, (-cand.rank, push_count[0], cand))
        push_count[0] += 1

    for o in outputs:
        if isinstance(o, chainer.Variable):
            o = o.node
        add_cand(o)

    while cands:
        _, _, cand = heapq.heappop(cands)
        if not isinstance(cand, chainer.variable.VariableNode):
            raise NotImplementedError(
                'TVM-Chainer does not support node type {}'.format(
                    type(cand)))
        creator = cand.creator_node
        if creator is None:
            continue
        assert isinstance(creator, chainer.FunctionNode)
        creator_id = id(creator)
        if creator_id in function_nodes:
            continue
        function_nodes[creator_id] = creator

        for input_ in creator.inputs:
            add_cand(input_)

    return reversed(function_nodes.values())

def get_relay_input_vars(input_shapes):
    """ Return Relay vars from input shapes """
    return {iname: _expr.var(iname, shape=ishape)
            for iname, ishape in input_shapes.items()}

def get_converter(op):
    """ Convert each Chainer operators to Relay converter """
    return CHAINER_OP_TVM_OP_MAP[op]

# Chainer-TVM Bridge
class ChainerTVMBridge(object):
    """A helper class for handling relay functions from chainer model.
    """

    def __init__(self, model, shape, dtype='float32'):
        self._model = model
        self._shape = shape
        self._dtype = dtype
        self._datastore = VariableStore(model)
        self._function_nodes = {}

    def convert_chainer_ops(self, func):
        """Convert all chainer ops to corresponding TVM Ops."""
        if isinstance(func, chainer.function.FunctionAdapter):
            func = func.function

        #TODO: Link input given names and shape with nodes
        input_nodes = []
        for input_var in func.inputs:
            # 'input_var' is a VariableNode,
            # so check if it has a Variable/Parameter
            var = input_var.get_variable_or_none()
            if var is None:
                # Use VariableNode as is
                input_name = self._datastore.get_name(input_var)
            else:  # It is a paramdeter inside a Link or network input
                input_name = self._datastore.get_name(var)

            input_nodes.append(self._datastore.nodes[input_name])

        tvm_output = get_converter(func.label)(input_nodes, func)

        # This is to get corresponding VariableNode id from the output
        # Variable of the network
        output_names = []
        for i, output_ref in enumerate(func.outputs):
            if output_ref() is None:
                var = output_ref
            else:
                var = output_ref().get_variable_or_none()
                if var is None:
                    var = output_ref()
            output_names.append(self._datastore.get_name(var))

        if not isinstance(tvm_output, _expr.TupleWrapper):
            self._datastore.nodes[output_names[0]] = tvm_output
        else:
            assert len(tvm_output) == len(func.outputs)
            for k, i in zip(output_names, range(len(tvm_output))):
                self._datastore.nodes[k] = tvm_output[i]

    def from_chainer(self):
        """To convert the Chainer symbol to relay functions."""

        # Form dummy input based on input shape and datatype provided
        #TODO: Make it multi-input later
        x = chainer.Variable(np.zeros(self._shape, dtype=np.float32))

        # Creates a context of Chainer with Computation Graph enabled
        with function.force_backprop_mode(), chainer.using_config('train', False):
            output = self._model(x)

        # Instance validation of output
        if isinstance(output, (list, tuple)):
            flat_outputs = output
        elif isinstance(output, dict):
            flat_outputs = list(output.values())
        elif isinstance(output, chainer.Variable):
            flat_outputs = [output]
        else:
            raise RuntimeError(
                'Unexpected output type from the model: {}'.format(
                    type(output)))

        if not all([isinstance(o, chainer.Variable) for o in flat_outputs]):
            raise ValueError('The all \'outputs\' must be Chainer Variable')
        network_outputs = collections.OrderedDict(
            [(self._datastore.get_name(var), var) for var in flat_outputs])

        # Dump the Chainer Graph
        self._function_nodes = trace_graph_funcs(network_outputs.values())

        # Prepare relay graphs
        for node in self._function_nodes:
            self.convert_chainer_ops(node)

        # Outputs
        out = []
        for var in flat_outputs:
            out.append(self._datastore.nodes[self._datastore.get_name(var)])

        if len(out) > 1:
            outputs = _expr.Tuple(out)
        else:
            outputs = out[0]

        sym = _function.Function(analysis.free_vars(outputs), outputs)
        return IRModule.from_expr(sym), self._datastore.tvmparams


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
