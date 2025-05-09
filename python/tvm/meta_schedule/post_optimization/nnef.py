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

"""NNEF: Neural Network Exchange Format frontend for TVM relay"""
import os
import typing
import nnef
import numpy as np

import tvm
from tvm import relay
from tvm.ir import IRModule
from tvm.relay import expr as tvm_expr
from tvm.relay import analysis, function
from tvm.relay.frontend.common import new_var, fold_constant, set_span, infer_type

from .nnef_ops import _get_converter_map


def get_type(elem_type: str):
    """
    Gives numpy style type for nnef primitive types, uses x32 versions.

    :param elem_type: string, (scalar, integer, logical, string)
    :return: returns numpy dtype equivalent (float32, int32, bool, string)
    """
    if elem_type == "scalar":
        return "float32"
    if elem_type == "integer":
        return "int32"
    if elem_type == "logical":
        return "bool"
    if elem_type == "string":
        return "string"
    raise TypeError(f'Type "{elem_type}" is not implemented')


def make_parameter_span(source_name_list, name_sep="."):
    return name_sep.join(source_name_list)


# Converter class
class NNEFConverter:
    """
    Helper class for class level attributes, for conversion of NNEF model.
    Public method to use is from_nnef.

    Parameters
    ----------

    freeze_vars : bool, optional
        If this parameter is true, the nnef variables will be converted to
        constants, and be embedded into the relay model, allowing optimizations
        at compile time.

    """

    def __init__(self, freeze_vars=False):
        self._nodes = {}
        self._consts = {}
        self._inputs = {}
        self._num_inputs = 0
        self._params = {}
        self._num_params = 0
        self._freeze_vars = freeze_vars

    def from_nnef(self, graph: nnef.Graph) -> typing.Tuple[tvm.IRModule, dict]:
        """
        Convert an NNEF model into an equivalent TVM Relay IRModule.

        Parameters
        ----------
        graph : nnef.Graph
            An NNEF Graph object that was imported with nnef.load_graph.
            Shapes should be inferred by nnef.infer_shapes on graph beforehand.

        Returns
        -------
        mod : tvm.IRModule
            The relay module for compilation

        params : dict of str to tvm.nd.NDArray
            The parameter dictionary to be used

        """
        self._parse_inputs(graph)
        self._construct_nodes(graph)

        outputs = [self._nodes[n] for n in graph.outputs]
        outputs = outputs[0] if len(outputs) == 1 else tvm_expr.Tuple(outputs)

        nodes = {v: k for k, v in self._nodes.items()}
        free_vars = analysis.free_vars(outputs)
        free_vars = [nodes[var] for var in free_vars]
        for i_name in self._params.keys():
            if i_name in free_vars and i_name not in self._inputs:
                self._inputs[i_name] = self._nodes[i_name]
        func = function.Function(list(self._inputs.values()), outputs)
        return IRModule.from_expr(func), self._params

    def _parse_inputs(self, graph):
        """Save inputs into class from inputs attrib of graph"""
        for inp in graph.inputs:
            self._num_inputs += 1
            tensor = graph.tensors[inp]
            self._nodes[inp] = new_var(inp, shape=tensor.shape, dtype=get_type(tensor.dtype))
            self._inputs[inp] = self._nodes[inp]

    def _construct_nodes(self, graph):
        """Construct TVM relay calls from every operation of the nnef graph"""
        for op in graph.operations:
            if op.name == "external":
                # externals are handled as input, not needed,
                # but nnef treats them as operations as well
                continue

            if op.name == "variable":
                self._set_variable(graph.tensors[op.outputs["output"]])

            elif op.name == "constant":
                self._set_const(op)

            else:
                # every other operator can be grouped more easily,
                # as it does not need self for conversion
                self._set_operator(op)

    def _set_operator(self, node):
        self._set_literal_inputs(node)
        self._set_parameter_span(node, node.name)
        inputs = []
        for ink, inv in node.inputs.items():
            if isinstance(inv, list):
                for i, linv in enumerate(inv):
                    if linv in self._nodes.keys():
                        inputs.append(self._nodes[linv])
                    else:  # handle literal inputs
                        name = f"{node.name}_{ink}_{i}"
                        assert name in self._nodes, f"{name} has not been properly handled"
                        inputs.append(self._nodes[name])

            else:
                if inv in self._nodes.keys():
                    inputs.append(self._nodes[inv])
                else:  # handle literal inputs
                    name = f"{node.name}_{ink}"
                    assert name in self._nodes, f"{name} has not been properly handled"
                    inputs.append(self._nodes[name])

        converted = self._get_relay_op_call(node.name, inputs, node.attribs)

        if not isinstance(converted, tvm_expr.TupleWrapper):
            outputs_num = 1
        else:
            outputs_num = len(converted)

        if outputs_num == 1:
            if not isinstance(converted, tvm_expr.TupleWrapper):
                converted = fold_constant(converted)
            else:
                converted = fold_constant(converted.astuple())
        else:
            converted = tvm_expr.TupleWrapper(fold_constant(converted.astuple()), len(converted))

        converted = set_span(converted, node.name)

        if outputs_num == 1:
            # check if the singular ret val is a list of only one element
            ret_val = list(node.outputs.values())[0]
            if isinstance(ret_val, list):
                self._nodes[ret_val[0]] = converted
            else:
                self._nodes[ret_val] = converted
        else:
            for i, out in zip(range(outputs_num), node.outputs["values"]):
                self._nodes[out] = converted[i]

    def _set_const(self, node):
        """Create a tvm.relay.Constant from a nnef constant tensor"""
        name = node.outputs["output"]
        data = node.attribs["value"]
        shape = node.attribs["shape"]
        if len(data) == 1:
            data = np.full(shape, data, dtype=get_type(node.dtype))
        else:
            data = np.array(data, dtype=get_type(node.dtype))
        self._consts[name] = tvm_expr.const(data)
        self._nodes[name] = self._consts[name]

    def _set_variable(self, tensor):
        """Create a tvm.relay.Var (or Constant if freeze_vars) from a nnef variable tensor"""
        tens_data = tensor.data
        if self._freeze_vars:
            self._consts[tensor.name] = tvm_expr.const(tens_data)
            self._nodes[tensor.name] = self._consts[tensor.name]
        else:
            self._nodes[tensor.name] = new_var(
                tensor.name, shape=tensor.shape, dtype=get_type(tensor.dtype)
            )
            self._params[tensor.name] = tens_data

    def _set_literal_inputs(self, node):
        """Checks if node has literal inputs and saves them into a tvm.relay.Constant.
        naming as {node.name}_{input field name}"""
        for field_name, value in node.inputs.items():
            if isinstance(value, list):
                for v in value:
                    if v not in self._nodes.keys():
                        self._nodes[f"{node.name}_{v}"] = tvm_expr.const(v)

            else:
                if value not in self._nodes.keys():
                    self._nodes[f"{node.name}_{field_name}"] = tvm_expr.const(value)

    def _set_parameter_span(self, node, node_source_name):
        for field_name, name in node.inputs.items():
            if isinstance(name, list):
                for n in name:
                    self._set_par_span_helper(node, node_source_name, n, field_name)
            else:
                self._set_par_span_helper(node, node_source_name, name, field_name)

    def _set_par_span_helper(self, node, node_source_name, name, field_name):
        if name not in self._nodes.keys():
            name = f"{node.name}_{field_name}"

        expr = self._nodes.get(name)
        if expr:
            expr_with_span = set_span(expr, make_parameter_span([node_source_name, name]))
            self._nodes[name] = expr_with_span
            if name in self._inputs:
                self._inputs[name] = expr_with_span
            if isinstance(expr, relay.Constant):
                self._consts[name] = expr_with_span

    def _get_relay_op_call(self, name, inputs, attrs):
        """Returns the tvm.Call equivalent to the nnef operator"""
        conv_map = _get_converter_map()
        if name in conv_map:
            call = conv_map[name](*inputs, **attrs)
        else:
            # This error is reached if NNEF is expanded with additional ops
            raise NotImplementedError(
                f"Operator {name} is not implemented, as {name} has been added after 1.0.5."
            )
        return call

    def _infer_type(self, val):
        if isinstance(val, bool):
            return "bool", True
        if isinstance(val, float):
            return "float32", True
        if isinstance(val, int):
            return "int32", True
        if isinstance(val, str):
            # the string vals can be names of nodes in some of the cases
            if isinstance(val, nnef.Identifier):
                if val in self._nodes.keys():
                    node = self._nodes[val]
                    if isinstance(node, tvm_expr.Var):
                        return node.type_annotation.dtype, False
                    if isinstance(node, tvm_expr.Constant):
                        return node.data.dtype, False
                    if isinstance(node, tvm_expr.Call):
                        return infer_type(node).checked_type.dtype, False
                raise Exception(
                    f"{val} has not been loaded into the model "
                    "but it should have been, as a var or call."
                )
            return "string", True

        raise TypeError(f'Value "{val}" is not a recognized type')


def from_nnef(
    model: typing.Union[str, os.PathLike, nnef.Graph],
    freeze_vars: bool = False,
) -> typing.Tuple[IRModule, dict]:
    """
    Convert an NNEF model into an equivalent TVM Relay IRModule.


    Parameters
    ----------
    model : os.PathLike or str or nnef.Graph
        Path to an NNEF model directory, containing the graph.nnef (and weight files)

    freeze_vars : bool, optional
        If this parameter is true, the nnef variables will be converted to
        constants, and be embedded into the relay model, allowing optimizations
        at compile time.

    Returns
    -------
    mod : tvm.IRModule
        The relay module for compilation

    params : dict of str to tvm.nd.NDArray
        The parameter dictionary to be used
    """
    conv_clss = NNEFConverter(freeze_vars)

    if not isinstance(model, nnef.Graph):
        model = nnef.load_graph(model)

    # fills in the nnef graph's shape information
    nnef.infer_shapes(model)

    return conv_clss.from_nnef(graph=model)
