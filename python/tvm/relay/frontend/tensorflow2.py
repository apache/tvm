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
# pylint: disable=invalid-name, unused-argument, too-many-lines, len-as-condition, broad-except
"""Tensorflow2.x graph to relay converter.

If model is constructed using tf2.x API, then use this converter:
    from tvm.relay.frontend.tensorflow2 import from_tensorflow
Otherwise use the tf1.x converter:
    from tvm.relay.frontend.tensorflow import from_tensorflow

"""

import numpy as np
from tensorflow.python.framework import function_def_to_graph, tensor_util, dtypes

import tvm
from tvm.relay.transform import InferType
from tvm.relay.prelude import Prelude
from tvm.ir import IRModule
from .. import expr as _expr
from .. import analysis
from .. import function as _function
from ..loops import while_loop as _while_loop
from .common import infer_type as _infer_type

from .tensorflow_ops import _convert_map as _convert_map_common
from .tensorflow_ops import _need_prelude_for_shape_inference

from ..ty import Any

__all__ = ["from_tensorflow"]


def _infer_type_with_prelude(val, prelude):
    body = _infer_type(val, prelude.mod)
    return body.checked_type


def set_span(sym, node_name):
    """set span of symbol"""

    span = tvm.relay.Span(tvm.relay.SourceName(node_name), 0, 0, 0, 0)
    if isinstance(sym, _expr.Call):
        sym = _expr.Call(sym.op, sym.args, sym.attrs, sym.type_args, span)
    elif isinstance(sym, _expr.TupleWrapper):
        tuple_value = sym.tuple_value
        if isinstance(tuple_value, _expr.Call):
            tuple_value = _expr.Call(
                tuple_value.op, tuple_value.args, tuple_value.attrs, tuple_value.type_args, span
            )
            sym = _expr.TupleWrapper(tuple_value, sym.size)
    return sym


def convert_const_node(node, shape):
    """convert tf const node into relay const or var"""

    # get the value of the constant
    tensor_value = node.attr["value"].tensor
    np_array = tensor_util.MakeNdarray(tensor_value)

    if np_array.dtype == np.dtype(object):
        if shape and node.name in shape:
            var_shape = shape[node.name]
        else:
            var_shape = tensor_util.TensorShapeProtoToList(tensor_value.tensor_shape)
        param = None
        sym = [_expr.var(node.name, shape=var_shape, dtype="uint8")]
        return sym, param

    if len(np_array.shape) == 0:
        param = None
        sym = [tvm.relay.const(np_array, np_array.dtype)]
    else:
        param = tvm.nd.array(np_array)
        sym = [_expr.var(node.name, shape=param.shape, dtype=param.dtype)]

    return sym, param


def get_attr(buf):
    """convert value of a node attribute. node attribute is part of a node in a graph.

    Parameters
    ----------
    buf: attrvalue protobuf.  <class 'tensorflow.core.framework.attr_value_pb2.AttrValue'>

    Returns
    -------
    The value of the attr, as a Python object.

    Raises:
    -------
    ValueError: If this op does not have an attr with the given `name`.
    """

    fields = ["s", "i", "f", "b", "type", "shape", "tensor", "func"]

    ret = []

    if not buf.WhichOneof("value"):
        return ret

    if buf.HasField("list"):
        for f in fields:
            if getattr(buf.list, f):
                if f == "type":
                    ret += [dtypes.as_dtype(x) for x in list(getattr(buf.list, f))]
                else:
                    ret += list(getattr(buf.list, f))
    else:
        for f in fields:
            if buf.HasField(f):
                if f == "type":
                    ret = dtypes.as_dtype(getattr(buf, f))
                else:
                    ret = getattr(buf, f)
    return ret


def parse_attr(attr_proto):
    """Convert node attributes (a serialized map of key-value pairs) in a node to a dict

    Parameters
    ----------
    attr_proto: <class 'google.protobuf.pyext._message.MessageMapContainer'>

    Returns
    -------
    Dict {string: python object}

    """
    attrs = {}
    for key, value in attr_proto.items():
        attrs[key] = get_attr(value)

    return attrs


def convert_placeholder(shape, node, in_type=None):
    """convert tf placeholder into relay var.

    Example
    --------
    a tf placeholder with name "x" is converted to [Var(x, ty=TensorType([], float32))]
    """

    if shape and node.name in shape:
        input_shape = list(shape[node.name])
    else:
        input_shape = tensor_util.TensorShapeProtoToList(node.attr["shape"].shape)
        for idx, dim in enumerate(input_shape):
            if dim < 0:
                input_shape[idx] = Any()
    attr = parse_attr(node.attr)
    if in_type is not None:
        sym = [_expr.var(node.name, type_annotation=in_type)]
    else:
        sym = [_expr.var(node.name, shape=input_shape, dtype=attr["dtype"].name)]
    return input_shape, sym


class RelayModule:
    """states related to the entire relay module (multiple functions)
    after converted from tf graphdef"""

    def __init__(self):
        self.mod = IRModule({})
        self.params = {}
        self.prelude = Prelude(self.mod)


class GraphProto:
    """Capturing states when converting a tf graph to a single relay function."""

    def __init__(self, module):
        self._module = module
        self._prelude = self._module.prelude
        self._params = {}
        self._nodes = {}
        self._input_shapes = {}
        self._output_shapes = {}
        self._tf_node_map = {}
        self._gdef_lib = {}

    def from_tensorflow(
        self, graph, layout="NHWC", shape=None, outputs=None, input_types=None, gdef_lib=None
    ):
        """Wrapper to _get_relay_func which converts Tensorflow graph to Relay function
        which is used as main function for the Relay module
        """
        if input_types is None:
            input_types = {}

        if gdef_lib is None:
            gdef_lib = {}

        self._gdef_lib = gdef_lib
        func = self._get_relay_func(
            graph, layout=layout, shape=shape, outputs=outputs, input_types=input_types
        )
        return func, self._params

    def _get_relay_func(self, graph, layout="NHWC", shape=None, outputs=None, input_types=None):
        if input_types is None:
            input_types = {}

        self._layout = layout
        for node in graph.node:
            name = node.name
            self._tf_node_map[name] = node
            if node.op == "Placeholder":
                in_type = None
                if node.name in input_types:
                    in_type = input_types[node.name]
                self._input_shapes[name], self._nodes[name] = convert_placeholder(
                    shape, node, in_type
                )
            elif node.op == "Const":
                sym, param = convert_const_node(node, shape)
                self._nodes[node.name] = sym
                if param:
                    self._params[node.name] = param
        for node in graph.node:
            self._backtrack_construct(graph, node.name)

        return self._func(graph, outputs)

    def _func(self, graph, outputs):
        out = []
        if outputs is None:
            last_node = graph.node[-1]
            op = self._nodes[last_node.name.split(":")[0]]
            if last_node.op == "Exit":
                out = [op[0].tuple_value]
            else:
                out = op
        else:
            for out_name in outputs:
                if ":" in out_name:
                    out_name = out_name.split(":")
                    out_name, out_num = out_name[0], out_name[-1]
                    out_num = int(out_num)
                    out.append(self._nodes[out_name][out_num])
                else:
                    out.append(self._nodes[out_name][0])

        if isinstance(out, _expr.TupleWrapper):
            out = out.astuple()
        else:
            out = out[0] if len(out) == 1 else _expr.Tuple(out)

        fvars = analysis.free_vars(out)
        func = _function.Function(fvars, out)
        final_params = {}
        for fv in fvars:
            if fv.name_hint in self._params:
                final_params[fv.name_hint] = self._params[fv.name_hint]
        self._params = final_params
        return func

    def _convert_operator(self, graph, op_name, node_name, inputs, attrs):
        """Convert from Tensorflow operator to relay operator.
        The converter must specify conversions explicitly for incompatible name, and
        apply handlers to operator attributes.

        Parameters
        ----------
        graph: <class 'tensorflow.core.framework.graph_pb2.GraphDef'>
            TF2 frozen graph def
        op_name : str
            Operator name, such as Conv2D, AvgPool
        node_name: str
             Name of the node in TF2 graph, such as Identity:0
        inputs : list of relay.op
            List of input symbols.
        attrs : dict
            Dict of operator attributes

        Returns
        -------
        sym : relay.op
            Converted relay operator
        """
        if op_name in ["PartitionedCall", "StatefulPartitionedCall"]:
            sym = _partition_call_operator(
                self._module,
                graph,
                inputs,
                attrs,
                self._prelude,
                gdef_lib=self._gdef_lib,
            )
        elif op_name in ["StatelessIf", "If"]:
            sym = _convert_if(
                self._module, graph, inputs, attrs, self._prelude, gdef_lib=self._gdef_lib
            )
        elif op_name in ["StatelessWhile", "While"]:
            sym = _convert_loop(
                self._module,
                graph,
                inputs,
                attrs,
                node_name,
                self._tf_node_map,
                self._prelude,
                gdef_lib=self._gdef_lib,
            )
        elif op_name in _convert_map_common:
            if _need_prelude_for_shape_inference(op_name):
                sym = _convert_map_common[op_name](inputs, attrs, self._params, self._prelude)
            else:
                sym = _convert_map_common[op_name](inputs, attrs, self._params, self._module.mod)
        else:
            raise NotImplementedError("Operator {} not implemented.".format(op_name))

        sym = set_span(sym, node_name)
        return sym

    def _backtrack_construct(self, graph, node_name):
        """Convert a specific tensorflow node to relay expression.

        If any of its ancestor node is not converted yet, backtrack as
        far as input node and covert all nodes on the path. resurion is used here.

        This is required when parsing control flow nodes, since the parsing
        order may not follow the original graph def.

        to discover input node, current tf node's input is iterated:

        tensorflow/core/framework/node_def.proto
            message NodeDef {
                repeated string input = 3;
            }

        a node has many inputs (other nodes). each input has the following format:
            data input is "node:src_output".  node is the string name.
            control input is "^node".

        Parameters
        ----------
        graph : <class 'tensorflow.core.framework.graph_pb2.GraphDef'>
            TF2 frozen graph def

        node_name : str
            node name

        Returns
        -------
        op : relay.Expr
            Converted relay expression.

        Examples
        --------
        tf expression "x+1" is converted to relay expression:
            CallNode(Op(add), [Var(x, ty=TensorType([], float32)), Constant(1.0)], (nullptr), [])

        """

        input_op_name = node_name.split(":")[0].split("^")[-1]
        if input_op_name not in self._nodes:
            node = self._tf_node_map[input_op_name]
            attr = parse_attr(node.attr)
            if "_output_shapes" in attr:
                self._output_shapes[node.name] = [
                    tensor_util.TensorShapeProtoToList(tshape) for tshape in attr["_output_shapes"]
                ]
            else:
                self._output_shapes[node.name] = [None]

            attr["_output_shapes"] = self._output_shapes[input_op_name]
            attr["_node_name"] = node.name
            attr["_target_layout"] = self._layout
            inputs = [self._backtrack_construct(graph, iname) for iname in node.input]
            op = self._convert_operator(graph, node.op, node.name, inputs, attr)

            if isinstance(op, np.ndarray):
                self._params[node.name] = tvm.nd.array(op)
                op = [
                    _expr.var(
                        node.name,
                        shape=self._params[node.name].shape,
                        dtype=self._params[node.name].dtype,
                    )
                ]
            elif isinstance(op, (_expr.Expr, _expr.TupleGetItem)):
                op = [op]
            self._nodes[input_op_name] = op

        out = self._nodes[input_op_name]
        if isinstance(out, _expr.TupleWrapper):
            tn = node_name.split(":")
            tensor_slot = int(tn[1]) if len(tn) > 1 else 0
            return out[tensor_slot]

        return out[0]


def _partition_call_operator(module, graph, inputs, attr, prelude, gdef_lib):
    """convert tf PartitionedCall node to a relay function call"""
    node_func_name = attr.get("f").name
    return _convert_function(
        module, graph, inputs, attr, node_func_name, prelude, gdef_lib=gdef_lib
    )


def _convert_if(module, graph, inputs, attr, prelude, gdef_lib):
    """Convert tf If/StatelessIf to Relay If"""
    cond_expr = inputs[0]
    branch_names = [attr.get(x).name for x in ["then_branch", "else_branch"]]
    then_fn, else_fn = [
        _convert_function(module, graph, inputs[1:], attr, name, prelude, gdef_lib=gdef_lib)
        for name in branch_names
    ]
    out = _expr.If(cond_expr, then_fn, else_fn)
    return out


def _convert_loop(module, graph, inputs, attr, node_name, nodes, prelude, gdef_lib):
    """convert tf while_loop to Relay loop"""
    input_size = len(inputs)
    cond_fn_name, body_fn_name = [attr.get(x).name for x in ["cond", "body"]]

    def convert_vars(loop_inputs, input_signature):
        """convert inputs to relay vars to be used as loop variables
        Loop inputs are packed as:
            [iteration_number, max_iterations, loop_variables...]
        """
        new_vars = []
        for i, v in enumerate(loop_inputs):
            if isinstance(v, _expr.Constant):
                vtype = _infer_type(v).checked_type.dtype
                new_vars.append(_expr.var(input_signature[i].name, shape=(), dtype=vtype))
            else:
                vtype = _infer_type_with_prelude(v, prelude)
                new_vars.append(_expr.var(input_signature[i].name, type_annotation=vtype))
        return new_vars

    while_func = next(
        (f for f in graph.library.function if f.signature.name == attr["body"].name),
        None,
    )
    loop_inputs = convert_vars(inputs, while_func.signature.input_arg)

    def cond_fn(*loop_inputs):
        return _convert_function(
            module, graph, loop_inputs, attr, cond_fn_name, prelude, gdef_lib=gdef_lib
        )

    # Define the loop body, in this function we need to unpack loop inputs,
    # convert the loop subgraph, and pack outputs for the next iteration.
    def body_fn(*loop_inputs):
        # Increment loop iteration counter
        loop_count = loop_inputs[0] + _expr.const(1, dtype="int32")
        max_count = loop_inputs[1]
        fn = _convert_function(
            module, graph, loop_inputs, attr, body_fn_name, prelude, gdef_lib=gdef_lib
        )

        # Repack loop variables
        out = [loop_count, max_count] + [_expr.TupleGetItem(fn, i) for i in range(2, input_size)]
        return out

    loop = _while_loop(cond_fn, loop_inputs, body_fn)
    outputs = loop(*inputs)
    outputs = _expr.TupleWrapper(
        _expr.Tuple([_expr.TupleGetItem(outputs, i) for i in range(input_size)]), input_size
    )
    return outputs


def _convert_function(
    module, graph, inputs, attr, node_func_name, prelude, gdef_lib, in_shapes=None
):
    """Convert given tf node to a relay function call

    Parameters
    ----------
    module : IRModule
        where converted function is stored

    graph: <class 'tensorflow.core.framework.graph_pb2.GraphDef'>
        top level tf graphdef

    inputs : List[tvm.relay.Expr]
        List of input symbols. Parameters for the function.

    attrs : Dict[tvm.Attrs]
        Dict of operator attributes.

    node_func_name : str
        Name of tf2 node to be converted

    Returns
    -------
    op : tvm.relay.Expr
        <class 'tvm.relay.expr.Call'>

    Examples
    --------
    a tf function "x+1", is implemented as a subgraph in the library section of the graph.
    this subgraph is converted to a relay function such as
        fn (%x: float32) {
        add(%x, 1f) /* Identity */
        }

    the subgraph has a function name such as __inference_add_95
    the tf function call operator is returned as relay expression, such as:
        free_var %x: float32;
        @func___inference_add_95(%x)

    """
    func = next(
        (f for f in graph.library.function if f.signature.name == node_func_name),
        None,
    )
    if func is None:
        raise Exception("Function not found - {}".format(node_func_name))
    devices = set(node.device for node in func.node_def)
    if len(devices) > 1:
        raise Exception(
            "node_def in function {} contains > 1 types of devices {}".format(
                node_func_name, devices
            )
        )

    subgraph = gdef_lib[node_func_name]
    # preserve library functions in subgraphs to make them available to nested functions
    for fn in graph.library.function:
        subgraph.library.function.add().CopyFrom(fn)

    # Computing subgraph's input shape and type dictionaries
    input_expr_dict = {}
    input_types = {}
    for f_arg, input_ in zip(func.signature.input_arg, inputs):
        input_expr_dict[f_arg.name] = input_
        input_types[f_arg.name] = _infer_type_with_prelude(input_, prelude)

    func_name = "func_{}".format(func.signature.name)
    try:
        global_func = module.mod[func_name]
        sub_func = global_func
        sub_params = module.params
    except ValueError:
        # Construct relay nodes from the subgraph
        g1 = GraphProto(module)
        output_sig = [func.ret[f.name] for f in func.signature.output_arg]
        # TODO: unify prelude and main IRModules
        sub_func, sub_params = g1.from_tensorflow(
            subgraph, outputs=output_sig, input_types=input_types, gdef_lib=gdef_lib
        )
        module.params.update(sub_params)
        func_expr = _function.Function(sub_func.params, sub_func.body)
        global_func = tvm.relay.GlobalVar(func_name)
        module.mod[global_func] = func_expr
        module.mod = InferType()(module.mod)
        prelude.mod = module.mod

    param_exprs = []
    for param_expr in sub_func.params:
        # sub_params is subset of sub_func.params
        param_name = param_expr.vid.name_hint
        if param_name in input_expr_dict.keys():
            param_exprs.append(input_expr_dict[param_name])
        elif param_name in sub_params.keys():
            param_exprs.append(param_expr)
        else:
            raise Exception("Input parameter {} not found".format(param_name))

    sb = tvm.relay.scope_builder.ScopeBuilder()
    loop_ret = global_func(*param_exprs)
    sb.ret(loop_ret)
    ret = sb.get()
    return ret


def from_tensorflow(graph_def, layout="NHWC", shape=None, outputs=None):
    """convert tensorflow2.x graph into relay function.

    Parameters
    ----------
    graph_def : must be frozen graph (no variables allowed).
        Placeholders are assumed to be inputs to the graph.

        tensorflow/core/framework/graph.proto
            message GraphDef {
              repeated NodeDef node = 1;
              FunctionDefLibrary library = 2;
            }
        tensorflow/core/framework/function.proto
            message FunctionDef {
              repeated NodeDef node_def = 3;
            }

    layout : str
        The layout for the model.

    shape : List[str, List[int]]
        Input to the model. It is a key and shape vector mapping. Applies to placeholders.

    outputs : List[str]
        The list of output nodes. The last node is treated as the output if not
        specified.

    Returns
    -------
    mod : tvm.IRModule
        The module that optimizations will be performed on.

    params : dict of str to tvm.nd.NDArray
        Dict of converted parameters stored in tvm.nd.NDArray format.

    Examples
    --------
    "x+1" tf module where x has a shape of (2,2) is converted as follows:

    mod : tvm.IRModule
        def @func___inference_add_95(%x: Tensor[(2, 2), float32], %add/y: Tensor[(2, 2), float32])
        -> Tensor[(2, 2), float32] {
        add(%x, %add/y) /* Identity */ /* ty=Tensor[(2, 2), float32] */
        }

        def @main(%x1: Tensor[(2, 2), float32], %add/y1: Tensor[(2, 2), float32]) {
        @func___inference_add_95(%x1, %add/y1) /* Identity */
        }

    params : dict of str to tvm.nd.NDArray
        {'add/y': <tvm.nd.NDArray shape=(2, 2), cpu(0)>

    """

    # Subgraph graph_defs are cached here to avoid a TF error when parsing after prelude init
    graph_def_library = {}
    for func in graph_def.library.function:
        inshape = func.attr["_input_shapes"].list.shape
        graph_def_library[func.signature.name], _ = function_def_to_graph.function_def_to_graph_def(
            func, inshape
        )
    module = RelayModule()
    g = GraphProto(module)
    func, params = g.from_tensorflow(graph_def, layout, shape, outputs, gdef_lib=graph_def_library)
    module.mod["main"] = func
    module.params.update(params)
    return module.mod, module.params
