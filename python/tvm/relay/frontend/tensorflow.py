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
# pylint: disable=import-self, invalid-name, unused-argument, too-many-lines, len-as-condition, broad-except
# pylint: disable=import-outside-toplevel, redefined-builtin
"""TF: Tensorflow frontend."""
import warnings
from collections import defaultdict

# Numpy support
import numpy as np
import tvm

from tvm.ir import IRModule
from tvm.relay.prelude import Prelude
from tvm.relay.transform import InferType

from .. import analysis
from .. import expr as _expr
from .. import function as _function
from ..ty import Any
from ..expr_functor import ExprMutator, ExprVisitor
from .common import get_relay_op
from .common import infer_type as _infer_type
from .common import infer_shape as _infer_shape
from .common import infer_value as _infer_value
from .common import set_span

from .tensorflow_ops import _convert_map
from .tensorflow_ops import _need_prelude_for_shape_inference
from .tensorflow_ops import _get_more_static_shape

__all__ = ["from_tensorflow"]

# The default configurations of Relay TensorFlow frontend.
TF_DEFAULT_CONFIGS = {
    # By default, TVM converts `tf.matmul` to `transpose(weight) + nn.dense`, which introduces
    # unnecessary overhead in weight transpose. Change this flag to False to directly convert to
    # `nn.matmul` to get rid of the overhead.
    # However, please note that `nn.matmul` is in experimental so it may have some performance
    # issues.
    "use_dense": True,
    # By default, TVM converts `tf.batch_matmul` to `transpose(weight) + nn.batch_matmul_NT`.
    # Change this flag to False to directly convert to `nn.batch_matmul`.
    # Note that `nn.batch_matmul` with format other than NT is in experimental, it may have some
    # performance issues.
    "use_nt_batch_matmul": True,
}

# compatible operators that do NOT require any conversion.
_identity_list = []

# Operators that get pruned away when the complete graph is frozen.
# These operators are not needed for inference.
_freezed_graph_pruned_op_list = [
    "ReadVariableOp",
    "ResourceGather",
    "Variable",
    "VariableV2",
    "VarHandleOp",
    "Assign",
    "AssignVariableOp",
]

# An internal list to contain all the control flow primitives used in Tensorflow
# 1.x.
_control_flow_nodes = ["Merge", "Switch", "NextIteration", "Exit", "Enter", "LoopCond"]

# A map to record tensor array write ops and input ta/tensor indices
# Value is (index of tensor array, index of written node)
_tensor_array_write_ops = {
    "TensorArrayWrite": (3, 2),
    "TensorArrayScatter": (0, 2),
    "TensorArraySplit": (0, 1),
}


def is_tensor_array_constuctor(tf_node):
    """Check whether is tensor array constructor node."""
    is_ta = False
    ta_start = "TensorArrayV"
    if tf_node.op.startswith(ta_start):
        is_ta = tf_node.op[len(ta_start)].isnumeric()
    return is_ta


def find_parent_loop_name(node_name, while_loop_name_set):
    """Find name of direct parent while loop."""
    ploop_name = ""
    name_prefix = node_name.rsplit("/", 1)[0]
    if name_prefix.startswith("^"):
        name_prefix = name_prefix[1:]
    for lname in while_loop_name_set:
        if name_prefix.startswith(lname) and len(ploop_name) < len(lname):
            ploop_name = lname

    if len(ploop_name) == 0:
        ploop_name = name_prefix

    return ploop_name


def _in_while_loop(control_flow_node_map, op_name):
    """
    Check if a given control flow operator is part of a while loop execution
    frame. This is based on the fact that there is only one occurrence of
    `LoopCond` for a loop execution frame and it is only presented in the loop
    construct.

    Parameters
    ----------
    control_flow_node_map : Dict[str, Set[str]]
        A dictionary contains the unique control flow execution frame name to
        a set of primitive operators mapping.

    op_name : str
        The name of a control flow primitive.

    Returns
    -------
    ret : bool
        Return true if the operator is in a while loop execution frame,
    otherwise, return false.
    """
    return op_name in control_flow_node_map and "LoopCond" in control_flow_node_map[op_name]


class RewriteSubgraph(ExprMutator):
    """
    A helper class to rewrite expr in while loop function to variable.

    Parameters
    ----------
    rewrite_map : Dict[expr, expr]
        A dictionary contains a set of expr to var mapping.
    """

    def __init__(self, rewrite_map):
        ExprMutator.__init__(self)
        self.rewrite_map = rewrite_map

    def visit(self, expr):
        if expr in self.rewrite_map:
            return self.rewrite_map[expr]
        return super().visit(expr)


def rewrite_subgraph(expr, rewrites):
    """Rewrite loop body."""
    return RewriteSubgraph(rewrites).visit(expr)


class Branch:
    """A class contains the components that are used to build up a Relay if
    node.

    Parameters
    ----------
    cond : tvm.relay.Expr
        The condition of a if node.

    true_branch : tvm.relay.Expr
        The body of the true branch of a if expression.

    false_branch: tvm.relay.Expr
        The body of the false branch of a if expression.

    _if : tvm.relay.Expr
        An internal variable indicates where an if expression is already created
        for a matched TF condition construct.

    Examples
    --------
    The following is a cond statement written in TensorFlow:

    .. code-block:: python

        def vanilla_cond():
            i = tf.constant(1)
            j = tf.constant(4)

             def f1():
                return tf.multiply(1, 17)

             def f2():
                return tf.add(4, 23)
            r = tf.cond(tf.less(i, j), f1, f2)

    This condition statement should be converted into Relay in the following
    form:

    .. code-block:: python

        fn (%Const: Tensor[(1,), int32],
            %Const_1: Tensor[(1,), int32],
            %cond/Mul/x: Tensor[(1,), int32],
            %cond/Mul/y: Tensor[(1,), int32],
            %cond/Add/x: Tensor[(1,), int32],
            %cond/Add/y: Tensor[(1,), int32]) {
          %0 = less(%Const, %Const_1) # ty=Tensor[(1,), bool]
          %1 = min(%0)
          if (%1) {
            %2 = multiply(%cond/Mul/x, %cond/Mul/y)
            %2
          }  else {
            %3 = add(%cond/Add/x, %cond/Add/y)
            %3
          }
        }
    """

    def __init__(self):
        self._if = None
        self.cond = None
        self.true_branch = None
        self.false_branch = None

    def _if_node(self):
        """An internal API to create a relay if node from the matched TF
        condition construct.
        """
        # `cond`  returns a tensor that contains boolean values. We add a `min`
        # operator to checks if there is any false value. If so, this condition
        # doesn't not hold.
        cond = tvm.relay.op.min(self.cond)
        return tvm.relay.If(cond, self.true_branch, self.false_branch)

    def if_node(self):
        """Create an tvm.relay.If node if it hasn't been created yet."""
        if self._if is None:
            self._if = self._if_node()
        return self._if


class VarChecker(ExprVisitor):
    """Check whether a Variable is used in loop body.

    Parameters
    ----------
    var : relay.expr.Var
        Relay Variable to be checked.
    """

    def __init__(self, var):
        ExprVisitor.__init__(self)
        self._var = var
        self.used = False

    def visit(self, expr):
        if self._var == expr:
            self.used = True
        super().visit(expr)


class Loop:
    """
    A class contains the components that are used to build up a Relay
    recursive call.
    Parameters
    ----------
    mod : tvm.IRModule
        Module for current parsed IR.

    loop_name : str
        Name prefix of while loop in TensorFlow graph.

    lvar2expr : dict from str to dict from Relay.expr.Var to Relay.expr
        A dictionary recording all loop vars and corresponding
        relay expression.

    Examples
    --------
    The following is a vanilla loop from TensorFlow:
    .. code-block:: python
        i = tf.constant(0)
        c = lambda i: tf.less(i, 10)
        b = lambda i: tf.add(i, 1)
        r = tf.while_loop(c, b, [i])
    It will be converted to the following recursive call in Relay:
    .. code-block:: python
        fn (%while/Less/y: Tensor[(1,), int32],
            %while/Add/y: Tensor[(1,), int32],
            %Const: Tensor[(1,), int32]) {
          %0 = fn(%loop_var0: Tensor[(1,), int32]) {
            %1 = less(%loop_var0, %while/Less/y)
            %2 = min(%1)
            if (%2) {
              %3 = add(%loop_var0, %while/Add/y)
              free_var %while_loop
              %4 = %while_loop(%3)
              %4
            }    else {
              %5 = (%loop_var0,)
              %5
            }
          }
          let %while_loop1 = %0
          %6 = %while_loop1(%Const)
          %6
        }
    """

    def __init__(self, mod, loop_name, lvar2expr):
        self.cond = None
        self.body = []
        self._loop = None
        self._mod = mod
        self._loop_name = loop_name
        self._lvar2expr = lvar2expr
        self.loop_vars = []

        self.aligned = False

    def _while_loop(self):
        """An internal API to create a Relay recursive call for a matched TF
        `while_loop` construct.
        """
        bind_map = {}
        wl = set_span(tvm.relay.var("while_loop"), self._loop_name)
        sb = tvm.relay.scope_builder.ScopeBuilder()

        lv_list = []
        expr_list = []
        extra_vars = []

        for i, lv in enumerate(self.loop_vars):
            if self._loop_name not in self._lvar2expr:
                self._lvar2expr[self._loop_name] = {}

            # Handle the case when loop var is not properly lifted.
            # This can happen when loop var node name is set accidentally
            # beginning with loop name.
            if lv not in self._lvar2expr[self._loop_name]:
                var_name = f"{self._loop_name}_loop_var_{i}"
                var_type = _infer_type(lv, self._mod).checked_type
                loop_var = set_span(tvm.relay.var(var_name, type_annotation=var_type), var_name)
                self._lvar2expr[self._loop_name][loop_var] = lv
                bind_map[lv] = loop_var
                self.loop_vars[i] = loop_var
                lv = loop_var

            lv_list.append(lv)
            expr_list.append(self._lvar2expr[self._loop_name][lv])

        if bind_map:
            self.cond = rewrite_subgraph(self.cond, bind_map)
            self.body = [rewrite_subgraph(b, bind_map) for b in self.body]

        cond = set_span(tvm.relay.op.min(self.cond), self.cond.span)

        for lv, exp in self._lvar2expr[self._loop_name].items():
            if lv not in self.loop_vars:
                var_checker = VarChecker(lv)
                for bd in self.body + [cond]:
                    var_checker.visit(bd)
                    if var_checker.used:
                        lv_list.append(lv)
                        expr_list.append(exp)
                        extra_vars.append(lv)
                        break

        with sb.if_scope(cond):
            sb.ret(wl(*list(self.body + extra_vars)))
        with sb.else_scope():
            sb.ret(tvm.relay.Tuple(lv_list))

        loop_fn = tvm.relay.Function(lv_list, sb.get())
        sb = tvm.relay.scope_builder.ScopeBuilder()
        sb.let(wl, loop_fn)
        loop_ret = wl(*expr_list)

        sb.ret(loop_ret)
        ret = sb.get()
        return ret

    def while_loop(self):
        """Instantiate a while loop if it has not been created yet."""
        if self._loop is None:
            self._loop = self._while_loop()
            return self._loop
        return self._loop


class GraphProto(object):
    """A helper class for handling relay graph copying from Tensorflow GraphDef.
    Definition:
        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/graph.proto
    """

    def __init__(self):
        self._nodes = {}
        self._tf_node_map = {}
        self._params = {}
        self._input_shapes = {}
        self._output_shapes = {}
        self._num_rnn_layer = False
        self._input_shapes = {}
        self._loops = {}
        self._branches = {}
        self._mod = IRModule({})
        self._prelude = Prelude(self._mod)
        self._control_flow_node_map = defaultdict(set)
        self._loop_body_order = {}
        self._loop_var_order = {}
        self._lvar2expr = {}
        self._lname_map = {}
        self._sorted_cf_node_names = []
        self._while_loop_name_set = set()
        self._main_graph_proto = self
        self._tensor_array_shapes = {}
        self._tensor_array_shape_nodes = {}

    def _get_relay_func(self, graph, layout="NHWC", shape=None, outputs=None):
        """Construct relay nodes from tensorflow graph definition - GraphDef.

        Follow the tensorflow graph definition to parse and convert it to Relay.
        Some of the assumptions listed below.

            -> All Placeholders are considered as graph input.
            -> All Const nodes are params.
            -> Last node is assumed as graph output.
            -> _output_shapes : Graph should be frozen with add_shapes=True.
                                Or user can pass input shape dictionary optionally.
            -> DecodeJpeg, ResizeBilinear: These are dummy operators.
                                           Hence user should handle preprocessing outside.
            -> CheckNumerics: No implementation as of now for this.
                              Just copies input to output.

        Parameters
        ----------
        graph : tensorflow graph definition object
            The loaded tensorflow GraphDef

        layout : target layout to be used (Optional)
            NCHW only supported now to enable NHWC models on GPU.

        shape : Dictionary of input dimensions (Optional)
            Graph level input shape dictionary.

        outputs : List of output tensor names (Optional)
            if not specified then the last node is assumed as graph output.

        Returns
        -------
        mod : tvm.IRModule
            The module that optimizations will be performed on.

        params : dict
            A dict of name: tvm.nd.array pairs, used as pretrained weights
        """
        try:
            from tensorflow.python.framework import tensor_util
        except ImportError as e:
            raise ImportError(f"Unable to import tensorflow which is required {e}")

        missing_operators = self._parse_import_prerequisites(graph)
        control_flow_nodes = []
        ta_write_nodes = []
        ta_gather_nodes = []
        ta_construct_nodes = []
        self._in_shape = shape
        self._layout = layout
        self._graph = graph

        if missing_operators:
            freezed_ops = [op for op in missing_operators if op in _freezed_graph_pruned_op_list]
            if freezed_ops:
                raise Exception(
                    f"Graph is not frozen. Provide a frozen graph. "
                    f"Found operators {freezed_ops}"
                )

            raise NotImplementedError(
                f"The following operators are not implemented: {missing_operators}"
            )

        for node in graph.node:
            node_name_prefix = node.name.rsplit("/", 1)[0]
            self._control_flow_node_map[node_name_prefix].add(node.op)
            self._tf_node_map[node.name] = node

            # Parse output_shapes attribute
            parsed_attr = self._parse_attr(node.attr)
            if "_output_shapes" in parsed_attr:
                self._output_shapes[node.name] = [
                    tensor_util.TensorShapeProtoToList(tshape)
                    for tshape in parsed_attr["_output_shapes"]
                ]
            else:
                self._output_shapes[node.name] = [None]

            # Parse placeholder and const here since input shape info is required.
            if node.op == "Placeholder" or node.op == "PlaceholderWithDefault":
                # Give priority to user argument.
                if shape and node.name in shape:
                    self._input_shapes[node.name] = list(shape[node.name])
                else:
                    self._input_shapes[node.name] = tensor_util.TensorShapeProtoToList(
                        node.attr["shape"].shape
                    )
                    for idx, dim in enumerate(self._input_shapes[node.name]):
                        if dim < 0:
                            self._input_shapes[node.name][idx] = Any()

                self._output_shapes[node.name] = [self._input_shapes[node.name]]
                attr = self._parse_attr(node.attr)
                self._nodes[node.name] = [
                    set_span(
                        _expr.var(
                            node.name, shape=self._input_shapes[node.name], dtype=attr["dtype"].name
                        ),
                        node.name,
                    )
                ]

                # Ignore user's input shape for Non placeholder
            elif node.op == "Const":
                tensor_value = node.attr["value"].tensor
                self._input_shapes[node.name] = tensor_util.TensorShapeProtoToList(
                    tensor_value.tensor_shape
                )
                self._output_shapes[node.name] = [self._input_shapes[node.name]]
                if shape and node.name in shape:
                    warnings.warn(
                        f"Ignore the passed shape. Shape in graphdef "
                        f"will be used for operator {node.name}."
                    )
                for key, value in node.attr.items():
                    self._parse_param(key, value, node.name, self._in_shape)
            elif node.op in _control_flow_nodes:
                # We assume that the direct parent node of Exit is a while loop block
                if node.op == "Exit":
                    self._while_loop_name_set.add(node_name_prefix)
                control_flow_nodes.append(node)
            elif node.op.startswith("TensorArray"):
                if is_tensor_array_constuctor(node):
                    ta_construct_nodes.append(node)
                else:
                    for ta_write_name, idx in _tensor_array_write_ops.items():
                        if node.op.startswith(ta_write_name):
                            ta_write_nodes.append((node, idx))
                            break
                    if node.op.startswith("TensorArrayGather"):
                        ta_gather_nodes.append(node)

        # Use tensor array gather to infer static tensor array shape
        for gather_node in ta_gather_nodes:
            input_ta_name = gather_node.input[0]
            input_ta_node = self._tf_node_map[input_ta_name]
            if is_tensor_array_constuctor(input_ta_node):
                gather_attr = self._parse_attr(gather_node.attr)
                if "element_shape" not in gather_attr:
                    continue
                raw_elem_shape = tensor_util.TensorShapeProtoToList(gather_attr["element_shape"])
                elem_shape = []
                for dim in raw_elem_shape:
                    if dim < 0:
                        elem_shape.append(Any())
                    else:
                        elem_shape.append(int(dim))
                self._tensor_array_shapes[input_ta_node.name] = elem_shape

        # Fetch node contains static tensor array shape
        for item in ta_write_nodes:
            wnode = item[0]
            ta_idx, inode_idx = item[1]

            stack = [self._tf_node_map[wnode.input[ta_idx].split(":")[0]]]
            while stack:
                cnode = stack.pop(0)
                if not cnode.op.startswith("TensorArray"):
                    for iname in cnode.input:
                        stack.append(self._tf_node_map[iname.split(":")[0]])
                elif cnode.name != wnode.name:
                    if is_tensor_array_constuctor(cnode):
                        inode = self._tf_node_map[wnode.input[inode_idx].split(":")[0]]
                        tn = wnode.input[inode_idx].split(":")
                        output_index = int(tn[1]) if len(tn) > 1 else 0
                        self._tensor_array_shape_nodes[cnode.name] = (inode, wnode.op, output_index)
                    break

        # First, parse all control flow nodes.
        # Convert tf.cond to Branch and tf.while_loop to Loop.
        sorted_cf_nodes = []
        exit_pos_map = {}
        ordered_prefix = []
        # Sort control flow nodes to move all Exit nodes to the end
        # of corresponding while_loop block.
        for node in control_flow_nodes:
            loop_name = find_parent_loop_name(node.name, self._while_loop_name_set)
            if node.op == "Exit":
                if loop_name not in exit_pos_map:
                    ordered_prefix.append(loop_name)
                    exit_pos_map[loop_name] = len(sorted_cf_nodes)
                sorted_cf_nodes.append(node)
            elif loop_name in self._while_loop_name_set:
                if loop_name not in exit_pos_map:
                    sorted_cf_nodes.append(node)
                else:
                    sorted_cf_nodes.insert(exit_pos_map[loop_name], node)
                    for j in range(ordered_prefix.index(loop_name), len(ordered_prefix)):
                        exit_pos_map[ordered_prefix[j]] += 1
            else:
                sorted_cf_nodes.append(node)

        for node in sorted_cf_nodes:
            self._sorted_cf_node_names.append(node.name)

        for node in sorted_cf_nodes:
            self._backtrack_construct(node.name)

        # Second, parse other nodes to re-create TF graph using Relay operators.
        for node in graph.node:
            self._backtrack_construct(node.name)

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
                    out_name, out_num = out_name.split(":")
                    out_num = int(out_num)
                    out.append(self._nodes[out_name][out_num])
                else:
                    out.append(self._nodes[out_name][0])

        if isinstance(out, _expr.TupleWrapper):
            out = out.tuple_value
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

    def from_tensorflow(self, graph, layout="NHWC", shape=None, outputs=None):
        """Wrapper to _get_relay_func which converts Tensorflow graph to Relay function
        which is used as main function for the Relay module
        """
        func = self._get_relay_func(graph, layout=layout, shape=shape, outputs=outputs)
        self._mod["main"] = func
        return self._mod, self._params

    def _parse_import_prerequisites(self, graph):
        """Calculate the named preconditions from TensorFlow `graph`.
        Return prerequisites for parsing:
        a. Set of operator names which don't have their mapping in TVM, i.e.
            which are not supported
        """
        missing_operators = set()
        from tensorflow.python.framework import op_def_registry

        for node in graph.node:
            getOpDef = (
                op_def_registry._registered_ops.get
                if hasattr(op_def_registry, "_registered_ops")
                else op_def_registry.get
            )
            op_def = getOpDef(node.op)
            if node.op == "Placeholder" or node.op == "PlaceholderWithDefault":
                pass
            elif node.op == "Const":
                pass
            elif node.op in ["PartitionedCall", "StatefulPartitionedCall"]:
                pass
            else:
                if any([node.op in t for t in [_identity_list, _convert_map, _control_flow_nodes]]):
                    pass
                elif op_def is not None and op_def.is_stateful:
                    missing_operators.add(node.op)
                else:
                    missing_operators.add(node.op)

        return missing_operators

    def _parse_param(self, key, value, name, shape):
        try:
            from tensorflow.python.framework import tensor_util
        except ImportError as e:
            raise ImportError(f"Unable to import tensorflow which is required {e}")

        if key == "value":
            np_array = tensor_util.MakeNdarray(value.tensor)

            if np_array.dtype == np.dtype(object):
                # Object types are generally tensorflow DT_STRING (DecodeJpeg op).
                # Just leave it as placeholder.
                if shape and name in shape:
                    var_shape = shape[name]
                else:
                    var_shape = tensor_util.TensorShapeProtoToList(value.tensor.tensor_shape)
                self._nodes[name] = [
                    set_span(_expr.var(name, shape=var_shape, dtype="uint8"), span=name)
                ]
                return

            array_ndim = len(np_array.shape)
            if array_ndim == 0:
                self._nodes[name] = [set_span(tvm.relay.const(np_array, np_array.dtype), name)]
            else:
                self._params[name] = tvm.nd.array(np_array)
                self._nodes[name] = [
                    set_span(
                        _expr.var(
                            name, shape=self._params[name].shape, dtype=self._params[name].dtype
                        ),
                        name,
                    )
                ]
        else:
            if key not in ("dtype", "_output_shapes", "_class"):
                raise NotImplementedError(f"Other attributes for a Const(param) Node {key} ? .")

    def _get_attr(self, buf):
        """Returns the value of the attr of this buf with the given `name`.

        Args:
          buf: attrvalue protobuf.

        Returns:
          The value of the attr, as a Python object.

        Raises:
          ValueError: If this op does not have an attr with the given `name`.
        """
        fields = ["s", "i", "f", "b", "type", "shape", "tensor", "func"]

        x = buf

        ret = []

        try:
            from tensorflow.python.framework import dtypes
        except ImportError as e:
            raise ImportError(f"Unable to import tensorflow which is required {e}")

        # Treat an empty oneof value as an empty list.
        if not x.WhichOneof("value"):
            return ret
        if x.HasField("list"):
            for f in fields:
                if getattr(x.list, f):
                    if f == "type":
                        ret += [dtypes.as_dtype(x) for x in list(getattr(x.list, f))]
                    else:
                        ret += list(getattr(x.list, f))
        else:
            for f in fields:
                if x.HasField(f):
                    if f == "type":
                        ret = dtypes.as_dtype(getattr(x, f))
                    else:
                        ret = getattr(x, f)
        return ret

    def _parse_attr(self, attr_proto):
        """Convert a list of AttributeProto to a dict, with names as keys."""
        attrs = {}
        for key, value in attr_proto.items():
            attrs[key] = self._get_attr(value)

        return attrs

    def _convert_control_flow_operator(self, node, inputs, attrs, control_flow_node_map):
        """
        Convert the Relay control flow primitive into corresponding component
        of a Relay control flow construct, i.e. `tf.cond` and `tf.while_loop`
        are converted in Relay `If` and recusrive call, respectively.

        Parameters
        ----------
        node: TensorFlow graph node object.
            A TensorFlow graph node object.

        inputs : List[tvm.relay.Expr]
            List of input symbols.

        attrs : Dict[tvm.Attrs]
            Dict of operator attributes.

        control_flow_node_map : Dict[str, Set[str]]
            A dictionary contains the execution frame name to primitives
            mapping.

        Returns
        -------
        op : tvm.relay.Expr
            Converted relay expression.
        """
        node_name_prefix = node.name.rsplit("/", 1)[0]
        plname = find_parent_loop_name(node.name, self._while_loop_name_set)
        if node.op == "Merge":
            if _in_while_loop(self._control_flow_node_map, node_name_prefix):
                op = self._licm_construct(plname, node.input[0])
                if node_name_prefix not in self._loops:
                    self._loops[node_name_prefix] = Loop(self._mod, plname, self._lvar2expr)
            else:
                if node_name_prefix not in self._branches:
                    switch_prefix = node_name_prefix + "/Switch"
                    merge_idx = self._sorted_cf_node_names.index(node.name)
                    for i in range(merge_idx - 1, -1, -1):
                        cf_name = self._sorted_cf_node_names[i]
                        if cf_name.startswith(switch_prefix):
                            self._backtrack_construct(cf_name)
                            break

                branch = self._branches[node_name_prefix]
                false_br = self._licm_construct(plname, node.input[0])
                true_br = self._licm_construct(plname, node.input[1])
                branch.true_branch = true_br
                branch.false_branch = false_br
                op = branch.if_node()
                if node_name_prefix not in self._while_loop_name_set:
                    try:
                        cond_val = np.all(
                            _infer_value(branch.cond, self._params, self._mod).numpy()
                        )
                        if cond_val:
                            op = branch.true_branch
                        else:
                            op = branch.false_branch
                    except Exception:
                        op = branch.if_node()
        elif node.op == "Exit":
            loop = self._loops[node_name_prefix]

            # Check whether the order of loop variables aligns
            # with loop body. If not, create new loop variable list
            # with correct order.
            if not loop.aligned:
                loop_vars = []
                for i in self._loop_body_order[node_name_prefix]:
                    for j, k in enumerate(self._loop_var_order[node_name_prefix]):
                        if k == i:
                            loop_vars.append(loop.loop_vars[j])
                loop.loop_vars = loop_vars
                loop.aligned = True
            exit_name = node.name.split("/")[-1]
            if "_" in exit_name:
                exit_number = int(exit_name[5:])
            else:
                exit_number = 0
            expr = loop.while_loop()
            body_pos = exit_number
            for i, j in enumerate(self._loop_body_order[node_name_prefix]):
                if exit_number == j:
                    body_pos = i
                    break
            op = _expr.TupleGetItem(expr, body_pos)
        elif node.op == "Enter":
            op = self._licm_construct(plname, node.input[0])
        elif node.op == "LoopCond":
            op = self._licm_construct(plname, node.input[0])
            self._loops[node_name_prefix].cond = op
        elif node.op == "Switch":
            op = self._licm_construct(plname, node.input[0])
            cond = self._licm_construct(plname, node.input[1])
            if _in_while_loop(self._control_flow_node_map, node_name_prefix):
                if node_name_prefix not in self._loop_var_order:
                    self._loop_var_order[node_name_prefix] = []
                if node.name.endswith("Switch"):
                    self._loop_var_order[node_name_prefix].append(0)
                else:
                    self._loop_var_order[node_name_prefix].append(
                        int(node.name.split("Switch_")[-1])
                    )
                self._loops[node_name_prefix].loop_vars.append(op)
            else:
                if node_name_prefix not in self._branches:
                    self._branches[node_name_prefix] = Branch()
                self._branches[node_name_prefix].cond = cond
        elif node.op == "NextIteration":
            if node_name_prefix not in self._loop_body_order:
                self._loop_body_order[node_name_prefix] = []
            if node.name.endswith("NextIteration"):
                self._loop_body_order[node_name_prefix].append(0)
            else:
                self._loop_body_order[node_name_prefix].append(
                    int(node.name.split("NextIteration_")[-1])
                )
            op = self._licm_construct(plname, node.input[0])
            self._loops[node_name_prefix].body.append(op)
        else:
            raise Exception(f"Cannot identify control flow operator: {node.op}")

        return op

    def _partition_call_operator(self, inputs, attr):
        """
        Convert the Relay Partition call ops into Relay Function calls and
        function definitions from Tensorflow graph library attribute to Relay global
        functions

        Parameters
        ----------
        node: TensorFlow graph node object.
            A TensorFlow graph node object.

        inputs : List[tvm.relay.Expr]
            List of input symbols.

        attrs : Dict[tvm.Attrs]
            Dict of operator attributes.

        Returns
        -------
        op : tvm.relay.Expr
            Converted relay expression.
        """

        try:
            from tensorflow.python.framework import function_def_to_graph
        except ImportError as e:
            raise ImportError(f"Unable to import tensorflow which is required {e}")

        main_graph_proto = self._main_graph_proto
        outer_graph_def = main_graph_proto._graph

        node_func_name = attr.get("f").name
        func = next(
            (f for f in outer_graph_def.library.function if f.signature.name == node_func_name),
            None,
        )
        if func:
            devices = set(node.device for node in func.node_def)
            if len(devices) > 1:
                raise Exception(
                    "Found inconsistent Device assignment in the "
                    "Stateful Partitioned SubGraph. Rejecting "
                    "the subgraph "
                )
            # Convert function definition to graph
            func_input_shapes = func.attr["_input_shapes"].list.shape
            subgraph, _ = function_def_to_graph.function_def_to_graph_def(func, func_input_shapes)

            # Computing subgraph's input shape dictionary
            subgraph_shape_dict, input_expr_dict = {}, {}
            for f_arg, input in zip(func.signature.input_arg, inputs):
                input_expr_dict[f_arg.name] = input
                subgraph_shape_dict[f_arg.name] = _infer_shape(input, main_graph_proto._mod)

            func_name = f"func_{func.signature.name}"
            try:
                global_func = main_graph_proto._mod[func_name]
                sub_func = global_func
                sub_params = main_graph_proto._params
            except ValueError:
                # Construct relay nodes from the subgraph
                g1 = SubGraphProto(main_graph_proto)
                sub_func, sub_params = g1.from_tensorflow(subgraph, shape=subgraph_shape_dict)
                main_graph_proto._params.update(sub_params)
                func_expr = _function.Function(sub_func.params, sub_func.body)
                global_func = tvm.relay.GlobalVar(func_name)
                main_graph_proto._mod[global_func] = func_expr
                main_graph_proto._mod = InferType()(main_graph_proto._mod)

            param_exprs = []
            for param_expr in sub_func.params:
                # sub_params is subset of sub_func.params
                param_name = param_expr.vid.name_hint
                if param_name in input_expr_dict.keys():
                    param_exprs.append(input_expr_dict[param_name])
                elif param_name in sub_params.keys():
                    param_exprs.append(param_expr)
                else:
                    raise Exception(f"Input parameter {param_name} not found")

            sb = tvm.relay.scope_builder.ScopeBuilder()
            loop_ret = global_func(*param_exprs)
            sb.ret(loop_ret)
            ret = sb.get()
        else:
            raise Exception(f"Function not found - {node_func_name}")
        return ret

    def _convert_operator(
        self, op_name, node_name, inputs, attrs, identity_list=None, convert_map=None
    ):
        """Convert from Tensorflow operator to relay operator.
        The converter must specify conversions explicitly for incompatible name, and
        apply handlers to operator attributes.

        Parameters
        ----------
        op_name : str
            Operator name, such as Conv2D, AvgPool
        node_name : str
            Node name, predefined by user or default setting of TF
        inputs : list of relay.op
            List of input symbols.
        attrs : dict
            Dict of operator attributes
        identity_list : list
            List of operators that don't require conversion
        convert_map : dict
            Dict of name : callable, where name is the op's name that
            require conversion to relay, callable are functions which
            take attrs and return (new_op_name, new_attrs)

        Returns
        -------
        sym : relay.op
            Converted relay operator
        """
        identity_list = identity_list if identity_list else _identity_list
        convert_map = convert_map if convert_map else _convert_map
        if op_name in identity_list:
            sym = get_relay_op(op_name)(*inputs, **attrs)
        elif op_name in convert_map:
            if _need_prelude_for_shape_inference(op_name):
                sym = convert_map[op_name](inputs, attrs, self._params, self._prelude)
            else:
                sym = convert_map[op_name](inputs, attrs, self._params, self._mod)
        elif op_name in ["PartitionedCall", "StatefulPartitionedCall"]:
            sym = self._partition_call_operator(inputs, attrs)
        else:
            raise NotImplementedError(f"Operator {op_name} not implemented.")

        sym = set_span(sym, node_name)

        return sym

    def _licm_construct(self, loop_name, node_name):
        """Construct a node by considering whether it is
        loop invariant with the given while loop. If yes, we
        generate a loop Variable. Otherwise, return regular
        converted relay expression.

        Parameters
        ----------
        loop_name : str
            TensorFlow while loop name to be checked.

        node_name : str
            TensorFlow node name.

        Returns
        -------
        out : relay.Expr or relay.Var
            Converted relay expression or loop var.
        """
        actual_expr = self._backtrack_construct(node_name)
        tn = node_name.split(":")
        node_name = tn[0].split("^")[-1]
        cloop_name = find_parent_loop_name(node_name, self._while_loop_name_set)

        if loop_name in self._while_loop_name_set and not cloop_name.startswith(loop_name):
            if loop_name not in self._lvar2expr:
                self._lvar2expr[loop_name] = {}
            if loop_name not in self._lname_map:
                self._lname_map[loop_name] = {}

            if node_name not in self._lname_map[loop_name]:
                var_name = f"{node_name}_loop_var"
                var_type = _infer_type(actual_expr, self._mod).checked_type
                loop_var = set_span(tvm.relay.var(var_name, type_annotation=var_type), var_name)
                try:
                    extra_param = _infer_value(actual_expr, self._params, self._mod)
                    self._params[var_name] = extra_param
                except Exception:
                    pass
                self._lvar2expr[loop_name][loop_var] = actual_expr
                self._lname_map[loop_name][node_name] = loop_var
                ret = loop_var
            else:
                ret = self._lname_map[loop_name][node_name]
        else:
            ret = actual_expr

        return ret

    def _backtrack_construct(self, node_name):
        """Convert a specific tensorflow node to relay expression.

        If any of its ancestor node is not converted yet, backtrack as
        far as input node and covert all nodes on the path.

        This is required when parsing control flow nodes, since the parsing
        order may not follow the original graph def.

        Parameters
        ----------
        node_name : str
            TensorFlow node name.

        Returns
        -------
        op : relay.Expr
            Converted relay expression
        """
        try:
            from tensorflow.python.framework import tensor_util
        except ImportError as e:
            raise ImportError(f"Unable to import tensorflow which is required {e}")

        input_op_name = node_name.split(":")[0].split("^")[-1]
        if input_op_name not in self._nodes:
            node = self._tf_node_map[input_op_name]
            attr = self._parse_attr(node.attr)

            if node.op in _control_flow_nodes:
                attr = self._parse_attr(node.attr)
                op = self._convert_control_flow_operator(
                    node, [], attr, self._control_flow_node_map
                )
            else:
                attr["_output_shapes"] = self._output_shapes[input_op_name]
                attr["_node_name"] = node.name
                attr["_target_layout"] = self._layout

                inputs = [self._backtrack_construct(iname) for iname in node.input]

                plname = find_parent_loop_name(node_name, self._while_loop_name_set)

                # For TensorArrayV3 op, we need to infer shape first
                if is_tensor_array_constuctor(node):
                    raw_elem_shape = tensor_util.TensorShapeProtoToList(attr["element_shape"])
                    elem_shape = []
                    for dim in raw_elem_shape:
                        if dim < 0:
                            elem_shape.append(Any())
                        else:
                            elem_shape.append(dim)

                    if elem_shape:
                        attr["shape"] = elem_shape
                    if attr["identical_element_shapes"] or elem_shape:
                        shape_node, wnode_op, output_index = self._tensor_array_shape_nodes[
                            node.name
                        ]
                        name = shape_node.name
                        if output_index > 0:
                            name += ":" + str(output_index)
                        converted = self._backtrack_construct(name)
                        shape = _infer_shape(converted, self._mod)
                        if wnode_op.startswith("TensorArraySplit"):
                            shape = (Any(),) + shape[1:]
                        elif wnode_op.startswith("TensorArrayScatter"):
                            shape = shape[1:]

                        if node.name in self._tensor_array_shapes:
                            preset_shape = self._tensor_array_shapes[node.name]
                            shape = _get_more_static_shape(shape, preset_shape)

                        if "shape" in attr:
                            attr["shape"] = _get_more_static_shape(shape, attr["shape"])
                        else:
                            attr["shape"] = shape

                # LICM
                if plname in self._while_loop_name_set:
                    for i, iname in enumerate(node.input):
                        actual_input = self._licm_construct(plname, iname)
                        inputs[i] = actual_input

                op = self._convert_operator(node.op, node.name, inputs, attr)
            if isinstance(op, np.ndarray):
                self._params[node.name] = tvm.nd.array(op)
                op = [
                    set_span(
                        _expr.var(
                            node.name,
                            shape=self._params[node.name].shape,
                            dtype=self._params[node.name].dtype,
                        ),
                        node.name,
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


class SubGraphProto(GraphProto):
    """A helper class for handling relay subgraph copying from Tensorflow GraphDef."""

    def __init__(self, main_graph_proto):
        super().__init__()
        self._main_graph_proto = main_graph_proto  # holds main graph proto object

    def from_tensorflow(self, graph, layout="NHWC", shape=None, outputs=None):
        """Wrapper to _get_relay_func which converts Tensorflow graph to Relay function.
        Return Relay function and params
        """
        func = self._get_relay_func(graph, layout=layout, shape=shape, outputs=outputs)
        return func, self._params


def from_tensorflow(graph, layout="NHWC", shape=None, outputs=None, convert_config=None):
    """Load tensorflow graph which is a python tensorflow graph object into relay.
    The companion parameters will be handled automatically.

    Parameters
    ----------
    graph : GraphDef object
        Tensorflow GraphDef

    layout : target layout to be used (Optional)
        NCHW only supported now to enable NHWC models on GPU.

    shape : Dictionary of input dimensions (Optional)
        Graph level input shape dictionary.

    outputs : List of output tensor names (Optional)
        if not specified then the last node is assumed as graph output.

    convert_config : Optional[Dict[str, Any]]
        Default config:
            use_dense : bool = True
                Ture to convert `tf.matmul` to `nn.dense`, else to `nn.matmul`.
                The `nn.dense` op requires the data tensor to be non-transposed and weight tensor
                to be transposed, may insert extra `transpose` to the original graph.
            use_nt_batch_matmul : bool = True
                True to convert `tf.batch_matmul` to `nn.batch_matmul` strict to NT format
                (transpose_a=False, transpose_b=True).

    Returns
    -------
    mod : tvm.IRModule
        The module that optimizations will be performed on.

    params : dict of str to tvm.nd.NDArray
        Dict of converted parameters stored in tvm.nd.NDArray format
    """
    global TF_DEFAULT_CONFIGS
    if convert_config is not None:
        TF_DEFAULT_CONFIGS.update(convert_config)

    g = GraphProto()
    mod, params = g.from_tensorflow(graph, layout, shape, outputs)
    return mod, params
