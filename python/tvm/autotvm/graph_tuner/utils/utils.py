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
# pylint: disable=eval-used,invalid-name,too-many-arguments
"""Utility functions"""
import tvm
from tvm import relay
from tvm.relay import transform


def has_multiple_inputs(node_list, node_idx, input_names, opt_out_op):
    """Check whether a node has multiple input nodes
    except variable nodes.

    Parameters
    ----------
    node_list : list of dict of str to object
        List of all nodes in a graph.

    node_idx : int
        Node index to be checked.

    input_names : list of str
        List of input names of graph.

    Returns
    -------
    out : bool
        Whether the specified node has multiple input nodes
    """
    num_inputs = 0
    node = node_list[node_idx]
    for in_idx in node["inputs"]:
        in_idx = in_idx[0]
        in_node = node_list[in_idx]
        # Exclude parameter nodes
        if in_node["op"] is not None and in_node["op"].name in opt_out_op:
            increase = False
            for t_idx in in_node["inputs"]:
                increase = has_multiple_inputs(node_list, t_idx[0], input_names, opt_out_op)
            if increase:
                num_inputs += 1
        elif in_node["op"] is not None or ("name" in in_node and in_node["name"] in input_names):
            num_inputs += 1
    return num_inputs > 1


def is_boundary_node(node_entry, input_names):
    """Whether a node is a boundary node.
    Currently input node and nodes in LAYOUT_FIXED_OP are
    counted as boundary.

    Parameters
    ----------
    node_entry : dict
        Node entry.

    input_names : list of str
        List of input names of graph.

    Returns
    -------
    out : bool
        whether node is a boundary node.
    """
    # Operators dependent on original layouts.
    _LAYOUT_FIXED_OP = [
        relay.op.get(name)
        for name in (
            "nn.batch_flatten",
            "transpose",
            "reshape",
            "vision.multibox_prior",
            "vision.multibox_transform_loc",
            "where",
            "vision.non_max_suppression",
            "strided_slice",
        )
    ]

    out = node_entry["op"] in _LAYOUT_FIXED_OP or (
        "name" in node_entry and node_entry["name"] in input_names
    )
    return out


def is_skipped_node(node_entry):
    """Whether a node is not counted.

    Parameters
    ----------
    node_entry : dict
        Node entry.

    Returns
    -------
    out : bool
        whether node is skipped.
    """
    # Operators not counted in graph tuner.
    return isinstance(node_entry["node"], relay.Tuple)


def bind_inputs(expr, input_shapes=None, input_dtypes="float32"):
    """Bind input variables of a relay function expression
    to new shapes and/or dtypes.

    Parameters
    ----------
    expr : tvm.relay.Expr.Function
        Input relay function expression.

    input_shapes : dict of str to tuple of int, optional
        Input shapes.

    input_dtypes : str or dict of str to str, optional
        Input dtypes.

    Returns
    -------
    out : tvm.relay.Expr.Function
        Bind relay function expression.
    """
    if input_shapes is None:
        return expr
    if isinstance(input_dtypes, str):
        input_dtypes = {key: input_dtypes for key in input_shapes.keys()}

    updated_input_dict = {}
    for input_name in input_shapes.keys():
        updated_input = relay.var(
            input_name, shape=input_shapes[input_name], dtype=input_dtypes[input_name]
        )
        updated_input_dict[input_name] = updated_input

    rebind_dict = {}
    for var in expr.params:
        if var.name_hint in updated_input_dict:
            rebind_dict[var] = updated_input_dict[var.name_hint]
    updated_expr = relay.expr.bind(expr, rebind_dict)

    mod = tvm.IRModule.from_expr(updated_expr)
    mod = transform.InferType()(mod)
    entry = mod["main"]
    return entry if isinstance(updated_expr, relay.Function) else entry.body
