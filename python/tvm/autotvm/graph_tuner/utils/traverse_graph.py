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
# pylint: disable=too-many-locals,too-many-statements,too-many-branches,protected-access
"""API for graph traversing."""
import threading
import re

import tvm
from tvm import relay, autotvm
from tvm.relay import transform
from tvm.relay.expr import Call, TupleGetItem, Var, Constant, Tuple
from tvm.relay.function import Function
from tvm.relay.ty import TupleType, TensorType
from tvm.autotvm.task import TaskExtractEnv

from .utils import has_multiple_inputs, is_boundary_node, is_skipped_node
from .._base import OPT_OUT_OP


def expr2graph(expr, target_ops, node_dict, node_list, tvm_target):
    """Convert relay expr to graph data structure
    and fetch workloads of target operators.

    Parameters
    ----------
    expr : tvm.relay.Expr.Function
        Input relay function expression.

    target_ops: List of tvm.ir.Op
        List of target relay ops

    node_dict : dictionary from tvm.relay.Expr to int
        Dictionary to record node index

    node_list : list of dictionary
        List of nodes which contains all expr in the input relay function.
        Each node will be stored as a dictionary in the format of
        {"op": str, "node": tvm.relay.expr, "inputs": [int], "types": [tvm.relay.Type],
         "name": str, "workloads": [tuple], "topi_op": [function]}

    tvm_target : tvm.target
        The TVM target object.
    """
    # TODO(@kevinthesun, @icemelon9): Currently graph tuning pass relies on the fact
    #   that # autotvm tasks == # ops. But this won't be true after having relay op
    #   strategy. We need to find a solution to fix this.
    env = TaskExtractEnv.get(allow_duplicate=True)
    env.reset(target_ops)
    # pylint: disable=not-context-manager
    with env:
        _expr2graph_impl(expr, target_ops, node_dict, node_list, tvm_target)
        task_pos = 0
        for node_entry in node_list:
            if node_entry["op"] in target_ops:
                task_name, args = env.task_collection[task_pos]
                task = autotvm.task.create(task_name, args, target=tvm_target)
                node_entry["workloads"] = [task.workload]
                node_entry["topi_op"] = [task_name]
                task_pos += 1


def _infer_type(node):
    """A method to infer the type of a relay expression."""
    mod = tvm.IRModule.from_expr(node)
    mod = transform.InferType()(mod)
    entry = mod["main"]
    return entry if isinstance(node, relay.Function) else entry.body


def _replace_device_with_tracing(target):
    """This is to replace -device=XXX with -device=tracing in the tvm_target string.
    It is a stand-along function for testability.
    We need to have device=tracing in order to fetch the workloads, it is not used
    for anything beyond that so it is safe to override the device here only."""
    target = str(target)
    if "-device" in target:
        return re.sub("-device=[^\\-$]+", "-device=tracing ", target).strip(" ")
    return target + " -device=tracing"


def _expr2graph_impl(expr, target_ops, node_dict, node_list, tvm_target):
    """Implementation to convert relay expr to graph data structure"""

    def _traverse_expr(node):
        if node in node_dict:
            return
        node_index = len(node_list)
        node_entry = {"node": node, "inputs": [], "types": [], "op": None, "name": None}

        if isinstance(node, Call):
            op = node.op
            node_entry["op"] = node.op
            for arg in node.args:
                in_node_idx = node_dict[arg]
                if isinstance(arg, (Tuple, TupleGetItem)):
                    node_entry["inputs"] += node_list[in_node_idx]["inputs"]
                else:
                    node_entry["inputs"].append([in_node_idx, 0, 0])
            infer_out = _infer_type(node)
            out_type = infer_out._checked_type_
            if isinstance(out_type, TensorType):
                node_entry["types"].append(out_type)
            elif isinstance(out_type, TupleType):
                for tupe_type in out_type.fields:
                    node_entry["types"].append(tupe_type)
            else:
                raise RuntimeError(
                    f"Unsupported output type {type(out_type)} in operator {op.name}"
                )

            # Utilize tracing target to fetch workload with topo-order.
            # Since we only need workload, dummy target can be used to
            # create task.
            if op in target_ops:
                params = []
                for i, input_idx in enumerate(node_entry["inputs"]):
                    input_node_entry = node_list[input_idx[0]]
                    input_type = input_node_entry["types"][input_idx[1]]
                    if not isinstance(input_node_entry["node"], (Var, Constant, Call)):
                        raise RuntimeError(
                            "Graph tuner can only tune target "
                            "operators with input node of type "
                            "relay.expr.Var/Constant/Call. Now "
                            "find a target op %s with input type %s"
                            % (op, str(type(input_node_entry["node"])))
                        )
                    free_var = relay.Var(f"var_{i}", input_type)
                    params.append(free_var)
                call = relay.Call(node.op, params, node.attrs)
                mod = tvm.IRModule.from_expr(relay.Function(params, call))
                relay.backend.te_compiler.get().clear()
                tracing_target = _replace_device_with_tracing(tvm_target)
                build_thread = threading.Thread(target=relay.build, args=(mod, tracing_target))
                build_thread.start()
                build_thread.join()
        elif isinstance(node, Var):
            node_entry["name"] = node.name_hint
            node_entry["types"] = [node.type_annotation]
        elif isinstance(node, Function):
            # Ignore root node since it equals to input function expression
            if node != expr:
                _expr2graph_impl(node, target_ops, node_dict, node_list, tvm_target)
            return
        elif isinstance(node, TupleGetItem):
            in_node_idx = node_dict[node.tuple_value]
            node_entry["inputs"].append([in_node_idx, node.index, 0])
        elif isinstance(node, Tuple):
            for tuple_item in node:
                in_node_idx = node_dict[tuple_item]
                if isinstance(tuple_item, TupleGetItem):
                    node_entry["inputs"] += node_list[in_node_idx]["inputs"]
                elif isinstance(tuple_item, Tuple):
                    raise RuntimeError("Graph tuner doesn't support nested tuple.")
                else:
                    node_entry["inputs"].append([in_node_idx, 0, 0])
        elif isinstance(node, Constant):
            node_entry["name"] = "Constant_" + str(node_index)
            node_entry["types"] = [node.checked_type]
        elif isinstance(node, tvm.ir.Op):
            return
        else:
            raise RuntimeError(f"Not supported relay node type in graph tuning: {type(node)}")
        node_dict[node] = node_index
        node_list.append(node_entry)

    relay.analysis.post_order_visit(expr, _traverse_expr)


def get_direct_ancestor(node_list, visited_dict, target_ops, node_idx, input_names):
    """Given a node_list in relay function and a node index, return the
    closest ancestor which has op_name as operator name or is multi_input operator.

    If node has multiple inputs, multiple ancestor nodes will be returned.

    Parameters
    ----------
    node_list : list of dict of str to object
        List of all nodes in a graph.

    visited_dict : dict of int to int
        Nodes and corresponding ancestors which have been visited.

    target_ops: List of str
        List of target relay base op name

    node_idx : int
        Input node index.

    input_names : list of str
        Names of graph input nodes.

    Returns
    -------
    out : list of int
        List of ancestor node index.
    """
    if node_idx in visited_dict:
        return visited_dict[node_idx]
    node = node_list[node_idx]
    if is_boundary_node(node, input_names):
        return [node_idx]

    node_direct_ancestor = []
    for item_idx in node["inputs"]:
        item = node_list[item_idx[0]]
        is_multiple_inputs = has_multiple_inputs(node_list, item_idx[0], input_names, OPT_OUT_OP)
        if item["op"] in target_ops or is_multiple_inputs:
            node_direct_ancestor.append(item_idx[0])
        else:
            tmp = get_direct_ancestor(node_list, visited_dict, target_ops, item_idx[0], input_names)
            for tmp_item in tmp:
                if tmp_item not in node_direct_ancestor:
                    node_direct_ancestor.append(tmp_item)
    visited_dict[node_idx] = node_direct_ancestor
    return node_direct_ancestor


def get_in_nodes(node_list, target_ops, input_names):
    """Create a dictionary mapping from op_name nodes or multi-input
    nodes to closest input ancestors.

    Parameters
    ----------
    node_list : list of dict of str to object
        List of all nodes in a graph.

    target_ops: List of str
        List of target relay op

    input_names : list of str
        Names of graph input nodes.

    Returns
    -------
    out : dict of int to list of int
        Dictionary maps node index to closest input ancestors.
    """

    visited_dict = {}
    in_node_dict = {}
    for i, node in enumerate(node_list):
        if is_boundary_node(node, input_names) or is_skipped_node(node):
            continue
        get_direct_ancestor(node_list, visited_dict, target_ops, i, input_names)
    for key, val in visited_dict.items():
        node = node_list[key]
        is_multiple_inputs = has_multiple_inputs(node_list, key, input_names, OPT_OUT_OP)
        if node["op"] in target_ops or is_multiple_inputs:
            in_node_dict[key] = val

    # Reduce boundary nodes
    out_node_dict = get_out_nodes(in_node_dict)
    has_reduced_node = True
    while has_reduced_node:
        boundary_nodes = []
        for key, val in in_node_dict.items():
            node = node_list[key]
            is_boundary = True
            # Target ops can't be boundary nodes
            if node["op"] not in target_ops:
                for input_idx in val:
                    in_node = node_list[input_idx]
                    if not is_boundary_node(in_node, input_names) and input_idx in in_node_dict:
                        is_boundary = False
                    else:
                        val.remove(input_idx)
                    if is_boundary:
                        boundary_nodes.append(key)
        if boundary_nodes:
            for idx in boundary_nodes:
                if idx in in_node_dict:
                    del in_node_dict[idx]
        else:
            has_reduced_node = False

    # Remove empty nodes to ignore pre-computed sub-graph
    has_empty_node = True
    while has_empty_node:
        empty_nodes = []
        for key, val in in_node_dict.items():
            if not val:
                empty_nodes.append(key)
        if empty_nodes:
            has_empty_node = True
            for node in empty_nodes:
                del in_node_dict[node]
                if node in out_node_dict:
                    for out_node in out_node_dict[node]:
                        in_node_dict[out_node].remove(node)
        else:
            has_empty_node = False

    return in_node_dict


def get_out_nodes(in_node_dict):
    """Create output dictionary from input dictionary.

    Parameters
    ----------
    in_node_dict : dict of int to list of int
        Dictionary maps node index to closest input ancestors.
        It can be created with get_in_nodes.

    Returns
    -------
    out : dict of int to list of int
        Dictionary maps node index to closest output nodes.
    """
    out_node_dict = {}
    for key in in_node_dict:
        out_node_dict[key] = []
    for key, val in in_node_dict.items():
        for item in val:
            if item in out_node_dict:
                out_node_dict[item].append(key)
            else:
                out_node_dict[item] = [key]

    return out_node_dict
