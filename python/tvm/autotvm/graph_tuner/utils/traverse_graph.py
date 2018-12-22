"""API for graph traversing."""
import json

import nnvm

from tvm import relay
from tvm.relay.expr import Call, Function, TupleGetItem, Var

from .._base import RULE_OUT_NODE_NAMES
from .utils import has_multiple_inputs, is_input_node


def expr2graph(expr, node_dict, node_list):
    """Convert relay expr to graph data structure.

    Parameters
    ----------
    expr : tvm.relay.Expr.Function
        Input relay function expression.

    node_dict : dictionary from tvm.relay.Expr to int
        Dictionary to record node index

    node_list : nodes stored in graph data structure
    """
    def _traverse_expr(node):
        if node in node_dict:
            return
        node_dict[node] = len(node_list)
        node_entry = {"node": node, "inputs": [],
                      "op": "null", "name": None}

        if isinstance(node, Call):
            op_name = node.op.name.split(".")[-1]
            node_entry["op"] = op_name
            for in_node in node.args:
                if isinstance(in_node, TupleGetItem):
                    in_node_idx = node_dict[in_node.tuple_value]
                else:
                    in_node_idx = node_dict[in_node]
                node_entry["inputs"].append([in_node_idx, 0, 0])
        elif isinstance(node, Var):
            node_entry["name"] = node.name_hint
        elif isinstance(node, Function):
            # Ignore root node since it equals to input function expression
            if node != expr:
                expr2graph(node, node_dict, node_list)
            return
        elif isinstance(node, (relay.op.op.Op, TupleGetItem)):
            return
        else:
            raise RuntimeError("Not supported relay node type in graph tuning: %s"
                               % str(type(node)))
        node_list.append(node_entry)

    relay.ir_pass.post_order_visit(expr, _traverse_expr)


def get_direct_ancestor(node_list, visited_dict, op_name, node_idx, input_names):
    """Given a node_list in nnvm graph and a node index, return the
    closest ancestor which has op_name as operator name or is multi_input operator.

    If node has multiple inputs, multiple ancestor nodes will be returned.

    Parameters
    ----------
    node_list : list of dict of str to object
        List of all nodes in a graph.

    visited_dict : dict of int to int
        Nodes and corresponding ancestors which have been visited.

    op_name : str
        Operator name.

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
    if is_input_node(node_list, node_idx):
        return [node_idx]
    node = node_list[node_idx]
    # Rule out injective operators
    is_rule_out = False
    for item_idx in node["inputs"]:
        item = node_list[item_idx[0]]
        for rule_out_item in RULE_OUT_NODE_NAMES:
            if rule_out_item in item["op"]:
                is_rule_out = True
                break
    if is_rule_out:
        return []

    node_direct_ancestor = []
    for item_idx in node["inputs"]:
        item = node_list[item_idx[0]]
        is_multiple_inputs = has_multiple_inputs(node_list, item_idx[0], input_names)
        if item["op"] == op_name or is_multiple_inputs:
            node_direct_ancestor.append(item_idx[0])
        else:
            tmp = get_direct_ancestor(node_list, visited_dict, op_name,
                                      item_idx[0], input_names)
            for tmp_item in tmp:
                node_direct_ancestor.append(tmp_item)
    if not has_multiple_inputs(node_list, node_idx, input_names) and node_direct_ancestor:
        node_direct_ancestor = [node_direct_ancestor[0]]
    visited_dict[node_idx] = node_direct_ancestor
    return node_direct_ancestor


def get_in_nodes(graph, op_name, input_names):
    """Create a dictionary mapping from op_name nodes or multi_input
    nodes to closest input ancestors.

    Parameters
    ----------
    graph : nnvm Graph or tvm.relay.Expr.Function
        Input graph.

    op_name : str
        Operator name.

    input_names : list of str
        Names of graph input nodes.

    Returns
    -------
    out : dict of int to list of int
        Dictionary maps node index to closest input ancestors.
    """
    if isinstance(graph, nnvm.graph.Graph):
        g_dict = json.loads(graph.json())
        node_list = g_dict["nodes"]
    elif isinstance(graph, Function):
        node_dict = {}
        node_list = []
        expr2graph(graph, node_dict, node_list)
    else:
        raise RuntimeError("Input graph must be either NNVM Graph or Relay Function.")

    visited_dict = {}
    in_node_dict = {}
    for i, node in enumerate(node_list):
        get_direct_ancestor(node_list, visited_dict, op_name, i, input_names)
    for key, val in visited_dict.items():
        node = node_list[key]
        is_multiple_inputs = has_multiple_inputs(node_list, key, input_names)
        if node["op"] == op_name or (is_multiple_inputs and val):
            in_node_dict[key] = val

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
    for key in in_node_dict.keys():
        out_node_dict[key] = []
    for key, val in in_node_dict.items():
        for item in val:
            if item in out_node_dict:
                out_node_dict[item].append(key)
            else:
                out_node_dict[item] = [key]

    return out_node_dict
