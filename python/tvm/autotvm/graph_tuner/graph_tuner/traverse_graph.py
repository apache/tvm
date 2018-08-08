"""API for graph traversing."""
import json

from .._base import _elemlike_node_names, _rule_out_node_names


def is_elemlike_op(node):
    """Check whether a node is an element-wise like operator.

    Parameters
    ----------
    node : dict of str to object
        Node entry in nnvm Graph json format.
    """
    is_elemlike = False
    for item_name in _elemlike_node_names:
        if item_name in node["op"]:
            is_elemlike = True
            break
    return is_elemlike


def get_direct_ancestor(node_list, visited_dict, op_name, node_idx, graph_input_nodes):
    """Given a node_list in nnvm graph and a node index, return the
    closest ancestor which has op_name as operator name or is element-wise like operator.

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

    graph_input_nodes : list of int
        List of graph input node index.

    Returns
    -------
    out : list of int
        List of ancestor node index.
    """
    if node_idx in visited_dict:
        return visited_dict[node_idx]
    if node_idx in graph_input_nodes:
        return [node_idx]
    node = node_list[node_idx]
    # Rule out injective operators
    is_rule_out = False
    for item_idx in node["inputs"]:
        item = node_list[item_idx[0]]
        for rule_out_item in _rule_out_node_names:
            if item["op"] == rule_out_item or rule_out_item in item["op"]:
                is_rule_out = True
                break
    if is_rule_out:
        return []

    node_direct_ancestor = []
    for item_idx in node["inputs"]:
        item = node_list[item_idx[0]]
        is_elemlike = is_elemlike_op(item)
        if item["op"] == op_name or is_elemlike or item_idx[0] in graph_input_nodes:
            node_direct_ancestor.append(item_idx[0])
        else:
            tmp = get_direct_ancestor(node_list, visited_dict, op_name,
                                      item_idx[0], graph_input_nodes)
            for tmp_item in tmp:
                node_direct_ancestor.append(tmp_item)
    if not is_elemlike_op(node_list[node_idx]) and node_direct_ancestor:
        node_direct_ancestor = [node_direct_ancestor[0]]
    visited_dict[node_idx] = node_direct_ancestor
    return node_direct_ancestor


def get_in_nodes(graph, op_name, input_names):
    """Create a dictionary mapping from op_name nodes or element-wise like
    nodes to closest input ancestors.

    Parameters
    ----------
    graph : nnvm Graph
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
    g_dict = json.loads(graph.json())
    node_list = g_dict["nodes"]
    visited_dict = {}
    in_node_dict = {}
    graph_input_nodes = set()
    for input_name in input_names:
        for i, node in enumerate(node_list):
            if node["name"] == input_name:
                graph_input_nodes.add(i)
    for i, node in enumerate(node_list):
        get_direct_ancestor(node_list, visited_dict, op_name, i, graph_input_nodes)
    for key, val in visited_dict.items():
        node = node_list[key]
        is_elemlike = is_elemlike_op(node)
        if node["op"] == op_name or (is_elemlike and val):
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
