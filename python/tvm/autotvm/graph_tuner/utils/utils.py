# pylint: disable=eval-used,invalid-name,too-many-arguments
"""Utility functions"""
import os
import json

from .._base import ELEMLIKE_NODE_NAMES


def is_elemlike_op(node):
    """Check whether a node is an element-wise like operator.

    Parameters
    ----------
    node : dict of str to object
        Node entry in nnvm Graph json format.
    """
    is_elemlike = False
    for item_name in ELEMLIKE_NODE_NAMES:
        if item_name in node["op"]:
            is_elemlike = True
            break
    return is_elemlike


def is_input_node(node_list, input_names, node_idx):
    """Whether a node is an input node.

    Parameters
    ----------
    node_list : list of dict of str to object
        List of all nodes in a graph.

    input_names : list of string
        List of input names.

    node_idx : int
        Node index to be checked.

    Returns
    -------
    out : bool
        whether node is a input node.
    """
    return node_list[node_idx]["name"] in input_names \
            or node_list[node_idx]["op"] == "null"


def shape2layout(shape, layout_template):
    """Given a shape and a layout template, return the actual layout.
    For example, given shape (1, 8, 32, 32, 4) and layout template NCWHc,
    the return would be NCHW4c.

    Parameters
    ----------
    shape : tuple of int
        Input shape.

    layout_template : str
        Input layout template.

    Returns
    -------
    out : str
        Output layout.
    """
    if len(shape) != len(layout_template):
        raise RuntimeError("Shape and layout_template format mismatch: "
                           "%s vs %s." % (str(shape), layout_template))
    layout = ""
    for i, c in enumerate(layout_template):
        if not c.isalpha():
            raise RuntimeError("layout_template can only "
                               "contains alphabet character.")
        if c.islower():
            layout += "%d%c" % (shape[i], c)
        else:
            layout += c
    return layout


def get_wkl_map(graph, workload_list, target_op,
                graph_wkl_list):
    """Get a dictionary maps node index of a graph to workload
    index in a workload list.

    Parameters
    ----------
    graph : nnvm Graph
        Input graph.

    workload_list : list of tuple
        Workload list containing all unique workloads in the input graph.

    target_op : str
        Target operator name.

    graph_wkl_list : list of tuple
        List contains all workloads of target_op in the input graph. The order
        of workloads should be the ascending order of node index. For conv2d_NCHWc,
        conversion from conv2d workload is required and get_conv2d_NCHWc_AVX_workload
        is provided as built-in function to deal with this. Make sure the workload
        format is consistent with the workload format in records.

    Returns
    -------
    out : dict of int to int
        Dictionary maps node index of a graph to workload index.
    """
    g_dict = json.loads(graph.json())
    node_list = g_dict["nodes"]
    workload_map = {}
    for i, wkl in enumerate(workload_list):
        workload_map[wkl] = i
    node_map = {}
    graph_wkl_idx = 0
    for idx, node in enumerate(node_list):
        if node["op"] != target_op:
            continue
        node_map[idx] = workload_map[graph_wkl_list[graph_wkl_idx]]
        graph_wkl_idx += 1
    return node_map


def get_real_node(in_node_dict, node_list, idx, target_op):
    """Get the index of first ancestor node with target_op as operator name.

    Parameters
    ----------
    in_node_dict : dict of int to list of int
        Dictionary maps node index to closest input ancestors.
        It can be created with get_in_nodes.

    node_list : list of dict of str to object
        List of all nodes in a graph.

    idx : int
        Input node index.

    target_op : str, optional
        Target operator name.

    Returns
    -------
    out : int
        Output node index.
    """
    if node_list[idx]["op"] == target_op or not in_node_dict[idx]:
        return idx
    anc_node_idx = in_node_dict[idx][0]
    anc_node = node_list[anc_node_idx]
    while anc_node["op"] != target_op:
        anc_node_idx = in_node_dict[anc_node_idx][0]
        anc_node = node_list[anc_node_idx]
    return anc_node_idx
