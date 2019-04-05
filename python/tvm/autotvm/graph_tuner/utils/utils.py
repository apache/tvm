# pylint: disable=eval-used,invalid-name,too-many-arguments
"""Utility functions"""
from tvm import relay


def has_multiple_inputs(node_list, node_idx, input_names):
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
        if in_node["op"] != "null" or is_input_node(in_node,
                                                    input_names):
            num_inputs += 1
    return num_inputs > 1


def is_input_node(node_entry, input_names):
    """Whether a node is an input node.

    Parameters
    ----------
    node_entry : dict
        Node entry.

    input_names : list of str
        List of input names of graph.

    Returns
    -------
    out : bool
        whether node is a input node.
    """
    return "name" in node_entry and node_entry["name"] in input_names


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


def get_wkl_map(node_list, workload_list, target_op,
                graph_wkl_list):
    """Get a dictionary maps node index of a graph to workload
    index in a workload list.

    Parameters
    ----------
    node_list : list of dict
        List of node entries.

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
        input_dtypes = {key : input_dtypes for key in input_shapes.keys()}

    updated_input_dict = {}
    for input_name in input_shapes.keys():
        updated_input = relay.var(input_name, shape=input_shapes[input_name],
                                  dtype=input_dtypes[input_name])
        updated_input_dict[input_name] = updated_input

    rebind_dict = {}
    for var in expr.params:
        if var.name_hint in updated_input_dict:
            rebind_dict[var] = updated_input_dict[var.name_hint]
    updated_expr = relay.expr.bind(expr, rebind_dict)

    return relay.ir_pass.infer_type(updated_expr)
