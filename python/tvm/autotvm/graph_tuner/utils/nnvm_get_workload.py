# pylint: disable=invalid-name,too-many-locals,keyword-arg-before-vararg
"""Get workloads for specific operator in a graph."""
import json
from functools import wraps

import nnvm

from nnvm.compiler import graph_attr

import tvm

from tvm import autotvm
from topi.nn.conv2d import conv2d_NCHWc
from topi.nn.depthwise_conv2d import depthwise_conv2d_NCHWc


def _parse_int(input_str):
    """Parse input string."""
    if input_str.endswith('L'):
        input_str = input_str[:-1]
    return int(input_str)


def nnvm_get_workload(op_name):
    """Register function for getting operator workloads.

    Parameters
    ----------
    op_name : str
        operator name.

    Returns
    -------
    The registered function.
    """
    def _wrapper(func):
        """Wrapper function for decorator.
        """
        @wraps(func)
        def _traverse_graph(graph, input_shapes, unique_wkl=True, *args, **kwargs):
            """Internal util function to fetch graph attr and generate workloads.
            """
            if not isinstance(graph, nnvm.graph.Graph):
                raise RuntimeError("First positonal argument must be a nnvm.graph.Graph.")
            if not isinstance(input_shapes, dict):
                raise RuntimeError("Second positonal argument must be a dict of str to tuple.")
            g = graph_attr.set_shape_inputs(graph, input_shapes)
            g = g.apply("InferShape")
            g_dict = json.loads(g.json())
            node_list = g_dict["nodes"]
            shape_list = g_dict['attrs']['shape'][1]
            node_map = g_dict["node_row_ptr"]
            workload_set = set()
            workload_list = []

            for node in node_list:
                if node['op'] != op_name:
                    continue
                attrs = node["attrs"]
                op_input_shapes = []
                for input_idx in node["inputs"]:
                    op_input_shapes.append(shape_list[node_map[input_idx[0]]])
                kwargs["op_input_shapes"] = op_input_shapes
                kwargs["attrs"] = attrs
                workload = func(*args, **kwargs)
                if (not unique_wkl) or workload not in workload_set:
                    workload_set.add(workload)
                    workload_list.append(workload)
            return workload_list
        return _traverse_graph
    return _wrapper


@nnvm_get_workload("conv2d")
def nnvm_get_conv2d_NCHWc_AVX_workload(**kwargs):
    """Get workload for conv2d of Intel CPU.

    Parameters
    ----------
    graph : nnvm Graph
        Input graph.

    input_shapes : dict of str to tuple.
        Input shapes of graph

    unique_wkl : bool, optional
        Whether to ignore duplicated workloads.

    Returns
    -------
    out : list of namedtuple
        List of workloads for all convolution operator in graph
    """
    attrs = kwargs["attrs"]
    layout = str(attrs["layout"]) if "layout" in attrs else "NCHW"
    if layout != "NCHW" and layout != "NHWC":
        raise RuntimeError("Only support NCHW or NHWC layouts for conv2d.")

    data_shape = kwargs["op_input_shapes"][0]
    kernel_shape = kwargs["op_input_shapes"][1]
    if layout == "NHWC":
        layout = "NCHW"
        data_shape = (data_shape[0], data_shape[3], data_shape[1], data_shape[2])
        kernel_shape = (kernel_shape[3], kernel_shape[2], kernel_shape[0], kernel_shape[1])
    data = tvm.placeholder(data_shape, name="data")
    kernel = tvm.placeholder(kernel_shape, name="kernel")
    in_channel = data_shape[1]
    out_channel = _parse_int(attrs["channels"])
    groups = _parse_int(attrs["groups"]) if "groups" in attrs else 1
    is_depthwise = groups == in_channel and groups == out_channel

    if groups != 1 and not is_depthwise:
        raise RuntimeError("Currenly only depthwise or direct convolution are supported.")
    padding = [_parse_int(i) for i in (attrs["padding"])[1:-1].split(',')] \
        if "padding" in attrs else (0, 0)
    strides = [_parse_int(i) for i in (attrs["strides"])[1:-1].split(',')] \
        if "strides" in attrs else (1, 1)
    dilation = [_parse_int(i) for i in (attrs["dilation"])[1:-1].split(',')] \
        if "dilation" in attrs else (1, 1)
    out_dtype = str(attrs["out_dtype "]) if "out_dtype" in attrs else "float32"
    op_func = depthwise_conv2d_NCHWc if is_depthwise else conv2d_NCHWc
    workload = autotvm.task.args_to_workload([data, kernel, strides, padding, dilation,
                                              layout, out_dtype], op_func)
    return workload
