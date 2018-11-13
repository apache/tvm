# pylint: disable=invalid-name,too-many-locals,keyword-arg-before-vararg
"""Get workloads for specific operator in a graph."""
import json
from functools import wraps

import nnvm
import tvm

from nnvm.compiler import graph_attr
from tvm import autotvm
from topi.nn.conv2d import conv2d_NCHWc


def _parse_int(input):
    """Parse input."""
    if input.endswith('L'):
        input = input[:-1]
    return int(input)


def get_workload(op_name):
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


@get_workload("conv2d")
def get_conv2d_NCHWc_AVX_workload(**kwargs):
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

    if "groups" in attrs and _parse_int(attrs["groups"]) != 1:
        raise RuntimeError("Currenly depthwise convolution is not supported")
    padding = [_parse_int(i) for i in (attrs["padding"])[1:-1].split(',')] \
        if "padding" in attrs else (0, 0)
    strides = [_parse_int(i) for i in (attrs["strides"])[1:-1].split(',')] \
        if "strides" in attrs else (1, 1)
    out_dtype = str(attrs["out_dtype "]) if "out_dtype" in attrs else "float32"
    workload = autotvm.task.args_to_workload([data, kernel, strides, padding, layout,
                                              layout, out_dtype], conv2d_NCHWc)
    return workload
