# pylint: disable=invalid-name,too-many-locals,keyword-arg-before-vararg
"""Get workloads for specific operator in a graph."""
import json
from functools import wraps

import nnvm

from nnvm.compiler import graph_attr
from topi.nn.conv2d import Workload


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
def get_conv2d_workload(in_dtype='float32', out_dtype='float32', **kwargs):
    """Get workload for conv2d.

    Parameters
    ----------
    graph : nnvm Graph
        Input graph.

    input_shapes : dict of str to tuple.
        Input shapes of graph

    unique_wkl : bool, optional
        Whether to ignore duplicated workloads.

    in_dtype : str, optional
        Input data type for convolution operator.

    out_dtype : str, optional
        Output data type for convolution operator.

    Returns
    -------
    out : list of namedtuple
        List of workloads for all convolution operator in graph
    """
    data_shape = kwargs["op_input_shapes"][0]
    attrs = kwargs["attrs"]
    if "groups" in attrs and int(attrs["groups"]) != 1:
        raise RuntimeError("Currenly depthwise convolution is not supported")
    if "layout" not in attrs or attrs["layout"] == "NCHW":
        height, width, in_filter = data_shape[2], data_shape[3], data_shape[1]
    else:
        height, width, in_filter = data_shape[1], data_shape[2], data_shape[3]
    out_filter = attrs["channels"]
    hkernel, wkernel = (attrs["kernel_size"])[1:-1].split(',')
    hpad, wpad = (attrs["padding"])[1:-1].split(',') if "padding" in attrs else (0, 0)
    hstride, wstride = (attrs["strides"])[1:-1].split(',') if "strides" in attrs else (1, 1)

    workload = Workload(*[in_dtype, out_dtype, height, width, in_filter,
                          int(out_filter), int(hkernel), int(wkernel),
                          int(hpad), int(wpad), int(hstride), int(wstride)])
    return workload
