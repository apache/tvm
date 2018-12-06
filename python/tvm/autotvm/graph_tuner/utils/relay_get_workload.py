# pylint: disable=invalid-name,too-many-locals,keyword-arg-before-vararg
"""Get workloads for specific operator in a graph."""
import json

from functools import wraps
from .utils import bind_inputs

import tvm

from tvm import autotvm, relay
from topi.nn.conv2d import conv2d_NCHWc
from topi.nn.depthwise_conv2d import depthwise_conv2d_NCHWc


def relay_get_workload(target_op_name):
    """Register function for getting operator workloads.

    Parameters
    ----------
    target_op_name : str
        Target operator name.

    Returns
    -------
    The registered function.
    """
    def _wrapper(func):
        """Wrapper function for decorator.
        """
        @wraps(func)
        def _traverse_graph(expr, input_shapes, unique_wkl=True, *args, **kwargs):
            """Internal util function to fetch graph attr and generate workloads.
            """
            if not isinstance(expr, tvm.relay.Expr):
                raise RuntimeError("First positonal argument must be a tvm.relay.Expr")
            expr = bind_inputs(expr, input_shapes)
            workload_set = set()
            workload_list = []

            def _node_to_workload(node):
                """Functions for post order dfs."""
                if not isinstance(node, relay.expr.Call):
                    return
                node_op_name = node.op.name.split(".")[-1]
                if node_op_name != target_op_name:
                    return
                kwargs["attrs"] = node.attrs
                kwargs["type_args"] = node.type_args
                workload = func(*args, **kwargs)
                if (not unique_wkl) or workload not in workload_set:
                    workload_set.add(workload)
                    workload_list.append(workload)

            relay.ir_pass.post_order_visit(expr, _node_to_workload)
            return workload_list
        return _traverse_graph
    return _wrapper


@relay_get_workload("conv2d")
def relay_get_conv2d_NCHWc_AVX_workload(**kwargs):
    """Get workload for conv2d of Intel CPU.

    Parameters
    ----------
    expr : tvm.relay.Expr.Function
        Input relay function expression.

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
    layout = attrs.data_layout if attrs.data_layout else "NCHW"
    if layout != "NCHW" and layout != "NHWC":
        raise RuntimeError("Only support NCHW or NHWC layouts for conv2d.")

    data_shape = kwargs["type_args"][0].shape
    kernel_shape = kwargs["type_args"][1].shape
    if layout == "NHWC":
        layout = "NCHW"
        data_shape = (data_shape[0], data_shape[3], data_shape[1], data_shape[2])
        kernel_shape = (kernel_shape[3], kernel_shape[2], kernel_shape[0], kernel_shape[1])
    data = tvm.placeholder(data_shape, name="data")
    kernel = tvm.placeholder(kernel_shape, name="kernel")

    in_channel = data_shape[1]
    out_channel = attrs.channels
    groups = attrs.groups if attrs.groups else 1
    is_depthwise = groups == in_channel and groups == out_channel

    if groups != 1 and not is_depthwise:
        raise RuntimeError("Currenly only depthwise or direct convolution are supported.")
    padding = tuple(attrs.padding) if attrs.padding else (0, 0)
    strides = tuple(attrs.strides) if attrs.strides else (1, 1)
    dilation = tuple(attrs.dilation) if attrs.dilation else (1, 1)
    out_dtype = attrs.out_dtype if attrs.out_dtype else "float32"
    op_func = depthwise_conv2d_NCHWc if is_depthwise else conv2d_NCHWc
    workload = autotvm.task.args_to_workload([data, kernel, strides, padding, dilation,
                                              layout, out_dtype], op_func)
    return workload
