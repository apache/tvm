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
# pylint: disable=invalid-name, unused-argument
"""BNNS library supported operators.
Is a part of Accelerate framework on macOS/iOS platforms. Apple provide several APIs
to handle tensor processing. Particularly:
 * BNNS (basic neural )
 * vDSP (1D and 2D tensor processing)
 * BLAS (gemm provide)

# There are two ways to registering a function for an op to indicate if it is
# supported by DNNL.

# - The first and simplest way is to use the helper so that
# users only need to provide the operator name and a boolean value to indicate if
# it is supported. For example:
#
#     .. code-block:: python
#
#       add = _register_external_op_helper("add")
#       add = _register_external_op_helper("add", True)
#       add = _register_external_op_helper("add", False)
#
# - The other way is to implement the function by themselves to
# check the attributes of the op and decide if it should be offloaded to DNNL.
"""
import math
import tvm.ir
from ...dataflow_pattern import wildcard, is_op, is_expr, is_constant
from .register import register_pattern_table, get_pattern_table

from tvm.relay import transform
from tvm.relay.expr import const
from tvm.relay.build_module import bind_params_by_name

def partition_for_bnns(mod, params=None):
    """Partition the graph greedily offloading supported
    operators to BNNS.

    Parameters
    ----------
    mod : Module
        The module to run passes on.
    params : Optional[Dict[str, NDArray]]
        Constant input parameters.

    Returns
    -------
    ret : annotated and partitioned module.
    """
    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)

    seq = tvm.transform.Sequential(
        [
            transform.InferType(),
            transform.FoldConstant(),
            transform.FoldScaleAxis(),
            transform.DynamicToStatic(),
            transform.AlterOpLayout(),
            # TODO(apeskov): WA. AlterOpLayout call lead to constants shape transformation
            #   Some expand_dims op may appears after constants. It breaks BNNS fusing.
            #   So we have to call FoldConstant right before bnns composite passes.
            transform.FoldConstant(),
            transform.MergeComposite(get_pattern_table("bnns")),
            transform.AnnotateTarget("bnns"),
            #   If you no need in per layer performance statistic you can
            #   uncomment next line
            # transform.MergeCompilerRegions(),
            transform.PartitionGraph(),
        ]
    )

    return seq(mod)


def _register_external_op_helper(op_name, supported=True):
    """The helper function to indicate that a given operator can be supported
    by BNNS.

    Paramters
    ---------
    op_name : Str
        The name of operator that will be registered.

    Returns
    -------
    f : callable
        A function that returns if the operator is supported by BNNS.
    """

    @tvm.ir.register_op_attr(op_name, "target.bnns")
    def _func_wrapper(expr):
        return supported

    return _func_wrapper

_register_external_op_helper("nn.batch_matmul")


# TODO [apeskov]:
#   1. enlarge list of supported types on
#   2. clarify meaning of "" value
def dtype_is_supported(dtype):
    return dtype == "float32" or dtype == ""


@tvm.ir.register_op_attr("nn.conv2d", "target.bnns")
def conv2d_check(expr):
    """Check if the conv2d can be executed in BNNS"""
    attrs, args = expr.attrs, expr.args
    data_typ = args[0].checked_type
    if len(data_typ.shape) != 4 or data_typ.dtype != "float32":
        return False
    if not isinstance(args[1], tvm.relay.expr.Constant):
        return False
    kernel_typ = args[1].checked_type
    if len(kernel_typ.shape) != 4 or kernel_typ.dtype != "float32":
        return False
    if attrs.data_layout != "NCHW":
        return False
    if not dtype_is_supported(attrs.out_dtype):
        return False
    return True


def bias_check(expr):
    """Check is bias added through the correct dimension"""
    attrs, args = expr.attrs, expr.args
    if not isinstance(args[1], tvm.relay.expr.Constant):
        return False
    if expr.op.name == "nn.bias_add":
        return attrs.axis == 1
    elif expr.op.name == "add":
        b_shape = args[1].checked_type.shape
        if len(b_shape) == 4:
            return bool(b_shape[0] == 1 and b_shape[2] == 1 and b_shape[3] == 1)
        elif len(b_shape) == 3:
            return bool(b_shape[1] == 1 and b_shape[2] == 1)

    return False


@tvm.ir.register_op_attr("nn.dense", "target.bnns")
def dense(expr):
    """Check if the dense can be used in BNNS."""
    attrs, args = expr.attrs, expr.args
    data_typ = args[0].checked_type
    if data_typ.dtype != "float32":
        return False
    kernel_typ = args[1].checked_type
    if len(kernel_typ.shape) != 2 or kernel_typ.dtype != "float32":
        return False
    if attrs.out_dtype != "float32" and attrs.out_dtype != "":
        return False
    return True


def make_conv_relu_pattern(with_bias=True, with_relu=True):
    data = wildcard()
    weight = wildcard()
    bias = wildcard()
    pat = is_op("nn.conv2d")(data, weight)
    if with_bias:
        pat = is_op("add")(pat, bias) | is_op("nn.bias_add")(pat, bias)
    if with_relu:
        pat = is_op("nn.relu")(pat)
    return pat


def check_conv(extract):
    """Check conv pattern is supported by BNNS."""
    is_ok = True

    def visit(op):
        nonlocal is_ok
        if isinstance(op, tvm.relay.Call):
            if op.op.name == "nn.conv2d":
                is_ok &= conv2d_check(op)
            elif op.op.name in ("nn.bias_add", "add"):
                is_ok &= bias_check(op)

    tvm.relay.analysis.post_order_visit(extract, visit)
    return is_ok


def make_dense_bias_pattern():
    data = wildcard()
    weight = wildcard()
    bias = wildcard()
    d = is_op("nn.dense")(data, weight)
    return is_op("add")(d, bias)


def make_dense_bias_gelu_pattern():
    dense_bias = make_dense_bias_pattern()
    const1 = is_expr(const(0.044715))
    const2 = is_expr(const(math.sqrt(2 / math.pi)))

    gelu = is_op("power")(dense_bias, is_expr(const(3, dtype="float32")))
    gelu = is_op("multiply")(gelu, const1)
    gelu = is_op("add")(gelu, dense_bias)
    gelu = is_op("multiply")(gelu, const2)
    gelu = is_op("tanh")(gelu)
    gelu = is_op("add")(gelu, is_expr(const(1, dtype="float32")))
    gelu = is_op("multiply")(gelu, is_expr(const(0.5)))
    gelu = is_op("multiply")(gelu, dense_bias)
    return gelu


def check_dense(extract):
    """Check conv pattern is supported by ACL."""
    call = extract
    while call.op.name != "nn.dense":
        call = call.args[0]
    return dense(call)


@register_pattern_table("bnns")
def pattern_table():
    conv2d_bias_pat = ("bnns.conv2d_bias", make_conv_relu_pattern(with_bias=True, with_relu=False), check_conv)
    conv2d_bias_relu_pat = ("bnns.conv2d_bias_relu", make_conv_relu_pattern(with_bias=True, with_relu=True), check_conv)
    conv2d_relu_pat = ("bnns.conv2d_relu", make_conv_relu_pattern(with_bias=False, with_relu=True), check_conv)
    dense_bias_gelu = ("bnns.dense_bias_gelu", make_dense_bias_gelu_pattern(), check_dense)
    dense_bias = ("bnns.dense_bias", make_dense_bias_pattern(), check_dense)
    bnns_patterns = [
        conv2d_bias_relu_pat,
        conv2d_relu_pat,
        conv2d_bias_pat,
        dense_bias_gelu,
        dense_bias,
    ]
    return bnns_patterns
