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
"""DNNL library supported operators.
There are two ways to registering a function for an op to indicate if it is
supported by DNNL.

- The first and simplest way is to use the helper so that
users only need to provide the operator name and a boolean value to indicate if
it is supported. For example:

    .. code-block:: python

      add = _register_external_op_helper("add")
      add = _register_external_op_helper("add", True)
      add = _register_external_op_helper("add", False)

- The other way is to implement the function by themselves to
check the attributes of the op and decide if it should be offloaded to DNNL.
"""
import logging

import tvm.ir
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name

from ...dataflow_pattern import wildcard, is_op, is_constant
from .register import register_pattern_table


def get_dnnl_version():
    """Return tuple with version or DNNL library if known
    Otherwise return unknown value which is bigger than any over real
    versions.
    """
    f = tvm.get_global_func("runtime.module.dnnl_version")
    return tuple(int(el) for el in f().split(".")) if f else (100500,)


dnnl_version = get_dnnl_version()

logger = logging.getLogger("DNNL")


def _register_external_op_helper(op_name, supported=True):
    """The helper function to indicate that a given operator can be supported
    by DNNL.

    Parameters
    ----------
    op_name : Str
        The name of operator that will be registered.

    Returns
    -------
    f : callable
        A function that returns if the operator is supported by DNNL.
    """

    @tvm.ir.register_op_attr(op_name, "target.dnnl")
    def _func_wrapper(expr):
        return supported

    return _func_wrapper


_register_external_op_helper("nn.batch_norm")
_register_external_op_helper("nn.conv2d")
_register_external_op_helper("nn.dense")
_register_external_op_helper("nn.relu")
_register_external_op_helper("tanh")
_register_external_op_helper("sigmoid")
_register_external_op_helper("add")
_register_external_op_helper("multiply")


def make_conv_pattern(with_bias=True, with_eltwise=None):
    """Create patterns related to nn.conv2d.

    Parameters
    ----------
    with_bias : bool
        Whether attach `bias_add` to `nn.conv2d`.
    with_eltwise : str
        The attached elementwise post-op name.
    Returns
    -------
    conv_out : CallPattern
        Call node sequence.
    """
    data = wildcard()
    weight = wildcard()
    bias = wildcard()
    conv = is_op("nn.conv2d")(data, weight)
    if with_bias:
        conv_out = is_op("add")(conv, bias)
    else:
        conv_out = conv
    if with_eltwise:
        return is_op(with_eltwise)(conv_out)
    return conv_out


def make_dense_pattern(with_bias=True, with_eltwise=None):
    """Create patterns related to nn.dense.

    Parameters
    ----------
    with_bias : bool
        Whether attach `bias_add` to `nn.dense`.
    with_eltwise : str
        The attached elementwise post-op name.
    Returns
    -------
    dense_out : CallPattern
        Call node sequence.
    """
    data = wildcard()
    weight = wildcard()
    bias = wildcard()
    dense = is_op("nn.dense")(data, weight)
    if with_bias:
        dense_out = is_op("add")(dense, bias)
    else:
        dense_out = dense
    if with_eltwise:
        dense_out = is_op(with_eltwise)(dense_out)
    return dense_out


def make_dnnl_pattern(op, with_bias, with_eltwise):
    """Create dnnl patterns.

    Parameters
    ----------
    op : str
        The first call node's op name.
    with_bias : bool
        Whether attach `bias_add` to `nn.dense`.
    with_eltwise : str
        The attached elementwise post-op name.
    Returns
    -------
    pattern : Tuple(pattern_name, CallPattern)
        Created pattern name, along with its CallPattern.
    """
    pat_name = "dnnl." + op
    pat_name += "_bias" if with_bias else ""
    pat_name += ("_" + with_eltwise.split(".")[-1]) if with_eltwise else ""
    if op == "conv2d":
        dnnl_pattern = (pat_name, make_conv_pattern(with_bias, with_eltwise))
    elif op == "dense":
        dnnl_pattern = (pat_name, make_dense_pattern(with_bias, with_eltwise))
    else:
        logger.warning("Currently, only conv2d and dense op are supported, but got %s.", op)
        dnnl_pattern = ()
    return dnnl_pattern


def make_qnn_conv2d_pattern(with_sum=False):
    """Make qnn.conv2d based pattern supported by DNNL

    Parameters
    ----------
    with_sum : bool
        Indicate to append qnn.sum at the end of pattern

    Returns
    -------
    pattern : Tuple(pattern_name, CallPattern)
        Created pattern name, along with its CallPattern.
    """
    weight = is_constant()  # |const requirements, have to recalculate bias to compensate src_zp
    bias = is_constant()

    pat = wildcard()
    pat = is_op("qnn.conv2d")(
        pat, weight, is_constant(), is_constant(), is_constant(), is_constant()
    )
    pat = is_op("add")(pat, bias) | pat
    pat = is_op("qnn.requantize")(pat, is_constant(), is_constant(), is_constant(), is_constant())
    pat = is_op("clip")(pat)
    pat = is_op("cast")(pat)
    if with_sum is True:
        pat = is_op("qnn.add")(
            pat,
            wildcard(),
            is_constant(),
            is_constant(),
            is_constant(),
            is_constant(),
            is_constant(),
            is_constant(),
        )
        pat = is_op("clip")(pat)

    pat_name = "dnnl.qnn.conv2d_sum" if with_sum else "dnnl.qnn.conv2d"

    return pat_name, pat


def make_qnn_dense_pattern(with_sum=False):
    """Make qnn.dense based pattern supported by DNNL

    Parameters
    ----------
    with_sum : bool
        Indicate to append qnn.sum at the end of pattern

    Returns
    -------
    pattern : Tuple(pattern_name, CallPattern)
        Created pattern name, along with its CallPattern.
    """
    weight = is_constant()
    bias = is_constant()

    pat = wildcard()
    pat = is_op("qnn.dense")(
        pat, weight, is_constant(), is_constant(), is_constant(), is_constant()
    )
    pat = is_op("add")(pat, bias) | pat
    pat = is_op("qnn.requantize")(pat, is_constant(), is_constant(), is_constant(), is_constant())
    pat = is_op("clip")(pat)
    pat = is_op("cast")(pat)
    if with_sum is True:
        pat = is_op("qnn.add")(
            pat,
            wildcard(),
            is_constant(),
            is_constant(),
            is_constant(),
            is_constant(),
            is_constant(),
            is_constant(),
        )
        pat = is_op("clip")(pat)

    pat_name = "dnnl.qnn.dense_sum" if with_sum else "dnnl.qnn.dense"

    return pat_name, pat


@register_pattern_table("dnnl")
def pattern_table():
    """Create dnnl patterns.

    Returns
    -------
    dnnl_patterns : List[dnnl_pattern]
        Created patterns.
    """
    elt_list = ["nn.relu", "tanh", "sigmoid", None]
    dnnl_patterns = []
    for with_bias in [True, False]:
        for elt in elt_list:
            if not with_bias and not elt:
                continue
            dnnl_patterns.append(make_dnnl_pattern("conv2d", with_bias, elt))
            dnnl_patterns.append(make_dnnl_pattern("dense", with_bias, elt))

    for with_sum in [True, False]:
        dnnl_patterns.append(make_qnn_conv2d_pattern(with_sum))
        # Old dnnl version doesn't support per channel o_scale
        if dnnl_version >= (2, 2) or not with_sum:
            dnnl_patterns.append(make_qnn_dense_pattern(with_sum))

    return dnnl_patterns


def partition_for_dnnl(mod, params=None):
    """Partition the graph greedily offloading supported operators to DNNL.

    Parameters
    ----------
    mod : Module
        The module to run passes on.
    params : Optional[Dict[str, NDArray]]
        Constant input parameters.
    Returns
    -------
    mod : Module
        Annotated and partitioned module.
    """

    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)
    seq = tvm.transform.Sequential(
        [
            transform.CanonicalizeOps(),
            transform.InferType(),
            transform.SimplifyInference(),
            transform.FoldConstant(),
            transform.FoldScaleAxis(),
            # fold consecutive add ops to simplify pattern `conv2d-bias_add-bn-relu`
            transform.SimplifyExpr(),
            transform.FoldConstant(),
            transform.MergeComposite(pattern_table()),
            transform.AnnotateTarget("dnnl"),
            transform.MergeCompilerRegions(),
            transform.PartitionGraph(),
        ]
    )
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    return mod
