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
from tvm import relay
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name

from ... import _ffi_api
from ...dataflow_pattern import wildcard, is_op
from .register import register_pattern_table

logger = logging.getLogger("DNNL")


def _register_external_op_helper(op_name, supported=True):
    """The helper function to indicate that a given operator can be supported
    by DNNL.

    Paramters
    ---------
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
_register_external_op_helper("nn.conv1d")
_register_external_op_helper("nn.conv2d")
_register_external_op_helper("nn.conv3d")
_register_external_op_helper("nn.conv2d_transpose")
_register_external_op_helper("nn.conv3d_transpose")
_register_external_op_helper("nn.dense")
_register_external_op_helper("nn.max_pool2d")
_register_external_op_helper("nn.avg_pool2d")
_register_external_op_helper("nn.max_pool3d")
_register_external_op_helper("nn.avg_pool3d")
_register_external_op_helper("abs")
_register_external_op_helper("clip")
_register_external_op_helper("exp")
_register_external_op_helper("log")
_register_external_op_helper("sqrt")
_register_external_op_helper("round")
_register_external_op_helper("logsumexp")
_register_external_op_helper("nn.relu")
_register_external_op_helper("nn.leaky_relu")
_register_external_op_helper("tanh")
_register_external_op_helper("sigmoid")
_register_external_op_helper("nn.softmax")
_register_external_op_helper("add")
_register_external_op_helper("multiply")


def make_conv_pattern(conv_name, with_bias=True, with_eltwise=None):
    """Create patterns related to conv and conv_transpose.

    Parameters
    ----------
    with_bias : bool
        Whether attach `bias_add` to `conv / conv_transpose`.
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
    conv = is_op(conv_name)(data, weight)
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


def make_dnnl_pattern(op_name, with_bias, with_eltwise):
    """Create dnnl patterns.

    Parameters
    ----------
    op_name : str
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
    pat_name = op_name.replace("nn", "dnnl")
    if "_transpose" in op_name:
        pat_name = "dnnl.deconv" + op_name.split("_")[0][-2::]
    pat_name += "_bias" if with_bias else ""
    pat_name += ("_" + with_eltwise.split(".")[-1]) if with_eltwise else ""
    if "conv" in op_name:
        dnnl_pattern = (pat_name, make_conv_pattern(op_name, with_bias, with_eltwise))
    elif op_name == "nn.dense":
        dnnl_pattern = (pat_name, make_dense_pattern(with_bias, with_eltwise))
    else:
        logger.warning(
            "Currently, only conv1d, conv2d, conv2d_transpose, conv3d_transpose and "
            "dense op are supported, but got %s.",
            op_name,
        )
        dnnl_pattern = ()
    return dnnl_pattern


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
                return dnnl_patterns
            for conv_name in [
                "nn.conv1d",
                "nn.conv2d",
                "nn.conv3d",
                "nn.conv2d_transpose",
                "nn.conv3d_transpose",
            ]:
                dnnl_patterns.append(make_dnnl_pattern(conv_name, with_bias, elt))
            dnnl_patterns.append(make_dnnl_pattern("nn.dense", with_bias, elt))
    return dnnl_patterns


def get_optimal_layout_for_conv(input_size, weight_shape, out_shape, paddings, strides, dilates, groups):
    """Get the optimal layout of dnnl, given shape of conv2d.

    Parameters
    ----------
    input_size, weight_shape, out_shape, paddings, strides, dilates, groups : Int, String
                                                                              Input argument.

    Returns
    -------
    layouts : string
              The result.
    """
    return _ffi_api.get_optimal_layout_for_conv(
        input_size,
        weight_shape,
        out_shape,
        paddings,
        strides,
        dilates,
        groups,
    )


def get_optimal_layout_for_conv_transpose(
    input_size, weight_shape, out_shape, paddings, output_paddings, strides, dilates, groups
):
    """Get the optimal layout of dnnl, given shape of tranposed conv2d.

    Parameters
    ----------
    input_size, weight_shape, out_shape, paddings, output_paddings, strides, dilates, groups
        : Int, String
          Input argument.

    Returns
    -------
    layouts : string
              The result.
    """
    return _ffi_api.get_optimal_layout_for_conv_transpose(
        input_size,
        weight_shape,
        out_shape,
        paddings,
        output_paddings,
        strides,
        dilates,
        groups,
    )


def get_shape(tensor):
    """Get tensor's shape."""
    if isinstance(tensor, relay.expr.Var):
        return tensor.type_annotation.concrete_shape
    elif isinstance(tensor, relay.expr.Constant):
        return tensor.data.shape
    elif isinstance(tensor, tvm.ir.tensor_type.TensorType):
        return tensor.concrete_shape
    elif isinstance(tensor, tvm.ir.container.Array):
        return tensor[-1].shape
    elif isinstance(tensor, relay.expr.Call):
        return tensor.checked_type.shape
    else:
        raise TypeError("Unsupport data type: %s" % type(tensor))


def tag2layout(input_data, is_weight=False, conv_type="Conv1D"):
    """Transfer layout, denoted with `a, b, c, d, e`,
    into valid layout (NCHW / OIHW) of TVM."""
    if conv_type == "Conv1D":
        data_dic = {"a": "N", "b": "C", "c": "W"}
        weight_dic = {"a": "O", "b": "I", "c": "W", "d": "G"}
    elif conv_type == "Conv2D":
        data_dic = {"a": "N", "b": "C", "c": "H", "d": "W"}
        weight_dic = {"a": "O", "b": "I", "c": "H", "d": "W"}
        if "e" in input_data:
            weight_dic = {"a": "G", "b": "O", "c": "I", "d": "H", "e": "W"}
    elif conv_type == "Conv3D":
        data_dic = {"a": "N", "b": "C", "c": "D", "d": "H", "e": "W"}
        weight_dic = {"a": "O", "b": "I", "c": "D", "d": "H", "e": "W", "f": "G"}

    dic = weight_dic if is_weight else data_dic
    res = ""

    for i in input_data:
        if i.isupper():
            i = i.lower()
            res += dic[i]
            dic[i] = dic[i].lower()
        elif i.islower():
            res += dic[i]
        elif i.isdigit():
            res += i
        else:
            raise ValueError("Unsupport layout format: %s" % input_data)
    return res


def legalize_group_conv(attrs, inputs, types):
    """Legalize group conv / conv_transpose calculation.
    Alter weight layout from OIHW to GOIHW / IOHW to GIOHW"""
    groups = attrs.groups
    if groups == 1:
        return
    data, weight = inputs
    OC, IC, H, W = get_shape(weight)
    new_attrs = dict(attrs)
    weight = relay.reshape(weight, (groups, OC // groups, IC, H, W))
    if "Transpose" not in type(attrs).__name__:
        new_attrs["kernel_layout"] = "GOIHW"
        return relay.nn.conv2d(data, weight, **new_attrs)
    else:
        new_attrs["kernel_layout"] = "GIOHW"
        return relay.nn.conv2d_transpose(data, weight, **new_attrs)


def alter_conv(attrs, inputs, tinfos, out_type):
    """The convolution's layout auto-query func for dnnl."""

    data, weight = inputs
    groups = str(attrs.groups)
    weight_shape = ",".join([str(x) for x in get_shape(weight)])
    out_shape = ",".join([str(x) for x in get_shape(out_type)])
    paddings = ",".join([str(x) for x in attrs.get_int_tuple("padding")])
    strides = ",".join([str(x) for x in attrs.get_int_tuple("strides")])
    dilates = ",".join([str(x) for x in attrs.get_int_tuple("dilation")])
    new_attrs = dict(attrs)
    conv_type = type(attrs).__name__.split("Attrs")[0]

    res = get_optimal_layout_for_conv(
        len(get_shape(out_type)), weight_shape, out_shape, paddings, strides, dilates, groups
    )
    src_df, weight_df, dst_df = res.split(",")
    new_attrs["data_layout"] = tag2layout(src_df, is_weight=False, conv_type=conv_type)
    new_attrs["kernel_layout"] = tag2layout(
        weight_df, is_weight=True, conv_type=conv_type
    )
    new_attrs["out_layout"] = tag2layout(dst_df, is_weight=False, conv_type=conv_type)

    if conv_type == "Conv1D":
        return relay.nn.conv1d(data, weight, **new_attrs)
    elif conv_type == "Conv2D":
        return relay.nn.conv2d(data, weight, **new_attrs)
    elif conv_type == "Conv3D":
        return relay.nn.conv3d(data, weight, **new_attrs)


def alter_conv_transpose(attrs, inputs, tinfos, out_type):
    """The transposed convolution's layout auto-query func for dnnl."""

    data, weight = inputs
    weight_shape = ",".join([str(x) for x in get_shape(weight)])
    out_shape = ",".join([str(x) for x in get_shape(out_type)])
    paddings = ",".join([str(x) for x in attrs.get_int_tuple("padding")])
    output_paddings = ",".join([str(x) for x in attrs.get_int_tuple("output_padding")])
    strides = ",".join([str(x) for x in attrs.get_int_tuple("strides")])
    dilates = ",".join([str(x) for x in attrs.get_int_tuple("dilation")])
    groups = str(attrs.groups)
    new_attrs = dict(attrs)
    conv_type = type(attrs).__name__.split("Attrs")[0]

    res = get_optimal_layout_for_conv_transpose(
        len(get_shape(out_type)),
        weight_shape,
        out_shape,
        paddings,
        output_paddings,
        strides,
        dilates,
        groups,
    )
    src_df, weight_df, dst_df = res.split(",")
    new_attrs["data_layout"] = tag2layout(src_df, is_weight=False, conv_type=conv_type)
    new_attrs["kernel_layout"] = tag2layout(
        weight_df, is_weight=True, conv_type=conv_type
    )
    new_attrs["out_layout"] = tag2layout(dst_df, is_weight=False, conv_type=conv_type)

    if conv_type == "Conv1DTranspose":
        return relay.nn.conv1d_transpose(data, weight, **new_attrs)
    elif conv_type == "Conv2DTranspose":
        return relay.nn.conv2d_transpose(data, weight, **new_attrs)
    elif conv_type == "Conv3DTranspose":
        return relay.nn.conv3d_transpose(data, weight, **new_attrs)


def partition_for_dnnl(mod, params=None, alter_layout=True):
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
    from tvm.relay.testing.temp_op_attr import TempOpAttr

    with TempOpAttr("nn.conv2d", "FTVMLegalize", legalize_group_conv):
        with TempOpAttr("nn.conv2d_transpose", "FTVMLegalize", legalize_group_conv):
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
                    # alter group conv /conv_transpose layout to `GOIHW` / `GIOHW`
                    transform.Legalize(),
                    transform.FoldConstant(),
                ]
            )
            with tvm.transform.PassContext(opt_level=3):
                mod = seq(mod)
    if alter_layout:
        from tvm.relay.testing.temp_op_attr import TempOpAttr

        with TempOpAttr("nn.conv1d", "FTVMAlterOpLayout", alter_conv):
            with TempOpAttr("nn.conv2d", "FTVMAlterOpLayout", alter_conv):
                with TempOpAttr("nn.conv3d", "FTVMAlterOpLayout", alter_conv):
                    with TempOpAttr(
                        "nn.conv2d_transpose", "FTVMAlterOpLayout", alter_conv_transpose
                    ):
                        with TempOpAttr(
                            "nn.conv3d_transpose", "FTVMAlterOpLayout", alter_conv_transpose
                        ):
                            alter_layout_seq = tvm.transform.Sequential(
                                [
                                    transform.AlterOpLayout(),
                                    transform.FoldConstant(),
                                ]
                            )
                            with tvm.transform.PassContext(opt_level=3):
                                mod = alter_layout_seq(mod)

    byoc_seq = tvm.transform.Sequential(
        [
            transform.MergeComposite(pattern_table()),
            transform.AnnotateTarget("dnnl"),
            transform.MergeCompilerRegions(),
            transform.PartitionGraph(),
        ]
    )
    with tvm.transform.PassContext(opt_level=3):
        mod = byoc_seq(mod)
    return mod
