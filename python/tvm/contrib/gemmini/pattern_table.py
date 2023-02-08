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
"""
Pattern table declaring the supported Gemmini operators
=====================
**Author**: `Federico Peccia <https://fPecc.github.io/>`_
"""

from typing import Callable, List, Tuple

import tvm  # type: ignore
from tvm import relay
from tvm.relay.op.contrib.register import register_pattern_table  # type: ignore
from tvm.relay.dataflow_pattern import is_constant, wildcard, is_op
from tvm.relay.frontend.common import infer_shape as _infer_shape
from .utils import QDenseArgs, RequantArgs, BinaryElementwiseArgs, QConv2DArgs

from .environment import Environment

ENV = Environment.instance()


class GEMMParams:
    """
    This class will parse a Call to a gemmini.gemm composite function
    """

    composite_name = "gemmini.gemm"

    def __init__(self, func_body: tvm.relay.Function):

        dense_op = func_body.args[0]
        self.weights = func_body.args[1]
        requantize_op = func_body

        bias_add = requantize_op.args[0]
        self.bias = bias_add.args[1]
        dense_op = bias_add.args[0]
        self.ifm_scale = dense_op.args[QDenseArgs.IFM_SCALE.value]
        self.ifm_offset = dense_op.args[QDenseArgs.IFM_ZERO_POINT.value]

        if requantize_op.op.name == "qnn.requantize":
            self.merge_requantize = True
            self.bias_scale = requantize_op.args[RequantArgs.IFM_SCALE.value]
            self.bias_offset = requantize_op.args[RequantArgs.IFM_ZERO_POINT.value]
            self.ofm_scale = requantize_op.args[RequantArgs.OFM_SCALE.value]
            self.ofm_offset = requantize_op.args[RequantArgs.OFM_ZERO_POINT.value]
        else:
            self.merge_requantize = False
            self.bias_scale = tvm.relay.const([1.0], "float")
            self.bias_offset = tvm.relay.const(0, "int32")
            self.ofm_scale = tvm.relay.const(1.0, "float")
            self.ofm_offset = tvm.relay.const(0, "int32")

    def is_valid(self) -> bool:
        """
        This function checks whether gemmini.gemm has compatible attributes with the Gemmini
        """
        # TODO (FP): complete this validation
        return True


class AddParams:
    """
    This class will parse a Call to a gemmini.add composite function
    """

    composite_name = "gemmini.add"
    activation_map = {"clip": "CLIP"}

    def __init__(self, func_body: tvm.relay.Function):
        if str(func_body.op) in self.activation_map:
            add_op = func_body.args[0]
        else:
            add_op = func_body

        self.ifm1_scale = add_op.args[BinaryElementwiseArgs.IFM1_SCALE.value]
        self.ifm1_offset = add_op.args[BinaryElementwiseArgs.IFM1_ZERO_POINT.value]
        self.ifm2_scale = add_op.args[BinaryElementwiseArgs.IFM2_SCALE.value]
        self.ifm2_offset = add_op.args[BinaryElementwiseArgs.IFM2_ZERO_POINT.value]
        self.ofm_scale = add_op.args[BinaryElementwiseArgs.OFM_SCALE.value]
        self.ofm_offset = add_op.args[BinaryElementwiseArgs.OFM_ZERO_POINT.value]
        self.output_shape = _infer_shape(add_op)
        self.ifm1_shape = _infer_shape(add_op.args[0])
        self.ifm2_shape = _infer_shape(add_op.args[1])

    def is_valid(self) -> bool:
        """
        This function checks whether gemmini.add has compatible attributes with the Gemmini
        """
        # TODO (FP): complete this validation
        # We only support 4 dimensions add operators... for now
        if len(self.output_shape) != 4:
            return False
        if self.ifm1_shape != self.ifm2_shape:
            return False
        return True


class CONV2DParams:
    """
    This class will parse a Call to a gemmini.conv2d composite function
    """

    composite_name = "gemmini.conv2d"
    activation_map = {"clip": "CLIP"}

    def __init__(self, func_body: tvm.relay.Function):
        activation = None
        self.pool_size = [0, 0]
        self.pool_strides = [0, 0]
        self.pool_padding = [0, 0, 0, 0]
        self.pool_dilation = [0, 0]
        self.has_pool = False
        self.has_activation = False
        self.a_min = None
        self.a_max = None
        self.has_external_pad = False
        self.activation_scale_in = tvm.relay.const(1.0, "float")
        self.activation_offset_in = tvm.relay.const(0, "int32")
        self.activation_scale_out = tvm.relay.const(1.0, "float")
        self.activation_offset_out = tvm.relay.const(0, "int32")

        _op = func_body

        if _op.args[0].op.name != "nn.bias_add":

            if _op.op.name == "clip":
                _op = _op.args[0]
            else:

                if _op.op.name == "nn.max_pool2d":
                    max_pool = _op
                    self.pool_size = max_pool.attrs.pool_size
                    self.pool_strides = max_pool.attrs.strides
                    self.pool_padding = max_pool.attrs.padding
                    self.pool_dilation = max_pool.attrs.dilation
                    self.has_pool = True
                    _op = max_pool.args[0]

                if _op.op.name == "clip":
                    _op = _op.args[0]
                elif _op.args[0].op.name == "clip":
                    self.activation_scale_in = _op.args[RequantArgs.IFM_SCALE.value]
                    self.activation_offset_in = _op.args[RequantArgs.IFM_ZERO_POINT.value]
                    self.activation_scale_out = _op.args[RequantArgs.OFM_SCALE.value]
                    self.activation_offset_out = _op.args[RequantArgs.OFM_ZERO_POINT.value]
                    clip = _op.args[0]
                    self.has_activation = True
                    _min = clip.args[0]
                    self.a_min = clip.attrs.a_min
                    self.a_max = clip.attrs.a_max
                    _op = _min.args[0]

        requantize_op = _op

        bias_add = requantize_op.args[0]

        conv2d_op = bias_add.args[0]

        self.has_input_requantize = False
        self.input_scale_in = tvm.relay.const(1.0, "float")
        self.input_offset_in = tvm.relay.const(0, "int32")
        self.input_scale_out = tvm.relay.const(1.0, "float")
        self.input_offset_out = tvm.relay.const(0, "int32")

        self.output_shape = _infer_shape(conv2d_op)
        self.strides = conv2d_op.attrs.strides
        self.padding = conv2d_op.attrs.padding
        self.groups = conv2d_op.attrs.groups
        self.is_depthwise = self.groups == conv2d_op.attrs.channels and self.groups != 1
        self.data = conv2d_op.args[0]
        self.input_shape = _infer_shape(self.data)
        if (
            not isinstance(self.data, relay.expr.Var)
            and not isinstance(self.data.op, relay.function.Function)
            and self.data.op.name == "nn.pad"
        ):
            padding = self.data.attrs.pad_width
            self.padding = [padding[1][0], padding[1][1], padding[2][0], padding[2][1]]
            self.has_external_pad = True
        self.weights = conv2d_op.args[1]
        self.weights_shape = _infer_shape(self.weights)
        self.bias = bias_add.args[1]
        self.ifm_scale = float(conv2d_op.args[QConv2DArgs.IFM_SCALE.value].data.numpy())
        self.ifm_offset = conv2d_op.args[QConv2DArgs.IFM_ZERO_POINT.value]
        self.ifm_offset_const = conv2d_op.args[QConv2DArgs.IFM_ZERO_POINT.value]
        self.weights_scale = 1.0
        self.weights_offset = 0.0

        if requantize_op.op.name == "qnn.requantize":
            self.bias_scale = requantize_op.args[RequantArgs.IFM_SCALE.value]
            self.bias_offset = requantize_op.args[RequantArgs.IFM_ZERO_POINT.value]
            self.ofm_scale = requantize_op.args[RequantArgs.OFM_SCALE.value]
            self.ofm_offset = requantize_op.args[RequantArgs.OFM_ZERO_POINT.value]
        else:
            self.bias_scale = tvm.relay.const([1.0], "float")
            self.bias_offset = tvm.relay.const(0, "int32")
            self.ofm_scale = tvm.relay.const(1.0, "float")
            self.ofm_offset = tvm.relay.const(0, "int32")

        if activation is not None:
            self.activation = False
        else:
            self.activation = False

    def is_valid(self) -> bool:
        """
        This function checks whether gemmini.conv2d has compatible attributes with the Gemmini
        """
        # TODO (FP): complete this validation
        if len(set(self.pool_padding)) != 1 or len(set(self.pool_strides)) != 1:
            return False

        if self.has_input_requantize:
            if (
                self.input_scale_in.data.numpy() != self.input_scale_out.data.numpy()
                or self.input_offset_in.data.numpy() != 0
            ):
                # Only this specific cases are supported, for now...
                return False

        if self.a_max is not None and self.a_max != 127:
            return False

        return True


class DepthwiseCONV2DParams(CONV2DParams):
    """
    This class will parse a Call to a gemmini.depthwiseconv2d composite function
    """

    composite_name = "gemmini.depthwiseconv2d"
    activation_map = {"clip": "CLIP"}


class MaxPoolParams:
    """
    This class will parse a Call to a gemmini.max_pool2d composite function
    """

    composite_name = "gemmini.max_pool2d"

    def __init__(self, func_body: tvm.relay.Function):
        self.pool_size = func_body.attrs.pool_size
        self.pool_strides = func_body.attrs.strides
        self.pool_padding = func_body.attrs.padding
        self.pool_dilation = func_body.attrs.dilation
        self.shape = _infer_shape(func_body)

    def is_valid(self) -> bool:
        """
        This function checks whether max_pool2d has compatible attributes with the Gemmini
        """
        # TODO (FP): complete this validation?
        if len(set(self.pool_padding)) != 1:
            return False
        if (self.shape[1] != self.shape[2]) or self.shape[1] == 1:
            return False
        return True


def make_dense_pattern() -> tvm.relay.dataflow_pattern.DFPattern:
    """Create patterns related to qnn.dense.

    Parameters
    ----------

    Returns
    -------
    dense_out : CallPattern
        Call node sequence.
    """
    data = wildcard()
    weight = wildcard()
    bias = wildcard()
    dense = is_op("qnn.dense")(
        data, weight, is_constant(), is_constant(), is_constant(), is_constant()
    )
    bias_add = is_op("nn.bias_add")(
        dense,
        bias,
    )
    req = is_op("qnn.requantize")(
        bias_add, is_constant(), is_constant(), is_constant(), is_constant()
    )
    return req


def make_add_pattern() -> tvm.relay.dataflow_pattern.DFPattern:
    """Create patterns related to qnn.add.

    Parameters
    ----------

    Returns
    -------
    add_out : CallPattern
        Call node sequence.
    """
    ifm1 = wildcard()
    ifm2 = wildcard()
    add_out = is_op("qnn.add")(
        ifm1,
        ifm2,
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
    )
    clip_or_req = add_out.optional(is_op("clip"))
    return clip_or_req


def make_conv2d_pattern(
    with_padded_input: bool = False, with_maxpool: bool = False, with_relu_6: bool = False
) -> tvm.relay.dataflow_pattern.DFPattern:
    """Create patterns related to qnn.conv2d.

    Parameters
    ----------

    Returns
    -------
    conv2d_out : CallPattern
        Call node sequence.
    """
    data = wildcard()
    if with_padded_input:
        data = is_op("nn.pad")(data, wildcard())
    weight = wildcard()
    bias = wildcard()
    conv2d_out = is_op("qnn.conv2d")(
        data, weight, is_constant(), is_constant(), is_constant(), is_constant()
    )
    bias_add = is_op("nn.bias_add")(
        conv2d_out,
        bias,
    )
    output = is_op("qnn.requantize")(
        bias_add, is_constant(), is_constant(), is_constant(), is_constant()
    )
    if with_relu_6:
        output = is_op("minimum")(output, wildcard())
        output = is_op("clip")(output)
        output = is_op("qnn.requantize")(
            output, is_constant(), is_constant(), is_constant(), is_constant()
        )
    else:
        output = output.optional(is_op("clip"))
    if with_maxpool:
        output = output.optional(is_op("nn.max_pool2d"))
        return output
    else:
        return output


def make_depthwiseconv2d_pattern() -> tvm.relay.dataflow_pattern.DFPattern:
    """Create patterns related to qnn.conv2d, but only if it is a depthwise convolution.

    Parameters
    ----------

    Returns
    -------
    conv2d_out : CallPattern
        Call node sequence.
    """
    data = wildcard()
    weight = wildcard()
    bias = wildcard()
    conv2d_out = is_op("qnn.conv2d")(
        data, weight, is_constant(), is_constant(), is_constant(), is_constant()
    ).has_attr({"kernel_layout": "HWOI"})
    bias_add = is_op("nn.bias_add")(
        conv2d_out,
        bias,
    )
    output = is_op("qnn.requantize")(
        bias_add, is_constant(), is_constant(), is_constant(), is_constant()
    )
    clip_or_req = output.optional(is_op("clip"))
    return clip_or_req


def make_maxpool_pattern() -> tvm.relay.dataflow_pattern.DFPattern:
    """Create patterns related to nn.max_pool2d.

    Parameters
    ----------

    Returns
    -------
    max_pool2d : CallPattern
        Call node sequence.
    """
    max_pool2d = is_op("nn.max_pool2d")(wildcard())
    return max_pool2d


@register_pattern_table("gemmini")
def pattern_table() -> List[Tuple[str, tvm.relay.dataflow_pattern.DFPattern, Callable]]:
    """Declares Gemminis pattern table

    Returns:
        List[Tuple[str, tvm.relay.dataflow_pattern.DFPattern, Callable]]: List of pattern, callable tuples
    """

    pattern_table_filters = []
    pattern_table_filters.append(
        (
            GEMMParams.composite_name,
            make_dense_pattern(),
            lambda pat: GEMMParams(pat).is_valid(),
        )
    )

    for pad in [True, False]:
        for max_pool in [True, False]:
            for relu6 in [True, False]:
                pattern_table_filters.append(
                    (
                        CONV2DParams.composite_name,
                        make_conv2d_pattern(
                            with_padded_input=pad, with_maxpool=max_pool, with_relu_6=relu6
                        ),
                        lambda pat: CONV2DParams(pat).is_valid(),
                    )
                )

    pattern_table_filters.append(
        (
            MaxPoolParams.composite_name,
            make_maxpool_pattern(),
            lambda pat: MaxPoolParams(pat).is_valid(),
        )
    )

    if ENV.use_experimental_qnn_add:
        pattern_table_filters.append(
            (
                AddParams.composite_name,
                make_add_pattern(),
                lambda pat: AddParams(pat).is_valid(),
            )
        )

    return pattern_table_filters
