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
# pylint: disable=ungrouped-imports, import-outside-toplevel
"""Arm(R) Ethos(TM)-U NPU supported operators."""
import functools
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np  # type: ignore

import tvm  # type: ignore
from tvm import relay
from tvm.ir import Op
from tvm.relay.build_module import bind_params_by_name  # type: ignore
from tvm.relay.dataflow_pattern import (  # type: ignore
    is_constant,
    is_op,
    is_tuple,
    wildcard,
)
from tvm.relay.expr import Call, Constant  # type: ignore
from tvm.relay.op.contrib.register import register_pattern_table  # type: ignore

try:
    # As ethos-u-vela package is an optional TVM dependency, we want to lazy load it
    # and check whether it is installed or not.
    #
    # In order to show the appropriate error messages when we try to invoke code that
    # rely on imports from ethos-u-vela, we protect them with the decorator @requires_vela
    # implemented below.
    from ethosu.vela import api as vapi  # type: ignore
except ImportError:
    vapi = None


def requires_vela(func):
    """Decorator to check whether we have the required dependency ethos-u-vela
    installed as a python package"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not vapi:
            raise ImportError(
                "The 'ethos-u-vela' python package is required for the Arm(R) Ethos(TM)-U NPU "
                "backend. Please install the dependency using your Python package manager."
            ) from None
        return func(*args, **kwargs)

    return wrapper


class TensorParams:
    """
    This class will parse a tvm Expr along with quantization scale
    and zero point to populate parameters that are required
    for the creation of tensors in Vela.
    """

    @requires_vela
    def __init__(self, tensor, layout=None, scale=None, zero_point=None):
        self.tensor = tensor
        if isinstance(tensor, Constant):
            self.values = tensor.data.asnumpy()
        else:
            self.values = None
        self.dtype = tensor.checked_type.dtype
        self.shape = [int(i) for i in tensor.checked_type.shape]
        self.layout = layout

        if scale is not None and zero_point is not None:
            self.q_params = vapi.NpuQuantization(
                scale.data.asnumpy().astype("float32"), zero_point.data.asnumpy().astype(self.dtype)
            )
        else:
            # put default values
            self.q_params = vapi.NpuQuantization(1.0, 0)


def check_strides(strides: List[int], stride_range=None) -> bool:
    """This function checks whether strides are within the limits supported by the NPU"""
    if stride_range is None:
        stride_range = (1, 3)
    smin, smax = stride_range
    if not smax >= strides[0] >= smin:
        return False
    if not smax >= strides[1] >= smin:
        return False
    return True


def check_same_ifm_and_kernel_shape(padding, ifm_shape, pool_shape):
    """
    This function checks whether AvgPool2D or MaxPool2D could be legalized as ethosu_pooling
    supported by the NPU.
    We consider only specific case: when there is no AvgPool2D padding, the spatial
    dimensions of ifm and the shape of pooling are equal, but stride size exceed 3
    by any of dimensions, e.g:
    ifm: (1, 8, 8, _), strides: (8, 8), pool_shape: (8, 8)
    ifm: (1, 25, 5, _), strides: (25, 5), pool_shape: (25, 5)
    """
    if list(padding) != [0, 0, 0, 0]:
        return False
    if [ifm_shape[1], ifm_shape[2]] != list(pool_shape):
        return False
    return True


def check_valid_dtypes(tensor_params: List[TensorParams], supported_dtypes: List[type]) -> bool:
    """This function checks whether dtypes are supported by the NPU"""
    for tep in tensor_params:
        # Check for dtypes
        if np.dtype(tep.dtype) not in supported_dtypes:
            return False
        # Check for shape sizes
        if any(dimlen > 65536 for dimlen in tep.shape):
            return False
    return True


def check_weights(weights: TensorParams, dilation: List[int]):
    """This function checks whether weight tensor is compatible with the NPU"""
    from tvm.relay.backend.contrib.ethosu.util import get_dim_value

    dilated_height_range = (1, 64)
    dilated_hxw_range = (1, 64 * 64)
    weights_limit = 127 * 65536
    dilated_width = (weights.shape[get_dim_value(weights.layout, "W")] - 1) * dilation[0] + 1
    dilated_height = (weights.shape[get_dim_value(weights.layout, "H")] - 1) * dilation[1] + 1
    dh_min, dh_max = dilated_height_range
    if not dh_min <= dilated_height <= dh_max:
        return False
    dilated_hxw = dilated_height * dilated_width
    dhxw_min, dhxw_max = dilated_hxw_range
    if not dhxw_min <= dilated_hxw <= dhxw_max:
        return False
    # A saturation upper bound check for accumulators
    weights.values = weights.values - weights.q_params.zero_point
    axis = (
        get_dim_value(weights.layout, "H"),
        get_dim_value(weights.layout, "W"),
        get_dim_value(weights.layout, "I"),
    )
    sum_weights = np.amax(np.sum(np.absolute(weights.values), axis=axis))
    return sum_weights <= weights_limit


def check_bias(bias: TensorParams):
    """This function checks whether the bias values fit in 40 bits"""
    if bias and bias.dtype == np.dtype("int64"):
        valid = all(len(bin(bias_value)[2:]) <= 40 for bias_value in bias.values)
        return valid
    return True


def check_batch_size(ifm: TensorParams):
    """This function checks for the number of batches vela currently supports"""
    return ifm.shape[0] == 1


def check_dilation(dilation: List[int], dilation_range=None):
    """This function checks whether dilation is within the limits supported by the NPU"""
    if dilation_range is None:
        dilation_range = (1, 2)
    dmin, dmax = dilation_range
    if not dmin <= dilation[0] <= dmax:
        return False
    if not dmin <= dilation[1] <= dmax:
        return False
    return True


def check_padding(padding: List[int], bounds: List[int]):
    """This function checks whether padding is within the limits supported by the NPU"""
    if len(padding) != 4 or len(bounds) != 4:
        return False
    top, left, bottom, right = padding
    topb, leftb, bottomb, rightb = bounds
    return not (top > topb or left > leftb or bottom > bottomb or right > rightb)


def check_pool_shape(pool_shape: tvm.ir.container.Array) -> bool:
    if len(pool_shape) != 2:
        return False
    if pool_shape[1] > 256:
        return False
    if pool_shape[0] * pool_shape[1] > 256 * 256:
        return False
    return True


def check_dimensions(tensor: TensorParams):
    """This function checks that the tensor has no more than 4 dimensions"""
    return len(tensor.shape) <= 4


class QnnConv2DParams:
    """
    This class will parse a Call to a ethosu.qnn_conv2d composite function
    and extract quantization information of all the associated tensors.
    """

    composite_name = "ethos-u.qnn_conv2d"
    # The NPU only supports padding upto the numbers as follows
    padding_bounds = [31, 31, 32, 32]
    activation_map = {"clip": "CLIP"}

    @requires_vela
    def __init__(self, func_body: tvm.relay.Function):
        from tvm.relay.backend.contrib.ethosu.util import QConv2DArgs  # type: ignore
        from tvm.relay.backend.contrib.ethosu.util import BiasAddArgs, RequantArgs

        activation = None
        separate_padding = None

        if str(func_body.op.name) in self.activation_map.keys():
            activation = func_body
            requantize_op = activation.args[0]
        else:
            requantize_op = func_body
        bias_add = requantize_op.args[0]
        qnn_conv2d = bias_add.args[0]
        if (
            isinstance(qnn_conv2d.args[0], relay.Call)
            and isinstance(qnn_conv2d.args[0].op, Op)
            and str(qnn_conv2d.args[0].op.name) == "nn.pad"
        ):
            separate_padding = qnn_conv2d.args[0]
        data_layout = qnn_conv2d.attrs.data_layout
        self.kernel_layout = qnn_conv2d.attrs.kernel_layout

        # We consider the weights & biases as params as it should be a Constant
        self.weights = TensorParams(
            qnn_conv2d.args[QConv2DArgs.WEIGHTS.value],
            self.kernel_layout,
            qnn_conv2d.args[QConv2DArgs.WEIGHTS_SCALE.value],
            qnn_conv2d.args[QConv2DArgs.WEIGHTS_ZERO_POINT.value],
        )

        self.biases = TensorParams(
            bias_add.args[BiasAddArgs.BIASES.value],
            data_layout,
            requantize_op.args[RequantArgs.IFM_SCALE.value],
            requantize_op.args[RequantArgs.IFM_ZERO_POINT.value],
        )
        ifm_tensor = (
            separate_padding.args[0] if separate_padding else qnn_conv2d.args[QConv2DArgs.IFM.value]
        )
        self.ifm = TensorParams(
            ifm_tensor,
            data_layout,
            qnn_conv2d.args[QConv2DArgs.IFM_SCALE.value],
            qnn_conv2d.args[QConv2DArgs.IFM_ZERO_POINT.value],
        )
        self.ofm = TensorParams(
            func_body,
            data_layout,
            requantize_op.args[RequantArgs.OFM_SCALE.value],
            requantize_op.args[RequantArgs.OFM_ZERO_POINT.value],
        )
        attrs = qnn_conv2d.attrs

        pad_value = int(qnn_conv2d.args[QConv2DArgs.IFM_ZERO_POINT.value].data.asnumpy())
        self.padding = self.extract_padding(attrs.padding, separate_padding, pad_value)

        self.strides = attrs.strides
        self.dilation = attrs.dilation
        self.activation = activation
        self.channels = attrs.channels

        # If groups are equal to channel, its a depthwise_conv2d
        self.groups = attrs.groups
        self.is_depthwise = False
        channels_axis = {"HWIO": 3, "HWOI": 2}
        if self.groups == self.weights.shape[channels_axis[self.kernel_layout]]:
            self.is_depthwise = True

    @staticmethod
    def extract_padding(
        operator_padding: Tuple[int, int, int, int],
        separate_padding: relay.Call,
        pad_value: int,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Convolution operations can sometimes have padding represented as a separate
        padding operation before the convolution operation itself. Here we can check
        whether these representations can be combined into a single padding attribute
        as part of the NPU convolution itself. If the padding specified by the separate
        nn.pad operation is not supported, None will be returned. This will cause the
        nn.pad to be offloaded separately.
        """
        if separate_padding is None:
            return operator_padding
        if pad_value != int(separate_padding.args[1].data.asnumpy()):
            return None
        pad_width = separate_padding.attrs["pad_width"]
        if len(pad_width) != 4:
            return None
        if list(pad_width[0]) != [0, 0] or list(pad_width[3]) != [0, 0]:
            return None
        top, left, bottom, right = operator_padding
        return [
            top + pad_width[1][0],
            left + pad_width[2][0],
            bottom + pad_width[1][1],
            right + pad_width[2][1],
        ]

    def is_valid(self) -> bool:
        """
        This function checks whether QnnConv2D has compatible attributes with the NPU
        """
        tensor_params = [self.weights, self.ifm, self.ofm]
        if not check_valid_dtypes(tensor_params, supported_dtypes=[np.uint8, np.int8]):
            return False
        if not check_weights(self.weights, self.dilation):
            return False
        if not check_bias(self.biases):
            return False
        if not check_strides(self.strides):
            return False
        if not check_batch_size(self.ifm):
            return False
        if not check_dilation(self.dilation):
            return False
        if not self.padding or not check_padding(self.padding, self.padding_bounds):
            return False
        legal_groups = [1, self.ofm.shape[3]]
        if self.groups not in legal_groups:
            return False
        # This should be a valid QnnDepthwiseConv2DParams, not QnnConv2DParams
        return not self.is_depthwise


class QnnConv2DTransposeParams:
    """
    This class will parse a Call to a ethosu.qnn_conv2d_transpose composite
    function and extract quantization information of all the associated tensors.
    """

    composite_name = "ethos-u.qnn_conv2d_transpose"
    # The NPU only supports padding upto the numbers as follows
    padding_bounds = [31, 31, 32, 32]

    @requires_vela
    def __init__(self, func_body: tvm.relay.Function):
        from tvm.relay.backend.contrib.ethosu.util import (
            QConv2DTransposeArgs,  # type: ignore
        )
        from tvm.relay.backend.contrib.ethosu.util import BiasAddArgs, RequantArgs

        requantize = func_body
        call = func_body.args[0]
        if str(call.op.name) == "nn.bias_add":
            bias_add = call
            call = call.args[0]
        else:
            bias_add = None
        qnn_conv2d_transpose = call

        data_layout = qnn_conv2d_transpose.attrs.data_layout
        self.kernel_layout = qnn_conv2d_transpose.attrs.kernel_layout

        self.weights = TensorParams(
            qnn_conv2d_transpose.args[QConv2DTransposeArgs.WEIGHTS.value],
            self.kernel_layout,
            qnn_conv2d_transpose.args[QConv2DTransposeArgs.WEIGHTS_SCALE.value],
            qnn_conv2d_transpose.args[QConv2DTransposeArgs.WEIGHTS_ZERO_POINT.value],
        )
        self.biases = (
            TensorParams(
                bias_add.args[BiasAddArgs.BIASES.value],
                data_layout,
                requantize.args[RequantArgs.IFM_SCALE.value],
                requantize.args[RequantArgs.IFM_ZERO_POINT.value],
            )
            if bias_add
            else None
        )
        self.ifm = TensorParams(
            qnn_conv2d_transpose.args[QConv2DTransposeArgs.IFM.value],
            data_layout,
            qnn_conv2d_transpose.args[QConv2DTransposeArgs.IFM_SCALE.value],
            qnn_conv2d_transpose.args[QConv2DTransposeArgs.IFM_ZERO_POINT.value],
        )
        self.ofm = TensorParams(
            func_body,
            data_layout,
            requantize.args[RequantArgs.OFM_SCALE.value],
            requantize.args[RequantArgs.OFM_ZERO_POINT.value],
        )

        attrs = qnn_conv2d_transpose.attrs
        self.strides = attrs.strides
        self.dilation = attrs.dilation
        self.padding = attrs.padding
        self.channels = attrs.channels
        self.groups = attrs.groups
        self.output_padding = attrs.output_padding

        kernel_size_map = {
            "IOHW": self.weights.shape[2:4],
        }
        self.kernel_shape = kernel_size_map[str(self.weights.layout)]

        # Different padding is used in the legalization from conv2d_transpose
        # to conv2d, so we to calculate it here to check that the new size fits
        # within the bounds of the NPU before offloading.
        pad_top = int(self.kernel_shape[0]) - 1 - int(self.padding[0])
        pad_left = int(self.kernel_shape[1]) - 1 - int(self.padding[1])
        pad_bottom = int(self.kernel_shape[0]) - 1 - int(self.padding[2])
        pad_right = int(self.kernel_shape[1]) - 1 - int(self.padding[3])
        if self.strides == [2, 2]:
            pad_bottom -= 1
            pad_right -= 1
        self.legalize_padding = [pad_top, pad_left, pad_bottom, pad_right]

    def is_valid(self) -> bool:
        """
        This function checks whether QnnConv2D has compatible attributes with the NPU
        """

        def check_compatible_output_size(ifm_shape, ofm_shape, padding, strides, kernel_shape):
            is_valid_padding = padding == [0, 0, 0, 0]
            if is_valid_padding:
                expected_height = ifm_shape[1] * strides[0] + (kernel_shape[0] - strides[0])
                expected_width = ifm_shape[2] * strides[1] + (kernel_shape[1] - strides[1])
            else:
                expected_height = ifm_shape[1] * strides[0]
                expected_width = ifm_shape[2] * strides[1]
            return ofm_shape[1] == expected_height and ofm_shape[2] == expected_width

        tensor_params = [self.weights, self.ifm, self.ofm]
        if not check_valid_dtypes(tensor_params, supported_dtypes=[np.int8]):
            return False
        if not check_weights(self.weights, self.dilation):
            return False
        if self.biases and not check_bias(self.biases):
            return False
        if not check_strides(self.strides, stride_range=(2, 2)):
            return False
        if not check_batch_size(self.ifm):
            return False
        if not check_dilation(self.dilation, dilation_range=(1, 1)):
            return False
        if not check_compatible_output_size(
            self.ifm.shape,
            self.ofm.shape,
            [int(x) for x in self.padding],
            self.strides,
            self.kernel_shape,
        ):
            return False
        if not check_padding(self.legalize_padding, self.padding_bounds):
            return False
        if self.kernel_shape[0] - 2 - int(self.padding[2]) < 0:
            return False
        if self.kernel_shape[1] - 2 - int(self.padding[3]) < 0:
            return False
        if self.groups != 1:
            return False
        if list(self.output_padding) != [0, 0]:
            return False
        return True


class QnnDepthwiseConv2DParams(QnnConv2DParams):
    """
    This class will parse a call to a ethosu.depthwise_conv2d composite function
    and extract the parameter information.
    """

    composite_name = "ethos-u.depthwise_conv2d"
    # The hardware only supports padding upto the numbers as follows
    padding_bounds = [31, 31, 32, 32]

    def __init__(self, func_body: tvm.relay.expr.Call):
        QnnConv2DParams.__init__(self, func_body)

    def is_valid(self):
        """
        Checks whether QnnDepthwiseConv2D + activation function has compatible attributes with HW
        """
        tensor_params = [self.weights, self.ifm, self.ofm]
        if not check_valid_dtypes(tensor_params, supported_dtypes=[np.uint8, np.int8]):
            return False
        if not check_weights(self.weights, self.dilation):
            return False
        if not check_bias(self.biases):
            return False
        if not check_strides(self.strides):
            return False
        if not check_batch_size(self.ifm):
            return False
        if not check_dilation(self.dilation):
            return False
        if not self.padding or not check_padding(self.padding, self.padding_bounds):
            return False
        if self.weights.layout != "HWOI":
            return False
        # only depth multiplier of size 1 is supported
        if self.weights.shape[3] != 1:
            return False
        if not self.is_depthwise:
            return False
        return True


def qnn_conv2d_pattern() -> tvm.relay.dataflow_pattern.DFPattern:
    """
    This function creates the pattern for qnn.conv2D with optional fused RELU activation.
    """
    optional_pad = is_op("nn.pad")(wildcard(), is_constant())
    qnn_conv2d = is_op("qnn.conv2d")(
        optional_pad | wildcard(),
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
    ).has_attr({"kernel_layout": "HWIO"})
    bias_add = is_op("nn.bias_add")(qnn_conv2d, is_constant())
    req = is_op("qnn.requantize")(
        bias_add, is_constant(), is_constant(), is_constant(), is_constant()
    )
    clip_or_req = req.optional(is_op("clip"))
    return clip_or_req


def qnn_depthwise_conv2d_pattern() -> tvm.relay.dataflow_pattern.DFPattern:
    """
    This function creates the pattern for depthwise qnn.conv2D with optional fused RELU activation.
    """
    optional_pad = is_op("nn.pad")(wildcard(), is_constant())
    qnn_conv2d = is_op("qnn.conv2d")(
        optional_pad | wildcard(),
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
    ).has_attr({"kernel_layout": "HWOI"})
    bias_add = is_op("nn.bias_add")(qnn_conv2d, is_constant())
    req = is_op("qnn.requantize")(
        bias_add, is_constant(), is_constant(), is_constant(), is_constant()
    )
    clip_or_req = req.optional(is_op("clip"))
    return clip_or_req


def qnn_conv2d_transpose_pattern() -> tvm.relay.dataflow_pattern.DFPattern:
    """
    This function creates the pattern for qnn.conv2d_transpose.
    """
    qnn_conv2d_transpose = is_op("qnn.conv2d_transpose")(
        wildcard(), is_constant(), is_constant(), is_constant(), is_constant(), is_constant()
    ).has_attr({"kernel_layout": "IOHW"})
    optional_bias_add = (
        is_op("nn.bias_add")(qnn_conv2d_transpose, is_constant()) | qnn_conv2d_transpose
    )
    req = is_op("qnn.requantize")(
        optional_bias_add, is_constant(), is_constant(), is_constant(), is_constant()
    )
    return req


class MaxPool2DParams:
    """
    This class will parse a call to a ethos-u.maxpool2d composite function
    and extract the parameter information.
    """

    composite_name = "ethos-u.maxpool2d"
    # The hardware only supports padding upto the numbers as follows
    padding_bounds = [127, 127, 128, 128]

    def __init__(self, func_body: Call):
        clip = None
        if str(func_body.op.name) == "clip":
            clip = func_body
            pool_op = clip.args[0]
        else:
            pool_op = func_body

        attrs = pool_op.attrs
        self.ifm = TensorParams(pool_op.args[0], attrs.layout)
        self.ofm = TensorParams(pool_op, attrs.layout)
        self.pool_shape = attrs.pool_size
        self.strides = attrs.strides
        self.padding = attrs.padding
        self.activation = clip
        self.pooling_type = "MAX"

    def is_valid(self):
        """
        This function checks whether MaxPool2D has compatible attributes with the NPU
        """
        tensor_params = [self.ifm, self.ofm]
        if not check_valid_dtypes(tensor_params, supported_dtypes=[np.uint8, np.int8]):
            return False
        if self.ifm.dtype != self.ofm.dtype:
            return False
        if not check_strides(self.strides) and not check_same_ifm_and_kernel_shape(
            self.padding, self.ifm.shape, self.pool_shape
        ):
            return False
        if not check_batch_size(self.ifm):
            return False
        if not check_padding(self.padding, self.padding_bounds):
            return False
        if not check_pool_shape(self.pool_shape):
            return False
        return True


def qnn_maxpool2d_pattern() -> tvm.relay.dataflow_pattern.DFPattern:
    """
    This function creates the pattern for nn.max_pool2d with optional fused RELU activation.
    """
    pattern = is_op("nn.max_pool2d")(wildcard())
    pattern = pattern.optional(is_op("clip"))
    return pattern


class AvgPool2DParams:
    """
    This class will parse a call to a ethos-u.avgpool2d composite function
    and extract the parameter information.
    """

    composite_name = "ethos-u.avgpool2d"
    # The hardware only supports padding upto the numbers as follows
    padding_bounds = [3, 3, 4, 4]

    def __init__(self, func_body: Call):
        clip = None
        if str(func_body.op.name) == "clip":
            clip = func_body
            cast2 = clip.args[0]
        else:
            cast2 = func_body

        avgpool = cast2.args[0]
        cast1 = avgpool.args[0]

        attrs = avgpool.attrs
        self.ifm = TensorParams(cast1.args[0], attrs.layout)
        self.ofm = TensorParams(cast2, attrs.layout)
        self.pool_shape = attrs.pool_size
        self.strides = attrs.strides
        self.padding = attrs.padding
        self.count_include_pad = attrs.count_include_pad
        self.activation = clip
        self.pooling_type = "AVG"

    def is_valid(self):
        """
        This function checks whether AvgPool2D has compatible attributes with the NPU
        """
        tensor_params = [self.ifm, self.ofm]
        if not check_valid_dtypes(tensor_params, supported_dtypes=[np.uint8, np.int8]):
            return False
        if self.ifm.dtype != self.ofm.dtype:
            return False
        if not check_strides(self.strides) and not check_same_ifm_and_kernel_shape(
            self.padding, self.ifm.shape, self.pool_shape
        ):
            return False
        if not check_batch_size(self.ifm):
            return False
        if self.count_include_pad:
            return False
        if not check_padding(self.padding, self.padding_bounds):
            return False
        if not check_pool_shape(self.pool_shape):
            return False
        # Average pool with padding only supports 1 <= pool_shape <= 8
        if list(self.padding) != [0, 0, 0, 0] and (
            self.pool_shape[0] > 8 or self.pool_shape[1] > 8
        ):
            return False
        return True


def qnn_avgpool2d_pattern() -> tvm.relay.dataflow_pattern.DFPattern:
    """
    This function creates the pattern for nn.avg_pool2d with optional fused RELU activation.
    """
    pattern = is_op("cast")(wildcard())
    pattern = is_op("nn.avg_pool2d")(pattern)
    pattern = is_op("cast")(pattern)
    pattern = pattern.optional(is_op("clip"))
    return pattern


class BinaryElementwiseParams:
    """
    This class will parse a call to a ethosu.binary_elementwise composite function
    and extract the parameter information.
    """

    def __init__(self, func_body: Call, operator_type: str, is_quantized_operation: bool):
        from tvm.relay.backend.contrib.ethosu.util import (
            BinaryElementwiseArgs,
            RequantArgs,
        )

        current_call = func_body
        clip = None
        requantize = None

        if str(current_call.op.name) == "clip":
            clip = current_call
            current_call = clip.args[0]
        elif str(current_call.op.name) == "qnn.requantize":
            requantize = current_call
            clip = current_call.args[0]
            current_call = clip.args[0]
        binary_op = current_call

        layout = "NHWC"

        if is_quantized_operation:
            self.ifm = TensorParams(
                binary_op.args[BinaryElementwiseArgs.IFM.value],
                layout,
                binary_op.args[BinaryElementwiseArgs.IFM_SCALE.value],
                binary_op.args[BinaryElementwiseArgs.IFM_ZERO_POINT.value],
            )
            self.ifm2 = TensorParams(
                binary_op.args[BinaryElementwiseArgs.IFM2.value],
                layout,
                binary_op.args[BinaryElementwiseArgs.IFM2_SCALE.value],
                binary_op.args[BinaryElementwiseArgs.IFM2_ZERO_POINT.value],
            )
            self.ofm = TensorParams(
                binary_op,
                layout,
                binary_op.args[BinaryElementwiseArgs.OFM_SCALE.value],
                binary_op.args[BinaryElementwiseArgs.OFM_ZERO_POINT.value],
            )
        else:
            self.ifm = TensorParams(
                binary_op.args[BinaryElementwiseArgs.IFM.value],
                layout,
                requantize.args[RequantArgs.IFM_SCALE.value] if requantize else None,
                requantize.args[RequantArgs.IFM_ZERO_POINT.value] if requantize else None,
            )
            self.ifm2 = TensorParams(
                binary_op.args[BinaryElementwiseArgs.IFM2.value],
                layout,
                requantize.args[RequantArgs.IFM_SCALE.value] if requantize else None,
                requantize.args[RequantArgs.IFM_ZERO_POINT.value] if requantize else None,
            )
            self.ofm = TensorParams(
                func_body,
                layout,
                requantize.args[RequantArgs.OFM_SCALE.value] if requantize else None,
                requantize.args[RequantArgs.OFM_ZERO_POINT.value] if requantize else None,
            )
        self.activation = clip
        self.operator_type = operator_type

        def can_broadcast(ifm, ifm2):
            if len(ifm.shape) < len(ifm2.shape):
                return False
            for m, n in zip(ifm.shape[::-1], ifm2.shape[::-1]):
                if m != n and m == 1:
                    return False
            return True

        if can_broadcast(self.ifm, self.ifm2):
            self.reversed_operands = False
            self.valid_broadcast = True
        elif can_broadcast(self.ifm2, self.ifm):
            self.reversed_operands = True
            self.ifm, self.ifm2 = self.ifm2, self.ifm
            self.valid_broadcast = True
        else:
            self.valid_broadcast = False

    def is_valid(self):
        """
        This function checks whether BinaryElementwise has compatible attributes with the NPU
        """
        if np.dtype(self.ofm) == np.int32 and self.activation is not None:
            return False
        # Due to identity operator requiring ofm != int32 for now
        if np.dtype(self.ofm) == np.int32 and len(self.ofm.shape) < 4:
            return False
        if len(self.ifm.shape) > 4 or len(self.ifm2.shape) > 4:
            return False
        if len(self.ifm.shape) == 4 and self.ifm.shape[0] != 1:
            return False
        if len(self.ifm2.shape) == 4 and self.ifm2.shape[0] != 1:
            return False
        if not self.valid_broadcast:
            return False
        return True


class AddParams(BinaryElementwiseParams):
    """
    This class will parse a call to a ethosu.binary_elementwise Add composite function
    and extract the parameter information.
    """

    composite_name = "ethos-u.add"

    def __init__(self, func_body: Call):
        BinaryElementwiseParams.__init__(self, func_body, "ADD", True)

    def is_valid(self):
        """
        This function checks whether Add has compatible attributes with the NPU
        """
        if not super().is_valid():
            return False
        if not check_valid_dtypes(
            [self.ifm, self.ifm2, self.ofm], supported_dtypes=[np.uint8, np.int8, np.int32]
        ):
            return False
        return True


def qnn_add_pattern() -> tvm.relay.dataflow_pattern.DFPattern:
    """
    This function creates the pattern for qnn.add with optional fused RELU activation.
    """
    pattern = is_op("qnn.add")(
        wildcard(),
        wildcard(),
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
    )
    pattern = pattern.optional(is_op("clip"))
    return pattern


class SubParams(BinaryElementwiseParams):
    """
    This class will parse a call to a ethosu.binary_elementwise Sub composite function
    and extract the parameter information.
    """

    composite_name = "ethos-u.sub"

    def __init__(self, func_body: Call):
        BinaryElementwiseParams.__init__(self, func_body, "SUB", True)

    def is_valid(self):
        """
        This function checks whether Sub has compatible attributes with the NPU
        """
        if not super().is_valid():
            return False
        if not check_valid_dtypes(
            [self.ifm, self.ifm2, self.ofm], supported_dtypes=[np.uint8, np.int8, np.int32]
        ):
            return False
        return True


def qnn_subtract_pattern() -> tvm.relay.dataflow_pattern.DFPattern:
    """
    This function creates the pattern for qnn.subtract with optional fused RELU activation.
    """
    pattern = is_op("qnn.subtract")(
        wildcard(),
        wildcard(),
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
    )
    pattern = pattern.optional(is_op("clip"))
    return pattern


class MulParams(BinaryElementwiseParams):
    """
    This class will parse a call to a ethosu.binary_elementwise Mul composite function
    and extract the parameter information.
    """

    composite_name = "ethos-u.mul"

    def __init__(self, func_body: Call):
        BinaryElementwiseParams.__init__(self, func_body, "MUL", True)

    def is_valid(self):
        """
        This function checks whether Mul has compatible attributes with the NPU
        """
        if not super().is_valid():
            return False
        if not check_valid_dtypes(
            [self.ifm, self.ifm2, self.ofm], supported_dtypes=[np.uint8, np.int8, np.int32]
        ):
            return False
        return True


def qnn_mul_pattern() -> tvm.relay.dataflow_pattern.DFPattern:
    """
    This function creates the pattern for qnn.mul with optional fused RELU activation.
    """
    pattern = is_op("qnn.mul")(
        wildcard(),
        wildcard(),
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
    )
    pattern = pattern.optional(is_op("clip"))
    return pattern


class MinParams(BinaryElementwiseParams):
    """
    This class will parse a call to a ethosu.binary_elementwise Min composite function
    and extract the parameter information.
    """

    composite_name = "ethos-u.min"

    def __init__(self, func_body: Call):
        BinaryElementwiseParams.__init__(self, func_body, "MIN", False)

    def is_valid(self):
        """
        This function checks whether Min has compatible attributes with the NPU
        """
        if not super().is_valid():
            return False
        if self.ifm.dtype != self.ifm2.dtype:
            return False
        if not check_valid_dtypes(
            [self.ifm, self.ifm2, self.ofm], supported_dtypes=[np.uint8, np.int8]
        ):
            return False
        # MIN with different scales is not supported on NPU
        # (please look at NPU_SET_OFM_SCALE register description
        # https://developer.arm.com/documentation/102420/0200/Programmers-model/Command-stream/cmd1-commands-).
        if self.ifm.q_params.scale_f32 != self.ofm.q_params.scale_f32:
            return False
        return True


# This pattern is for case when there are different scales for requantize and
# minimum + clip + qnn.requantize can't be offloaded to NPU by one operation
# due to hardware constraints.
# It's offloaded by two operations ethosu_binary_elementwise + ethosu_identity.
def minimum_pattern() -> tvm.relay.dataflow_pattern.DFPattern:
    """
    This function creates the pattern for minimum with optional fused RELU activation without
    requantize.
    """
    minimum = is_op("minimum")(wildcard(), wildcard())
    optional_min_clip = is_op("clip")(minimum)
    return minimum | optional_min_clip


def minimum_clip_requantize_pattern() -> tvm.relay.dataflow_pattern.DFPattern:
    """
    This function creates the pattern for minimum with fused RELU activation with requantize.
    """
    pattern = is_op("minimum")(wildcard(), wildcard())
    pattern = is_op("clip")(pattern)
    pattern = is_op("qnn.requantize")(
        pattern, is_constant(), is_constant(), is_constant(), is_constant()
    )
    return pattern


class MaxParams(BinaryElementwiseParams):
    """
    This class will parse a call to a ethosu.binary_elementwise Max composite function
    and extract the parameter information.
    """

    composite_name = "ethos-u.max"

    def __init__(self, func_body: Call):
        BinaryElementwiseParams.__init__(self, func_body, "MAX", False)

    def is_valid(self):
        """
        This function checks whether Max has compatible attributes with the NPU
        """
        if not super().is_valid():
            return False
        if self.ifm.dtype != self.ifm2.dtype:
            return False
        if not check_valid_dtypes(
            [self.ifm, self.ifm2, self.ofm], supported_dtypes=[np.uint8, np.int8]
        ):
            return False
        # MAX with different scales is not supported on NPU
        # (please look at NPU_SET_OFM_SCALE register description
        # https://developer.arm.com/documentation/102420/0200/Programmers-model/Command-stream/cmd1-commands-).
        if self.ifm.q_params.scale_f32 != self.ofm.q_params.scale_f32:
            return False
        return True


# This pattern is for case when there are different scales for requantize and
# maximum + clip + qnn.requantize can't be offloaded to NPU by one operation due to
# hardware constraints.
# It's offloaded by two operations ethosu_binary_elementwise + ethosu_identity.
def maximum_pattern() -> tvm.relay.dataflow_pattern.DFPattern:
    """
    This function creates the pattern for maximum with optional fused RELU activation without
    requantize.
    """
    maximum = is_op("maximum")(wildcard(), wildcard())
    optional_max_clip = is_op("clip")(maximum)
    return maximum | optional_max_clip


def maximum_clip_requantize_pattern() -> tvm.relay.dataflow_pattern.DFPattern:
    """
    This function creates the pattern for maximum with fused RELU activation with requantize.
    """
    pattern = is_op("maximum")(wildcard(), wildcard())
    pattern = is_op("clip")(pattern)
    pattern = is_op("qnn.requantize")(
        pattern, is_constant(), is_constant(), is_constant(), is_constant()
    )
    return pattern


class ShlParams(BinaryElementwiseParams):
    """
    This class will parse a call to a ethosu.binary_elementwise Shl composite function
    and extract the parameter information.
    """

    composite_name = "ethos-u.shl"

    def __init__(self, func_body: Call):
        BinaryElementwiseParams.__init__(self, func_body, "SHL", False)

    def is_valid(self):
        """
        This function checks whether Shl has compatible attributes with the NPU
        """
        if not super().is_valid():
            return False
        if not check_valid_dtypes([self.ifm, self.ifm2, self.ofm], supported_dtypes=[np.int32]):
            return False
        return True


def shl_pattern() -> tvm.relay.dataflow_pattern.DFPattern:
    """
    This function creates the pattern for left_shift with optional fused RELU activation.
    """
    pattern = is_op("left_shift")(wildcard(), wildcard())
    pattern = pattern.optional(is_op("clip"))
    return pattern


class ReshapeParams:
    """
    This class will parse a call to a ethosu.reshape composite function
    and extract the parameter information.
    """

    composite_name = "ethos-u.reshape"

    def __init__(self, func_body: Call):
        self.new_shape = func_body.attrs.newshape
        self.ifm = TensorParams(func_body.args[0])
        self.ofm = TensorParams(func_body)

    def is_valid(self):
        """
        This function checks whether reshape has compatible attributes with the NPU
        """
        if not check_dimensions(self.ifm) or not check_dimensions(self.ofm):
            return False
        if not check_valid_dtypes([self.ifm, self.ofm], supported_dtypes=[np.int8]):
            return False
        return True


def reshape_pattern():
    """Create pattern for reshape"""
    pattern = is_op("reshape")(wildcard())
    return pattern


class StridedSliceParams:
    """
    This class will parse a call to a ethosu.strided_slice composite function
    and extract the parameter information.
    """

    composite_name = "ethos-u.strided_slice"

    def __init__(self, func_body: Call):
        self.ifm = TensorParams(func_body.args[0])
        self.ofm = TensorParams(func_body)

        attrs = func_body.attrs
        # The indices where we begin the slice
        self.begin = attrs.begin
        # The indices where we end the slice
        self.end = attrs.end
        self.strides = attrs.strides
        self.axes = attrs.axes
        self.slice_mode = attrs.slice_mode

    def is_valid(self):
        """
        This function checks whether reshape has compatible attributes with the NPU
        """
        if not check_dimensions(self.ifm) or not check_dimensions(self.ofm):
            return False
        if not check_valid_dtypes([self.ifm, self.ofm], supported_dtypes=[np.int8]):
            return False
        if len(self.begin) != len(self.end):
            return False

        for begin_idx, end_idx in zip(self.begin, self.end):
            if begin_idx > end_idx:
                return False

        # Only strides of 1 are supported
        if self.strides:
            if not all([i == 1 for i in self.strides]):
                return False
        return True


def strided_slice_pattern():
    """Create pattern for strided_slice"""
    pattern = is_op("strided_slice")(wildcard())
    return pattern


class AbsParams:
    """
    This class will parse a call to a ethosu.unary_elementwise Abs composite function
    and extract the parameter information.
    """

    composite_name = "ethos-u.abs"

    def __init__(self, func_body: Call):
        from tvm.relay.backend.contrib.ethosu.util import DequantizeArgs, QuantizeArgs

        quantize = func_body
        abs_op = quantize.args[0]
        dequantize = abs_op.args[0]

        layout = "NHWC"

        self.ifm = TensorParams(
            dequantize.args[DequantizeArgs.IFM.value],
            layout,
            dequantize.args[DequantizeArgs.IFM_SCALE.value],
            dequantize.args[DequantizeArgs.IFM_ZERO_POINT.value],
        )
        self.ofm = TensorParams(
            quantize,
            layout,
            quantize.args[QuantizeArgs.OFM_SCALE.value],
            quantize.args[QuantizeArgs.OFM_ZERO_POINT.value],
        )

        self.operator_type = "ABS"
        self.activation = None

    def is_valid(self):
        """Checks whether Abs has compatible attributes with HW"""
        tensor_params = [self.ifm, self.ofm]
        if not check_valid_dtypes(tensor_params, supported_dtypes=[np.int8, np.uint8]):
            return False
        if self.ifm.dtype != self.ofm.dtype:
            return False
        if not check_dimensions(self.ifm):
            return False
        if len(self.ifm.shape) == 4 and self.ifm.shape[0] != 1:
            return False
        if self.ifm.shape != self.ofm.shape:
            return False
        return True


def abs_pattern() -> tvm.relay.dataflow_pattern.DFPattern:
    """Create pattern for abs"""
    pattern = is_op("qnn.dequantize")(wildcard(), is_constant(), is_constant())
    pattern = is_op("abs")(pattern)
    pattern = is_op("qnn.quantize")(pattern, is_constant(), is_constant())
    return pattern


class LutActivationParams:
    """
    A parent class for LUT based activation functions that extract the input and
    output tensors and check whether they are valid.
    """

    def __init__(self, func_body: Call):
        from tvm.relay.backend.contrib.ethosu.util import DequantizeArgs, QuantizeArgs

        layout = "NHWC"

        quantize = func_body
        activation = quantize.args[0]
        dequantize = activation.args[0]
        in_var = dequantize.args[0]

        self.ifm = TensorParams(
            in_var,
            layout=layout,
            scale=dequantize.args[DequantizeArgs.IFM_SCALE.value],
            zero_point=dequantize.args[DequantizeArgs.IFM_ZERO_POINT.value],
        )
        self.ofm = TensorParams(
            quantize,
            layout=layout,
            scale=quantize.args[QuantizeArgs.OFM_SCALE.value],
            zero_point=quantize.args[QuantizeArgs.OFM_ZERO_POINT.value],
        )

    def is_valid(self):
        """
        This function checks whether activation has compatible attributes with the NPU
        """
        if not check_valid_dtypes([self.ifm, self.ofm], supported_dtypes=[np.int8]):
            return False
        return True


class TanhParams(LutActivationParams):

    composite_name = "ethos-u.tanh"


def tanh_pattern():
    """Create pattern for tanh"""
    dequant = is_op("qnn.dequantize")(wildcard(), is_constant(), is_constant())
    tanh = is_op("tanh")(dequant)
    quant = is_op("qnn.quantize")(tanh, is_constant(), is_constant())
    return quant


class SigmoidParams(LutActivationParams):
    """
    This class will parse a call to a ethos-u.sigmoid composite function
    and extract the parameter information.
    """

    composite_name = "ethos-u.sigmoid"


def sigmoid_pattern():
    """Create pattern for sigmoid"""
    dequant = is_op("qnn.dequantize")(wildcard(), is_constant(), is_constant())
    sigmoid = is_op("sigmoid")(dequant)
    quant = is_op("qnn.quantize")(sigmoid, is_constant(), is_constant())
    return quant


class LeakyReLUParams(LutActivationParams):
    """
    This class will parse a call to ethos-u.leaky_relu composite function
    and extract the parameter information.
    """

    composite_name = "ethos-u.leaky_relu"

    def __init__(self, func_body: Call):
        super().__init__(func_body)
        self.alpha = func_body.args[0].attrs.alpha


def leaky_relu_pattern() -> tvm.relay.dataflow_pattern.DFPattern:
    """
    This function creates the pattern for leaky relu.
    """
    dequantize = is_op("qnn.dequantize")(wildcard(), is_constant(), is_constant())
    leaky_relu = is_op("nn.leaky_relu")(dequantize)
    return is_op("qnn.quantize")(leaky_relu, is_constant(), is_constant())


class MeanParams:
    """
    This class will parse a call to ethosu.mean composite function
    and extract the parameter information.
    """

    composite_name = "ethos-u.mean"

    def __init__(self, func_body: Call):
        from tvm.relay.backend.contrib.ethosu.util import RequantArgs

        requantize = func_body
        mean_op = requantize.args[0]
        attrs = mean_op.attrs
        cast = mean_op.args[0]

        layout = "NHWC"
        self.ifm = TensorParams(
            cast.args[0],
            layout,
            requantize.args[RequantArgs.IFM_SCALE.value],
            requantize.args[RequantArgs.IFM_ZERO_POINT.value],
        )
        self.ofm = TensorParams(
            requantize,
            layout,
            requantize.args[RequantArgs.OFM_SCALE.value],
            requantize.args[RequantArgs.OFM_ZERO_POINT.value],
        )

        ifm_shape = self.ifm.shape
        self.height = ifm_shape[0] if len(ifm_shape) in (2, 3) else ifm_shape[1]
        self.width = ifm_shape[1] if len(ifm_shape) in (2, 3) else ifm_shape[2]
        self.keepdims = attrs.keepdims

        self.axis = list(sorted(attrs.axis))
        if attrs.exclude:
            self.axis = [i for i in range(len(self.ifm.shape)) if i not in self.axis]

    def is_valid(self) -> bool:
        """
        Checks whether Mean has compatible attributes with HW.
        """

        def check_axis(num_dims, axis):
            if num_dims in (2, 3):
                return axis in ([0], [1], [0, 1])
            return axis in ([1], [2], [1, 2])

        def check_single_axis_across_height(num_dims, axis):
            return len(axis) == 1 and (num_dims in (2, 3) and axis == [0] or axis == [1])

        same_quantization = (
            self.ifm.q_params.scale_f32 == self.ofm.q_params.scale_f32
            and self.ifm.q_params.zero_point == self.ofm.q_params.zero_point
        )

        # IFM must be int8 or uint8
        if not check_valid_dtypes([self.ifm], [np.int8, np.uint8]):
            return False
        # OFM must be int8, uint8 or int16
        if not check_valid_dtypes([self.ofm], [np.int8, np.uint8, np.int16]):
            return False
        # Input tensor must be at least 2D
        if not len(self.ifm.shape) in [2, 3, 4]:
            return False
        # Axis indices must correspond to height and width axes
        if not check_axis(len(self.ifm.shape), self.axis):
            return False

        input_size = self.height * self.width

        # Product of height and width must be no greater than 65536
        if input_size > 65536:
            return False
        # Product of height and width must be no greater than 4096 when:
        #   IFM and OFM have different scale or zero point; or
        #   'keep_dims' is True
        if input_size > 4096 and (not same_quantization or self.keepdims):
            return False
        # For single axis averages across the height dimension:
        if check_single_axis_across_height(len(self.ifm.shape), self.axis):
            # IFM height must be no greater than 256 if the IFM and OFM scale and zero point match
            if self.height > 256 and same_quantization:
                return False
            # IFM height must be no greater than 64 if the IFM and OFM scale or zero point
            # do not match
            if self.height > 64 and not same_quantization:
                return False
        return True


def mean_pattern() -> tvm.relay.dataflow_pattern.DFPattern:
    """
    This function creates the pattern for mean.
    """
    pattern = is_op("cast")(wildcard())
    pattern = is_op("mean")(pattern)
    pattern = is_op("qnn.requantize")(
        pattern, is_constant(), is_constant(), is_constant(), is_constant()
    )
    return pattern


class SumParams:
    """
    This class will parse a call to ethosu.sum composite function
    and extract the parameter information.
    """

    composite_name = "ethos-u.sum"

    def __init__(self, func_body: Call):
        from tvm.relay.backend.contrib.ethosu.util import RequantArgs

        clip = None
        if str(func_body.op.name) == "clip":
            clip = func_body
            requantize = clip.args[0]
        else:
            requantize = func_body

        sum_op = requantize.args[0]
        attrs = sum_op.attrs
        cast = sum_op.args[0]

        layout = "NHWC"
        self.ifm = TensorParams(
            cast.args[0],
            layout,
            requantize.args[RequantArgs.IFM_SCALE.value],
            requantize.args[RequantArgs.IFM_ZERO_POINT.value],
        )
        self.ofm = TensorParams(
            requantize,
            layout,
            requantize.args[RequantArgs.OFM_SCALE.value],
            requantize.args[RequantArgs.OFM_ZERO_POINT.value],
        )

        self.activation = clip

        ifm_shape = self.ifm.shape
        self.height = ifm_shape[0] if len(ifm_shape) in (2, 3) else ifm_shape[1]
        self.width = ifm_shape[1] if len(ifm_shape) in (2, 3) else ifm_shape[2]
        self.keepdims = attrs.keepdims

        self.axis = list(sorted(attrs.axis))
        if attrs.exclude:
            self.axis = [i for i in range(len(self.ifm.shape)) if i not in self.axis]

    def is_valid(self) -> bool:
        """
        Checks whether Sum has compatible attributes with HW.
        """

        ifm_shape_len = len(self.ifm.shape)

        if not check_valid_dtypes([self.ifm], [np.uint8, np.int8, np.int16, np.int32]):
            return False
        if not check_valid_dtypes([self.ofm], [np.int8]):
            return False
        if not ifm_shape_len in (3, 4):
            return False
        if ifm_shape_len == 3 and self.axis not in [[2]]:
            return False
        if ifm_shape_len == 4 and self.axis not in [[3]]:
            return False

        return True


def sum_pattern() -> tvm.relay.dataflow_pattern.DFPattern:
    """
    This function creates the pattern for sum.
    """
    pattern = is_op("cast")(wildcard())
    pattern = is_op("sum")(pattern)
    pattern = is_op("qnn.requantize")(
        pattern,
        is_constant(),
        is_constant(),
        is_constant(),
        is_constant(),
    )
    pattern = pattern.optional(is_op("clip"))
    return pattern


class ConcatParams:
    """
    This class will parse a call to a ethos-u.concat composite function
    and extract the parameter information.
    """

    composite_name = "ethos-u.concat"

    def __init__(self, func_body):
        self.concat = func_body
        self.is_qnn_variant = self.concat.op.name == "qnn.concatenate"
        self.input_tensors = [TensorParams(tensor) for tensor in list(func_body.args[0])]
        self.axis = func_body.attrs.axis

        if self.is_qnn_variant:
            self.input_scales = [s.data.asnumpy() for s in list(func_body.args[1])]
            self.input_zero_points = [zp.data.asnumpy() for zp in list(func_body.args[2])]

    def is_valid(self):
        """Checks whether Concatenate has compatible attributes with the hardware"""
        if not check_valid_dtypes(self.input_tensors, supported_dtypes=[np.int8]):
            return False
        # Check that the scales and zero points of input tensors are the same
        if self.is_qnn_variant and not all(self.input_scales == self.input_scales[0]):
            return False
        if self.is_qnn_variant and not all(self.input_zero_points == self.input_zero_points[0]):
            return False

        input_dim = len(self.input_tensors[0].shape)
        for tensor in self.input_tensors:
            if len(tensor.shape) != input_dim:
                return False

        if self.axis is None:
            return False
        if self.axis < 0:
            return False
        if self.axis >= input_dim:
            return False

        output_shape = self.concat.checked_type.shape
        if len(output_shape) != input_dim:
            return False
        if len(output_shape) > 3 and output_shape[0] != 1:
            return False
        return True


def concat_pattern():
    """Create pattern for concat"""
    tensors = is_tuple(None)
    scales = is_tuple(None)
    zero_points = is_tuple(None)
    qnn_concat = is_op("qnn.concatenate")(
        tensors, scales, zero_points, is_constant(), is_constant()
    )
    concat = is_op("concatenate")(tensors)
    return concat | qnn_concat


class SplitParams:
    """
    This class will parse a call to a ethos-u.split composite function
    and extract the parameter information.
    """

    composite_name = "ethos-u.split"

    def __init__(self, func_body):
        self.split = func_body
        self.input = TensorParams(func_body.args[0])
        self.axis = func_body.attrs.axis
        self.indices_or_sections = self.convert_indices_or_sections(
            func_body.attrs.indices_or_sections
        )

    def convert_indices_or_sections(self, indices_or_sections):
        # split_v
        if isinstance(indices_or_sections, tvm.ir.container.Array):
            values = [i.value for i in indices_or_sections]
        # split
        else:
            values = indices_or_sections.value
        return values

    def is_valid(self):
        """Checks whether split has compatible attributes with the hardware"""
        if not check_valid_dtypes([self.input], supported_dtypes=[np.int8]):
            return False
        return True


def split_pattern():
    "Create the pattern for split"
    split = is_op("split")(wildcard())
    return split


class RequantizeParams:
    """
    This class will parse a call to ethos-u.requantize composite function
    and extract the parameter information.
    """

    composite_name = "ethos-u.requantize"

    def __init__(self, func_body: Call):
        from tvm.relay.backend.contrib.ethosu.util import RequantArgs

        layout = "NHWC"
        in_var = func_body.args[0]
        requantize = func_body

        self.ifm = TensorParams(
            in_var,
            layout=layout,
            scale=requantize.args[RequantArgs.IFM_SCALE.value],
            zero_point=requantize.args[RequantArgs.IFM_ZERO_POINT.value],
        )
        self.ofm = TensorParams(
            requantize,
            layout=layout,
            scale=requantize.args[RequantArgs.OFM_SCALE.value],
            zero_point=requantize.args[RequantArgs.OFM_ZERO_POINT.value],
        )

        attrs = requantize.attrs
        self.out_dtype = attrs.out_dtype

    def is_valid(self) -> bool:
        """
        Checks whether qnn.requantize has compatible attributes with HW.
        """
        tensor_params = [self.ifm, self.ofm]
        if not check_valid_dtypes(tensor_params, supported_dtypes=[np.int8]):
            return False
        if not check_dimensions(self.ifm) or not check_dimensions(self.ofm):
            return False
        if self.out_dtype and self.out_dtype != "int8":
            return False
        return True


def requantize_pattern() -> tvm.relay.dataflow_pattern.DFPattern:
    """
    This function creates the pattern for qnn.requantize.
    """
    return is_op("qnn.requantize")(
        wildcard(), is_constant(), is_constant(), is_constant(), is_constant()
    )


class Resize2dParams:
    """
    This class will parse a call to ethos-u.resize2d composite function
    and extract the parameter information.
    """

    composite_name = "ethos-u.resize2d"

    def __init__(self, func_body: Call):
        layout = "NHWC"

        resize_2d = func_body
        in_var = func_body.args[0]
        if (
            isinstance(resize_2d, tvm.relay.expr.Call)
            and isinstance(resize_2d.op, tvm.ir.Op)
            and resize_2d.op.name == "qnn.quantize"
        ):
            resize_2d = resize_2d.args[0]
            in_var = in_var.args[0].args[0]
        out_var = func_body

        self.ifm = TensorParams(in_var, layout=layout)
        self.ofm = TensorParams(out_var, layout=layout)

        attrs = resize_2d.attrs
        self.size = attrs.size
        self.method = attrs.method
        self.roi = attrs.roi
        self.coordinate_transformation_mode = attrs.coordinate_transformation_mode
        self.rounding_method = attrs.rounding_method
        self.out_dtype = attrs.out_dtype

    def is_valid(self) -> bool:
        """
        Checks whether image.resize2d has compatible attributes with HW.
        """

        def check_compatible_size(mode, method, upscale_size, ifm_size):
            """Checking the provided upscale_size is compatible with the NPU. The NPU only
            supports upsampling when the upsampling size is 2 * input_size, or when there is
            no upsampling to be done, so check that this is the case. In the special case of
            resize_bilinear with align_corners=True, the NPU only supports an upsampling
            size of 2 * input_size - 1."""
            delta = 1 if mode == "align_corners" and method == "linear" else 0
            upscale_size = np.array(upscale_size)
            ifm_size = np.array(ifm_size)
            ifm_upscaled = ifm_size * 2 - delta
            return (ifm_upscaled == upscale_size).all() or (ifm_size == upscale_size).all()

        tensor_params = [self.ifm, self.ofm]
        if not check_valid_dtypes(tensor_params, supported_dtypes=[np.int8]):
            return False
        if len(self.ifm.shape) != 4 or len(self.ofm.shape) != 4:
            return False
        if list(float(x) for x in self.roi) != [0.0] * 4:
            return False
        if self.method not in ("nearest_neighbor", "linear"):
            return False
        if self.coordinate_transformation_mode not in (
            "asymmetric",
            "align_corners",
            "half_pixel",
        ):
            return False
        if (
            self.coordinate_transformation_mode == "half_pixel"
            and self.rounding_method != "round_prefer_ceil"
            or self.coordinate_transformation_mode != "half_pixel"
            and self.rounding_method != ""
        ):
            return False
        if not check_compatible_size(
            self.coordinate_transformation_mode,
            self.method,
            self.size,
            self.ifm.shape[1:3],
        ):
            return False
        if self.out_dtype and self.out_dtype != "int8":
            return False
        return True


def resize2d_pattern() -> tvm.relay.dataflow_pattern.DFPattern:
    """
    This function creates the pattern for image.resize2d.
    """
    dequant = is_op("qnn.dequantize")(wildcard(), is_constant(), is_constant())
    resize_2d = is_op("image.resize2d")(dequant).has_attr({"method": "linear"})
    quant = is_op("qnn.quantize")(resize_2d, is_constant(), is_constant())
    return quant | is_op("image.resize2d")(wildcard()).has_attr({"method": "nearest_neighbor"})


class ExpandDimsParams:
    """
    This class will parse a call to a ethos-u.expand_dims composite function
    and extract the parameter information.
    """

    composite_name = "ethos-u.expand_dims"

    def __init__(self, func_body):
        self.expand_dims = func_body
        self.input = TensorParams(func_body.args[0])
        self.output = TensorParams(func_body)

    def is_valid(self):
        """Checks whether expand_dims has compatible attributes with the hardware."""
        if not check_dimensions(self.input) or not check_dimensions(self.output):
            return False
        if not check_valid_dtypes([self.input, self.output], supported_dtypes=[np.int8]):
            return False
        return True


def expand_dims_pattern():
    """Create the pattern for expand_dims."""
    return is_op("expand_dims")(wildcard())


class SqueezeParams:
    """
    This class will parse a call to a ethos-u.squeeze composite function
    and extract the parameter information.
    """

    composite_name = "ethos-u.squeeze"

    def __init__(self, func_body):
        self.squeeze = func_body
        self.input = TensorParams(func_body.args[0])
        self.output = TensorParams(func_body)

    def is_valid(self):
        """Checks whether squeeze has compatible attributes with the hardware."""
        if not check_dimensions(self.output):
            return False
        if not check_valid_dtypes([self.input, self.output], supported_dtypes=[np.int8]):
            return False
        return True


def squeeze_pattern():
    """Create the pattern for squeeze."""
    return is_op("squeeze")(wildcard())


class FullyConnectedParams:
    """
    This class will parse a call to an ethos-u.fully_connected composite
    function and extract the parameter information.
    """

    composite_name = "ethos-u.fully_connected"

    @requires_vela
    def __init__(self, func_body):
        from tvm.relay.backend.contrib.ethosu.util import QDenseArgs  # type: ignore
        from tvm.relay.backend.contrib.ethosu.util import BiasAddArgs, RequantArgs

        self.activation = None
        if str(func_body.op.name) == "clip":
            self.activation = func_body
            requantize_op = self.activation.args[0]
        else:
            requantize_op = func_body

        call = requantize_op.args[0]
        if str(requantize_op.args[0].op.name) == "nn.bias_add":
            bias_add = call
            qnn_dense = call.args[0]
        else:
            bias_add = None
            qnn_dense = call

        # weights & biases are params as they should be constant
        self.weights = TensorParams(
            qnn_dense.args[QDenseArgs.WEIGHTS.value],
            None,
            qnn_dense.args[QDenseArgs.WEIGHTS_SCALE.value],
            qnn_dense.args[QDenseArgs.WEIGHTS_ZERO_POINT.value],
        )
        self.biases = (
            TensorParams(
                bias_add.args[BiasAddArgs.BIASES.value],
                None,
                requantize_op.args[RequantArgs.IFM_SCALE.value],
                requantize_op.args[RequantArgs.IFM_ZERO_POINT.value],
            )
            if bias_add
            else None
        )
        self.ifm = TensorParams(
            qnn_dense.args[QDenseArgs.IFM.value],
            None,
            qnn_dense.args[QDenseArgs.IFM_SCALE.value],
            qnn_dense.args[QDenseArgs.IFM_ZERO_POINT.value],
        )
        self.ofm = TensorParams(
            func_body,
            None,
            requantize_op.args[RequantArgs.OFM_SCALE.value],
            requantize_op.args[RequantArgs.OFM_ZERO_POINT.value],
        )

    def is_valid(self) -> bool:
        """
        Checks whether Fully Connected has compatible attributes with HW
        """

        def check_weights_fc(weights):
            """Checks whether weight tensor is compatible with HW"""
            weights_limit = 127 * 65536
            # A saturation upper bound check for accumulators
            weights.values = weights.values - weights.q_params.zero_point
            axis = 1
            sum_weights = np.amax(np.sum(np.absolute(weights.values), axis=axis))
            if sum_weights > weights_limit:
                return False
            return True

        if not check_valid_dtypes([self.ifm, self.ofm], supported_dtypes=[np.int8]):
            return False
        if not check_weights_fc(self.weights):
            return False
        if not check_bias(self.biases):
            return False
        if not check_batch_size(self.ifm):
            return False
        # Check input shape
        if not len(self.ifm.shape) == 2:
            return False
        # Check output shape
        if not len(self.ofm.shape) == 2:
            return False
        return True


def qnn_fc_pattern():
    dense = is_op("qnn.dense")(
        wildcard(), is_constant(), is_constant(), is_constant(), is_constant(), is_constant()
    )
    optional_bias_add = is_op("nn.bias_add")(dense, is_constant())
    req = is_op("qnn.requantize")(
        dense | optional_bias_add, is_constant(), is_constant(), is_constant(), is_constant()
    )
    optional_clip = req.optional(is_op("clip"))
    return optional_clip


class MatMulParams(FullyConnectedParams):
    """
    This class will parse a call to an ethos-u.matmul composite
    function and extract the parameter information.
    """

    composite_name = "ethos-u.matmul"

    @requires_vela
    def __init__(self, func_body):
        FullyConnectedParams.__init__(self, func_body)

    def is_valid(self) -> bool:
        """
        Checks whether matrix multiplication has compatible attributes with HW
        """

        if not check_valid_dtypes([self.ifm, self.ofm], supported_dtypes=[np.int8]):
            return False
        if not len(self.ifm.shape) == 2:
            return False
        if not len(self.ofm.shape) == 2:
            return False
        # The weights must be transposed
        if self.ifm.shape[1] != self.weights.shape[1]:
            return False
        return True


def matmul_pattern():
    dense = is_op("qnn.dense")(
        wildcard(), wildcard(), is_constant(), is_constant(), is_constant(), is_constant()
    )
    req = is_op("qnn.requantize")(dense, is_constant(), is_constant(), is_constant(), is_constant())
    optional_clip = req.optional(is_op("clip"))
    return optional_clip


class HardSwishParams:
    """
    This class will parse a call to a ethos-u.hard_swish composite function
    and extract the parameter information.
    """

    composite_name = "ethos-u.hard_swish"

    def __init__(self, func_body):
        from tvm.relay.backend.contrib.ethosu.util import DequantizeArgs, QuantizeArgs

        quantize = func_body
        divide = quantize.args[0]
        multiply = divide.args[0]
        clip = multiply.args[1]
        add = clip.args[0]
        dequantize = add.args[0]

        self.ifm = TensorParams(
            dequantize.args[0],
            scale=dequantize.args[DequantizeArgs.IFM_SCALE.value],
            zero_point=dequantize.args[DequantizeArgs.IFM_ZERO_POINT.value],
        )
        self.ofm = TensorParams(
            quantize,
            scale=quantize.args[QuantizeArgs.OFM_SCALE.value],
            zero_point=quantize.args[QuantizeArgs.OFM_ZERO_POINT.value],
        )

    def is_valid(self):
        tensor_params = [self.ifm, self.ofm]
        if not check_valid_dtypes(tensor_params, supported_dtypes=[np.int8]):
            return False
        return True


def hard_swish_pattern():
    """Create the pattern for hard swish."""
    dequantize = is_op("qnn.dequantize")(wildcard(), is_constant(), is_constant())
    add = is_op("add")(dequantize, is_constant())
    clip = is_op("clip")(add)
    multiply = is_op("multiply")(dequantize, clip)
    divide = is_op("divide")(multiply, is_constant())
    quantize = is_op("qnn.quantize")(divide, is_constant(), is_constant())
    return quantize


class PadParams:
    """
    This class will parse a call to a ethosu.pad2d composite function
    and extract the parameter information.
    """

    composite_name = "ethos-u.pad2d"
    # The ethos-u.pad2d composite function will be transformed to the
    # ethosu_depthwise_conv2d operator.
    # For the ethosu_depthwise_conv2d the hardware only supports padding
    # upto the numbers as follows, so we define such padding limits
    padding_bounds = [31, 31, 32, 32]

    def __init__(self, func_body: Call):
        from tvm.relay.backend.contrib.ethosu.util import QPadArgs

        # there is no 'layout' attribute in nn.pad
        layout = "NHWC"
        self.ifm = TensorParams(
            tensor=func_body.args[QPadArgs.IFM.value],
            layout=layout,
            scale=tvm.relay.Constant(tvm.nd.array(np.array(1.0, dtype="float32"))),
            zero_point=func_body.args[QPadArgs.IFM_ZERO_POINT.value],
        )

        self.padding = self.extract_padding(func_body)
        self.ofm = TensorParams(
            tensor=func_body,
            layout=layout,
            scale=tvm.relay.Constant(tvm.nd.array(np.array(1.0, dtype="float32"))),
            zero_point=func_body.args[QPadArgs.IFM_ZERO_POINT.value],
        )

    @staticmethod
    def extract_padding(
        padding: relay.Call,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Here we check whether a separate spatial-dimension padding operation can be
        rewritten as NPU depthwise convolution. If the padding specified by the
        separate nn.pad operation is not supported by NPU depthwise convolution,
        None will be returned. This will cause the nn.pad not to be offloaded to NPU.
        """
        pad_width = padding.attrs["pad_width"]
        if len(pad_width) != 4:
            return None
        if list(pad_width[0]) != [0, 0] or list(pad_width[3]) != [0, 0]:
            return None
        return [
            pad_width[1][0],
            pad_width[2][0],
            pad_width[1][1],
            pad_width[2][1],
        ]

    def is_valid(self):
        """
        This function checks whether pad has compatible attributes
        with the NPU depthwise convolution
        """
        tensor_params = [self.ifm, self.ofm]
        if not check_valid_dtypes(tensor_params, supported_dtypes=[np.uint8, np.int8]):
            return False
        if self.ifm.dtype != self.ofm.dtype:
            return False
        if not check_batch_size(self.ifm):
            return False
        if not self.padding or not check_padding(self.padding, self.padding_bounds):
            return False
        if not check_dimensions(self.ifm) or not check_dimensions(self.ofm):
            return False
        return True


class ChannelPadParams:
    """
    This class will parse a call to a ethos-u.channel-pad composite function
    and extract the parameter information.
    """

    composite_name = "ethos-u.channel-pad"
    # The ethos-u.channel-pad composite function will be transformed
    # to the Relay concatenate operation.

    def __init__(self, func_body: Call):
        from tvm.relay.backend.contrib.ethosu.util import QPadArgs

        # there is no 'layout' attribute in nn.pad
        layout = "NHWC"
        self.ifm = TensorParams(
            tensor=func_body.args[QPadArgs.IFM.value],
            layout=layout,
            scale=tvm.relay.Constant(tvm.nd.array(np.array(1.0, dtype="float32"))),
            zero_point=func_body.args[QPadArgs.IFM_ZERO_POINT.value],
        )

        self.ch_padding = self.extract_ch_padding(func_body)
        self.ofm = TensorParams(
            tensor=func_body,
            layout=layout,
            scale=tvm.relay.Constant(tvm.nd.array(np.array(1.0, dtype="float32"))),
            zero_point=func_body.args[QPadArgs.IFM_ZERO_POINT.value],
        )

    @staticmethod
    def extract_ch_padding(
        padding: relay.Call,
    ) -> Optional[Tuple[int, int]]:
        """
        Here we check whether a separate channel-dimension padding operation can be
        rewritten as Relay concatenate operation. If the padding specified by the
        separate nn.pad operation is not supported by NPU, None will be returned.
        This will cause the nn.pad not to be offloaded to NPU.
        """
        pad_width = padding.attrs["pad_width"]
        if len(pad_width) != 4:
            return None
        if (
            list(pad_width[0]) != [0, 0]
            or list(pad_width[1]) != [0, 0]
            or list(pad_width[2]) != [0, 0]
        ):
            return None
        return [
            pad_width[3][0],
            pad_width[3][1],
        ]

    def is_valid(self):
        """
        This function checks whether pad has compatible attributes
        with the Relay concatenate operation
        """
        tensor_params = [self.ifm, self.ofm]
        if not check_valid_dtypes(tensor_params, supported_dtypes=[np.uint8, np.int8]):
            return False
        if self.ifm.dtype != self.ofm.dtype:
            return False
        if not check_batch_size(self.ifm):
            return False
        if not self.ch_padding:
            return False
        if not check_dimensions(self.ifm) or not check_dimensions(self.ofm):
            return False
        return True


def pad_pattern():
    """Create pattern for pad"""
    pattern = is_op("nn.pad")(wildcard(), is_constant())
    return pattern


class SoftMaxParams:
    """
    This class will parse a call to a ethos-u.softmax composite function
    and extract the parameter information.
    """

    composite_name = "ethos-u.softmax"

    def __init__(self, func_body: Call):
        from tvm.relay.backend.contrib.ethosu.util import QuantizeArgs
        from tvm.relay.backend.contrib.ethosu.util import DequantizeArgs

        quantize = func_body
        softmax_op = quantize.args[0]
        dequantize = softmax_op.args[0]

        layout = "NHWC"

        self.ifm = TensorParams(
            dequantize.args[DequantizeArgs.IFM.value],
            layout,
            dequantize.args[DequantizeArgs.IFM_SCALE.value],
            dequantize.args[DequantizeArgs.IFM_ZERO_POINT.value],
        )
        self.ofm = TensorParams(
            quantize,
            layout,
            quantize.args[QuantizeArgs.OFM_SCALE.value],
            quantize.args[QuantizeArgs.OFM_ZERO_POINT.value],
        )

        self.operator_type = "SOFTMAX"

    def is_valid(self):
        """Checks whether Softmax has compatible attributes with HW"""
        tensor_params = [self.ifm, self.ofm]
        if not check_valid_dtypes(tensor_params, supported_dtypes=[np.int8]):
            return False
        if self.ifm.dtype != self.ofm.dtype:
            return False
        if not check_dimensions(self.ifm):
            return False
        if self.ifm.shape != self.ofm.shape:
            return False
        return True


def softmax_pattern() -> tvm.relay.dataflow_pattern.DFPattern:
    """
    This function creates the pattern for Softmax.
    """
    pattern = is_op("qnn.dequantize")(wildcard(), is_constant(), is_constant())
    pattern = is_op("nn.softmax")(pattern)
    pattern = is_op("qnn.quantize")(pattern, is_constant(), is_constant())
    return pattern


@register_pattern_table("ethos-u")
def pattern_table() -> List[Tuple[str, tvm.relay.dataflow_pattern.DFPattern, Callable]]:
    return [
        (
            ChannelPadParams.composite_name,
            pad_pattern(),
            lambda pat: ChannelPadParams(pat).is_valid(),
        ),
        (
            QnnConv2DParams.composite_name,
            qnn_conv2d_pattern(),
            lambda pat: QnnConv2DParams(pat).is_valid(),
        ),
        (
            QnnDepthwiseConv2DParams.composite_name,
            qnn_depthwise_conv2d_pattern(),
            lambda pat: QnnDepthwiseConv2DParams(pat).is_valid(),
        ),
        (
            QnnConv2DTransposeParams.composite_name,
            qnn_conv2d_transpose_pattern(),
            lambda pat: QnnConv2DTransposeParams(pat).is_valid(),
        ),
        (
            FullyConnectedParams.composite_name,
            qnn_fc_pattern(),
            lambda pat: FullyConnectedParams(pat).is_valid(),
        ),
        (
            MatMulParams.composite_name,
            matmul_pattern(),
            lambda pat: MatMulParams(pat).is_valid(),
        ),
        (
            MaxPool2DParams.composite_name,
            qnn_maxpool2d_pattern(),
            lambda pat: MaxPool2DParams(pat).is_valid(),
        ),
        (
            AvgPool2DParams.composite_name,
            qnn_avgpool2d_pattern(),
            lambda pat: AvgPool2DParams(pat).is_valid(),
        ),
        (
            PadParams.composite_name,
            pad_pattern(),
            lambda pat: PadParams(pat).is_valid(),
        ),
        (
            AddParams.composite_name,
            qnn_add_pattern(),
            lambda pat: AddParams(pat).is_valid(),
        ),
        (
            SubParams.composite_name,
            qnn_subtract_pattern(),
            lambda pat: SubParams(pat).is_valid(),
        ),
        (
            MulParams.composite_name,
            qnn_mul_pattern(),
            lambda pat: MulParams(pat).is_valid(),
        ),
        (
            MinParams.composite_name,
            minimum_clip_requantize_pattern(),
            lambda pat: MinParams(pat).is_valid(),
        ),
        (
            MinParams.composite_name,
            minimum_pattern(),
            lambda pat: MinParams(pat).is_valid(),
        ),
        (
            MaxParams.composite_name,
            maximum_clip_requantize_pattern(),
            lambda pat: MaxParams(pat).is_valid(),
        ),
        (
            MaxParams.composite_name,
            maximum_pattern(),
            lambda pat: MaxParams(pat).is_valid(),
        ),
        (
            ShlParams.composite_name,
            shl_pattern(),
            lambda pat: ShlParams(pat).is_valid(),
        ),
        (
            ReshapeParams.composite_name,
            reshape_pattern(),
            lambda pat: ReshapeParams(pat).is_valid(),
        ),
        (
            StridedSliceParams.composite_name,
            strided_slice_pattern(),
            lambda pat: StridedSliceParams(pat).is_valid(),
        ),
        (
            AbsParams.composite_name,
            abs_pattern(),
            lambda pat: AbsParams(pat).is_valid(),
        ),
        (TanhParams.composite_name, tanh_pattern(), lambda pat: TanhParams(pat).is_valid()),
        (
            MeanParams.composite_name,
            mean_pattern(),
            lambda pat: MeanParams(pat).is_valid(),
        ),
        (
            SumParams.composite_name,
            sum_pattern(),
            lambda pat: SumParams(pat).is_valid(),
        ),
        (
            SoftMaxParams.composite_name,
            softmax_pattern(),
            lambda pat: SoftMaxParams(pat).is_valid(),
        ),
        (
            LeakyReLUParams.composite_name,
            leaky_relu_pattern(),
            lambda pat: LeakyReLUParams(pat).is_valid(),
        ),
        (ConcatParams.composite_name, concat_pattern(), lambda pat: ConcatParams(pat).is_valid()),
        (
            SigmoidParams.composite_name,
            sigmoid_pattern(),
            lambda pat: SigmoidParams(pat).is_valid(),
        ),
        (
            SplitParams.composite_name,
            split_pattern(),
            lambda pat: SplitParams(pat).is_valid(),
        ),
        (
            RequantizeParams.composite_name,
            requantize_pattern(),
            lambda pat: RequantizeParams(pat).is_valid(),
        ),
        (
            Resize2dParams.composite_name,
            resize2d_pattern(),
            lambda pat: Resize2dParams(pat).is_valid(),
        ),
        (
            ExpandDimsParams.composite_name,
            expand_dims_pattern(),
            lambda pat: ExpandDimsParams(pat).is_valid(),
        ),
        (
            SqueezeParams.composite_name,
            squeeze_pattern(),
            lambda pat: SqueezeParams(pat).is_valid(),
        ),
        (
            HardSwishParams.composite_name,
            hard_swish_pattern(),
            lambda pat: HardSwishParams(pat).is_valid(),
        ),
    ]


# pylint: disable=unused-argument
@requires_vela
def partition_for_ethosu(
    mod: tvm.ir.IRModule,
    params: Optional[Dict[str, tvm.runtime.NDArray]] = None,
    mod_name: str = "default",
    **opts,
):
    """This helper function partition the relay graph as produced by the
    relay frontend for a given model into external functions
    to be presented to the codegen.

    Parameters
    ----------
    mod : tvm.ir.IRModule
        The IRModule that gets generated from a relay frontend
    params : Optional[Dict[str, tvm.runtime.NDArray]]
        Constant input parameters.
    mod_name: str, optional
        The module name

    Returns
    -------
    mod : IRModule
        The partitioned IRModule with external global functions
    """
    from tvm.relay.backend.contrib.ethosu import preprocess, codegen

    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)

    pattern = relay.op.contrib.get_pattern_table("ethos-u")
    mod = relay.transform.InferType()(mod)
    mod = codegen.replicate_pads(mod)
    mod = relay.transform.InferType()(mod)
    mod = relay.transform.MergeComposite(pattern)(mod)
    mod = relay.transform.AnnotateTarget("ethos-u")(mod)
    mod = relay.transform.MergeCompilerRegions()(mod)
    mod = relay.transform.InferType()(mod)
    mod = relay.transform.PartitionGraph(mod_name)(mod)
    mod = relay.transform.InferType()(mod)
    mod = preprocess.preprocess_ext_io()(mod)
    return mod
