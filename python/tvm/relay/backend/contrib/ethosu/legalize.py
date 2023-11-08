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
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel
# pylint: disable=no-value-for-parameter, use-list-literal
"""A set of passes to legalize some of operations for the NPU"""
from typing import List, Type, Callable
import math

import numpy as np  # type: ignore
from ethosu.vela import scaling, fp_math

import tvm  # type: ignore
from tvm import relay
from tvm.relay.dataflow_pattern import DFPatternCallback  # type: ignore
from tvm.relay.dataflow_pattern import wildcard
from tvm.relay.dataflow_pattern import is_op
from tvm.relay.dataflow_pattern import rewrite
from tvm.relay.dataflow_pattern import CallPattern
from tvm.relay.backend.contrib.ethosu import op as ethosu_ops  # type: ignore
from tvm.relay.backend.contrib.ethosu import vela_api
from tvm.relay.backend.contrib.ethosu import util
from tvm.relay.backend.contrib.ethosu.softmax_rewriter import SoftmaxRewriter
from tvm.relay.op.contrib import ethosu as ethosu_patterns  # type: ignore


class SplitRewriter(DFPatternCallback):
    """This rewriting converts split operations into a sequence of
    strided_slice operations, because codegen is going to be based
    on strided_slices that will define the slice of the tensor that
    will be fed to the consumer.
    """

    def __init__(self):
        super().__init__(require_type=True)
        self.split_in = wildcard()
        self.pattern = is_op("split")(self.split_in)

    @staticmethod
    def get_section_begin_coords(split: tvm.relay.Expr) -> List[int]:
        """Currently, the split operator takes an array of indices or an integer
        indicating the number of splits. However, its an array of indices could
        represent both cases, therefore this function just make it an array of
        indices where each index represent the co-ordinate of beginning of each
        section -- defines as section begins.

        Parameters
        ----------
        split : tvm.relay.Expr
            The Relay Call expression for a split operator

        Returns
        -------
        section_begins : List[int]
            A list containing integers corresponding to section
            begins
        """
        indices_or_sections = split.attrs.indices_or_sections
        input_shape = split.args[0].checked_type.shape
        split_axis = split.attrs.axis

        if isinstance(indices_or_sections, tvm.ir.container.Array):
            # 0 is the beginning of the first section.
            return [0] + list(indices_or_sections)
        split_axis_len = input_shape[split_axis].value
        section_length = split_axis_len // indices_or_sections.value
        return list(range(0, split_axis_len, section_length))

    def callback(
        self, pre: tvm.relay.Expr, post: tvm.relay.Expr, node_map: tvm.ir.container.Map
    ) -> tvm.relay.Expr:
        split_input = post.args[0]
        split_begins = list()
        split_ends = list()
        section_begins_in_split_axis = self.get_section_begin_coords(post)
        for split_cord in section_begins_in_split_axis:
            # first begin is [0, 0, ... , 0]
            begin_shape = [0 for i in range(len(split_input.checked_type.shape))]
            begin_shape[post.attrs.axis] = split_cord
            split_begins.append(begin_shape)

            end_shape = list(split_input.checked_type.shape)
            # Only the split axis coordinate changes
            end_shape[post.attrs.axis] = split_cord
            split_ends.append(end_shape)

        # Coordinates needs to be shifted left because beginning
        # of the next section is the end of the previous
        split_ends = split_ends[1:]
        # Last section end is the shape of the tensor itself.
        split_ends.append(list(split_input.checked_type.shape))

        strided_slices = list()
        for sb, se in zip(split_begins, split_ends):
            strided_slices.append(relay.strided_slice(split_input, sb, se))

        return relay.Tuple(strided_slices)


class PartitionedSplitRewriter(DFPatternCallback):
    """This pass brings the split out of the partitioned function"""

    def __init__(self):
        super().__init__(require_type=True, rewrite_once=True)
        self.pattern = (
            wildcard().has_attr({"Composite": ethosu_patterns.SplitParams.composite_name})
        )(wildcard())

    def callback(
        self, pre: tvm.relay.Expr, post: tvm.relay.Expr, node_map: tvm.ir.container.Map
    ) -> tvm.relay.Expr:
        split_input = post.args[0]
        split_params = ethosu_patterns.SplitParams(post.op.body)
        indices_or_sections = split_params.indices_or_sections
        axis = split_params.axis
        return relay.op.split(split_input, indices_or_sections, axis=axis).astuple()


def get_lut_from_func(
    ifm_scale: float,
    ifm_zp: int,
    ofm_scale: float,
    ofm_zp: int,
    func: Callable[[float], float],
) -> List[int]:
    """Calculates the values of the lookup table based on the calculation function"""

    lut_values = list()
    # Only int8 is currently supported
    dtype = np.int8
    qmin, qmax = np.iinfo(dtype).min, np.iinfo(dtype).max
    for x in range(qmin, qmax + 1):
        x_real = ifm_scale * (x - ifm_zp)
        out_real = func(x_real)
        lut_result = int(util.round_away_zero(ofm_zp + out_real / ofm_scale))
        lut_result = min(qmax, max(qmin, lut_result))
        lut_values.append(lut_result)

    return lut_values


class LutActivationRewriter(DFPatternCallback):
    """A class to create an identity operator with the LUT"""

    def __init__(
        self,
        params_class: Type,
        activation_type: str,
        calc_func: Callable[[float], float],
    ):
        super().__init__(require_type=True, rewrite_once=True)
        self.params_class = params_class
        self.pattern = (wildcard().has_attr({"Composite": params_class.composite_name}))(wildcard())
        self.activation_type = activation_type
        self.calc_func = calc_func

    def callback(self, pre: tvm.relay.Expr, post: tvm.relay.Expr, node_map: tvm.ir.container.Map):
        params = self.params_class(post.op.body)
        params.ifm.tensor = post.args[0]

        input_scale = float(params.ifm.q_params.scale_f32)
        input_zp = int(params.ifm.q_params.zero_point)
        output_scale = float(params.ofm.q_params.scale_f32)
        output_zp = int(params.ofm.q_params.zero_point)

        lut_values = get_lut_from_func(
            input_scale,
            input_zp,
            output_scale,
            output_zp,
            self.calc_func,
        )
        lut = relay.const(lut_values, dtype=params.ifm.dtype)

        # We baked the requantization into the LUT, so we don't requantize the identity operator
        identity = ethosu_ops.ethosu_identity(
            ifm=params.ifm.tensor,
            lut=lut,
            ifm_scale=input_scale,
            ifm_zero_point=input_zp,
            ofm_scale=input_scale,
            ofm_zero_point=input_zp,
            activation=self.activation_type,
        )

        return identity


class TanhRewriter(LutActivationRewriter):
    """This pass adds tanh as a LUT to the identity operator"""

    def __init__(self):
        super().__init__(
            params_class=ethosu_patterns.TanhParams, activation_type="TANH", calc_func=math.tanh
        )


def sigmoid_calc_func(x: float) -> float:
    """Function to calculate the values for sigmoid"""
    # These limits are inherited from TFLite
    upper_limit = 8.0
    lower_limit = -8.0

    if x <= lower_limit:
        y = 0.0
    elif x >= upper_limit:
        y = 1.0
    else:
        y = 1 / (1 + math.exp(-x))
    return y


class SigmoidRewriter(LutActivationRewriter):
    """This pass adds sigmoid as a LUT for identity op"""

    def __init__(self):
        super().__init__(
            params_class=ethosu_patterns.SigmoidParams,
            activation_type="SIGMOID",
            calc_func=sigmoid_calc_func,
        )


def leaky_relu_calc_func(x: float, alpha: float) -> float:
    """Function to calculate the values for leaky relu."""
    return x if x >= 0 else x * alpha


class LeakyReLURewriter(DFPatternCallback):
    """This pass adds leaky relu as a LUT for identity op."""

    def __init__(self):
        super().__init__(require_type=True, rewrite_once=True)
        self.params_class = ethosu_patterns.LeakyReLUParams
        self.pattern = wildcard().has_attr({"Composite": self.params_class.composite_name})(
            wildcard()
        )

    def callback(self, pre: tvm.relay.Expr, post: tvm.relay.Expr, node_map: tvm.ir.container.Map):
        params = self.params_class(post.op.body)
        params.ifm.tensor = post.args[0]

        input_scale = np.double(float(params.ifm.q_params.scale_f32))
        input_zp = int(params.ifm.q_params.zero_point)
        output_scale = np.double(float(params.ofm.q_params.scale_f32))
        output_zp = int(params.ofm.q_params.zero_point)

        alpha = params.alpha

        # The calculation of the LUT values is similar to that in Vela
        # convert_lrelu_to_lut(op, arch)
        # (https://review.mlplatform.org/plugins/gitiles/ml/ethos-u/ethos-u-vela/+/refs/tags/3.2.0/ethosu/vela/tflite_graph_optimiser.py#864)  # pylint: disable=line-too-long
        alpha_scalar = 1
        alpha_scale, alpha_shift = scaling.elementwise_mul_scale(input_scale, alpha, output_scale)
        identity_scale, identity_shift = scaling.elementwise_mul_scale(input_scale, 1, output_scale)

        dtype = params.ifm.dtype
        qmin, qmax = np.iinfo(dtype).min, np.iinfo(dtype).max

        def calculate_lut_value(i):
            zp_shift = (
                fp_math.multiply_by_quantized_multiplier(
                    alpha_scalar * (i - input_zp), alpha_scale, alpha_shift
                )
                if i < input_zp
                else fp_math.multiply_by_quantized_multiplier(
                    i - input_zp, identity_scale, identity_shift
                )
            )

            return min(qmax, max(qmin, output_zp + zp_shift))

        values = list(map(calculate_lut_value, range(qmin, qmax + 1)))
        lut = relay.const(values, dtype=dtype)

        # We baked the requantization into the LUT, so we don't requantize the identity operator
        identity = ethosu_ops.ethosu_identity(
            ifm=params.ifm.tensor,
            lut=lut,
            ifm_scale=input_scale,
            ifm_zero_point=input_zp,
            ofm_scale=input_scale,
            ofm_zero_point=input_zp,
            activation="LUT",
        )

        return identity


class HardSwishRewriter(DFPatternCallback):
    """Convert ethosu.hard_swish composite function to add operation with LUT."""

    def __init__(self):
        super().__init__(require_type=True, rewrite_once=True)
        self.params_class = ethosu_patterns.HardSwishParams
        self.pattern = wildcard().has_attr({"Composite": self.params_class.composite_name})(
            wildcard()
        )

    def callback(self, pre: tvm.relay.Expr, post: tvm.relay.Expr, node_map: tvm.ir.container.Map):
        params = self.params_class(post.op.body)
        params.ifm.tensor = post.args[0]

        # The calculation of the LUT values is similar to that in Vela
        # convert_hardswish_to_lut(op, arch, nng)
        # (https://review.mlplatform.org/plugins/gitiles/ml/ethos-u/ethos-u-vela/+/refs/tags/3.2.0/ethosu/vela/tflite_graph_optimiser.py#719)  # pylint: disable=line-too-long
        input_scale = np.double(params.ifm.q_params.scale_f32)
        input_zp = int(params.ifm.q_params.zero_point)
        hires_input_scale = (1 / 128) * input_scale

        output_scale = np.double(params.ofm.q_params.scale_f32)
        output_zp = int(params.ofm.q_params.zero_point)
        output_scale, output_shift = scaling.quantise_scale(hires_input_scale / output_scale)
        output_scale_16 = fp_math.downscale_multiplier_int32_to_int16(output_scale)
        output_shift = 31 - output_shift
        output_shift = -output_shift if output_shift < 0 else 0

        dtype = params.ifm.dtype
        qmin, qmax = np.iinfo(dtype).min, np.iinfo(dtype).max

        def calculate_relu_multiplier(inp, input_scale):
            rmultiplier = np.double(3 / 32768)
            rscale, rshift = scaling.quantise_scale(input_scale / rmultiplier)
            rscale_16 = fp_math.downscale_multiplier_int32_to_int16(rscale)

            rvalue = np.int16(inp)
            if rshift < 31:
                rvalue = fp_math.shift_left16(rvalue, 30 - rshift)
                rvalue = fp_math.saturating_rounding_mul16(rvalue, rscale_16)
                rvalue = fp_math.shift_left16(rvalue, 1)
            elif rshift > 31:
                rvalue = fp_math.saturating_rounding_mul16(rvalue, rscale_16)
                rvalue = fp_math.rounding_divide_by_pot(rvalue, rshift - 31)
            else:
                rvalue = fp_math.saturating_rounding_mul16(rvalue, rscale_16)

            rvalue = (rvalue + (1 << 15)) >> 1
            return rvalue

        def calculate_lut_values(i):
            hires_input_value = (i - input_zp) * 128
            preshift_input_value = fp_math.saturating_rounding_mul16(
                hires_input_value, output_scale_16
            )
            relu_value = calculate_relu_multiplier(hires_input_value, hires_input_scale)
            lut_result = fp_math.saturating_mul16(relu_value, preshift_input_value)
            lut_result = fp_math.rounding_divide_by_pot(lut_result, output_shift) + output_zp
            return min(qmax, max(qmin, lut_result))

        values = list(map(calculate_lut_values, range(-128, 128)))
        lut = relay.const(values, dtype=dtype)

        # We baked the requantization into the LUT, so we don't requantize the identity operator
        identity = ethosu_ops.ethosu_identity(
            ifm=params.ifm.tensor,
            lut=lut,
            ifm_scale=input_scale,
            ifm_zero_point=input_zp,
            ofm_scale=input_scale,
            ofm_zero_point=input_zp,
            activation="LUT",
        )

        return identity


class Conv2DRewriter(DFPatternCallback):
    """Convert conv2d related composite functions into ethosu_conv2d operators"""

    def __init__(self):
        super().__init__(require_type=True)
        self.pattern = (wildcard().has_attr({"Composite": "ethos-u.qnn_conv2d"}))(wildcard())

    def callback(
        self, pre: tvm.relay.Expr, post: tvm.relay.Expr, node_map: tvm.ir.container.Map
    ) -> tvm.relay.Expr:
        params = ethosu_patterns.QnnConv2DParams(post.op.body)
        params.ifm.tensor = post.args[0]
        channels_map = {
            "NHWC": 3,
        }
        kernel_size_map = {
            "HWIO": params.weights.shape[0:2],
            "OHWI": params.weights.shape[1:3],
            "HWOI": params.weights.shape[0:2],
        }
        activation_map = {"clip": "CLIP"}
        weight_to_ohwi_transform_map = {"HWIO": [3, 0, 1, 2]}
        weights_values = params.weights.values
        weights_values_ohwi = np.transpose(
            weights_values, weight_to_ohwi_transform_map[str(params.weights.layout)]
        )
        if params.activation:
            activation = activation_map[params.activation.op.name]
            clip_min = int(params.activation.attrs.a_min)
            clip_max = int(params.activation.attrs.a_max)
        else:
            activation = "NONE"
            clip_min = 0
            clip_max = 0
        scale_bias = vela_api.pack_biases(
            biases=params.biases.tensor.data.asnumpy(),
            ifm_scale=params.ifm.q_params.scale_f32,
            ifm_dtype=np.dtype(params.ifm.dtype),
            weight_scales=params.weights.q_params.scale_f32,
            ofm_scale=params.ofm.q_params.scale_f32,
            is_activation_tanh_or_sigmoid=activation in ["TANH", "SIGMOID"],
        )
        ethosu_conv2d = ethosu_ops.ethosu_conv2d(
            ifm=post.args[0],
            weight=relay.const(weights_values_ohwi, params.weights.values.dtype),
            scale_bias=relay.const(scale_bias, "uint8"),
            lut=relay.const([], dtype="int8"),
            ifm_scale=float(params.ifm.q_params.scale_f32),
            ifm_zero_point=int(params.ifm.q_params.zero_point),
            weight_zero_point=int(params.weights.q_params.zero_point),
            ofm_scale=float(params.ofm.q_params.scale_f32),
            ofm_zero_point=int(params.ofm.q_params.zero_point),
            kernel_shape=kernel_size_map[str(params.weights.layout)],
            ofm_channels=params.ofm.shape[channels_map[str(params.ofm.layout)]],
            strides=params.strides,
            padding=params.padding,
            dilation=params.dilation,
            activation=activation,
            clip_min=clip_min,
            clip_max=clip_max,
            upscale="NONE",
            ifm_layout=str(params.ifm.layout),
            ofm_layout=str(params.ofm.layout),
        )
        return ethosu_conv2d


class Conv2DTransposeRewriter(DFPatternCallback):
    """Convert conv2d_transpose related composite functions into
    ethosu_conv2d_transpose operators."""

    def __init__(self):
        super().__init__(require_type=True)
        self.pattern = (wildcard().has_attr({"Composite": "ethos-u.qnn_conv2d_transpose"}))(
            wildcard()
        )

    def callback(
        self, pre: tvm.relay.Expr, post: tvm.relay.Expr, node_map: tvm.ir.container.Map
    ) -> tvm.relay.Expr:
        params = ethosu_patterns.QnnConv2DTransposeParams(post.op.body)
        params.ifm.tensor = post.args[0]

        ofm_shape = params.ofm.shape
        legalize_padding = params.legalize_padding

        weight_to_ohwi_transform_map = {"IOHW": [1, 2, 3, 0]}
        weights_values = params.weights.values
        weights_values_ohwi = np.transpose(
            weights_values, weight_to_ohwi_transform_map[str(params.weights.layout)]
        )
        weights_values_ohwi = np.flip(weights_values_ohwi, (1, 2))
        weights = relay.const(weights_values_ohwi, dtype=params.weights.values.dtype)

        bias_values = (
            params.biases.tensor.data.asnumpy()
            if params.biases
            else np.zeros((params.ifm.shape[-1]))
        )
        scale_bias = vela_api.pack_biases(
            biases=bias_values,
            ifm_scale=params.ifm.q_params.scale_f32,
            ifm_dtype=np.dtype(params.ifm.dtype),
            weight_scales=params.weights.q_params.scale_f32,
            ofm_scale=params.ofm.q_params.scale_f32,
            is_activation_tanh_or_sigmoid=False,
        )

        reduced_op = ethosu_ops.ethosu_conv2d(
            ifm=post.args[0],
            weight=weights,
            scale_bias=relay.const(scale_bias, "uint8"),
            lut=relay.const([], dtype="int8"),
            ifm_scale=float(params.ifm.q_params.scale_f32),
            ifm_zero_point=int(params.ifm.q_params.zero_point),
            weight_zero_point=int(params.weights.q_params.zero_point),
            ofm_scale=float(params.ofm.q_params.scale_f32),
            ofm_zero_point=int(params.ofm.q_params.zero_point),
            kernel_shape=params.kernel_shape,
            ofm_channels=int(ofm_shape[-1]),
            strides=(1, 1),
            padding=legalize_padding,
            dilation=params.dilation,
            ifm_layout=str(params.ifm.layout),
            ofm_layout=str(params.ofm.layout),
            upscale="ZEROS",
        )

        # Remove additional padding by 'cropping' back to expected size
        return relay.strided_slice(reduced_op, (0, 0, 0, 0), ofm_shape)


class DepthwiseConv2DRewriter(DFPatternCallback):
    """Convert ethosu.qnn_depthwise_conv2d composite functions to ethosu_depthwise_conv2d
    operators"""

    def __init__(self):
        super().__init__(require_type=True)
        self.pattern = (
            wildcard().has_attr(
                {"Composite": ethosu_patterns.QnnDepthwiseConv2DParams.composite_name}
            )
        )(wildcard())

    def callback(
        self, pre: tvm.relay.Expr, post: tvm.relay.Expr, node_map: tvm.ir.container.Map
    ) -> tvm.relay.Expr:
        params = ethosu_patterns.QnnDepthwiseConv2DParams(post.op.body)
        params.ifm.tensor = post.args[0]
        channels_map = {
            "NHWC": 3,
        }
        kernel_shape_map = {
            "HWOI": params.weights.shape[0:2],
        }

        weights_values = params.weights.values
        weights_values_ohwi = np.moveaxis(weights_values, [0, 1, 2, 3], [1, 2, 0, 3])

        activation = "NONE"
        # Activations requiring LUT is currently not supported, so setting it to an empty list
        lut = relay.const([], "int8")
        clip_min = 0
        clip_max = 0
        if params.activation:
            activation = ethosu_patterns.QnnDepthwiseConv2DParams.activation_map[
                params.activation.op.name
            ]
            if activation == "CLIP":
                clip_min = int(params.activation.attrs.a_min)
                clip_max = int(params.activation.attrs.a_max)
        scale_bias = vela_api.pack_biases(
            biases=params.biases.tensor.data.asnumpy(),
            ifm_scale=params.ifm.q_params.scale_f32,
            ifm_dtype=np.dtype(params.ifm.dtype),
            weight_scales=params.weights.q_params.scale_f32,
            ofm_scale=params.ofm.q_params.scale_f32,
            is_activation_tanh_or_sigmoid=activation in ["TANH", "SIGMOID"],
        )

        ethosu_depthwise_conv2d = ethosu_ops.ethosu_depthwise_conv2d(
            post.args[0],  # IFM
            relay.const(weights_values_ohwi, params.weights.values.dtype),
            relay.const(scale_bias, "uint8"),
            lut,
            float(params.ifm.q_params.scale_f32),
            int(params.ifm.q_params.zero_point),
            int(params.weights.q_params.zero_point),
            float(params.ofm.q_params.scale_f32),
            int(params.ofm.q_params.zero_point),
            kernel_shape_map[str(params.weights.layout)],
            params.ofm.shape[channels_map[str(params.ofm.layout)]],
            strides=params.strides,
            padding=params.padding,
            dilation=params.dilation,
            activation=activation,
            clip_min=clip_min,
            clip_max=clip_max,
            upscale="NONE",
            ifm_layout=str(params.ifm.layout),
            ofm_layout=str(params.ofm.layout),
            ofm_dtype=str(params.ofm.dtype),
        )
        return ethosu_depthwise_conv2d


class PoolingRewriter(DFPatternCallback):
    """Convert ethosu.avgpool2d and ethosu.maxpool2d composite functions to
    ethosu_pooling operators"""

    def __init__(
        self,
        params_class: Type,
        pattern: CallPattern,
    ):
        super().__init__(require_type=True)
        self.params_class = params_class
        self.pattern = pattern

    def callback(
        self, pre: tvm.relay.Expr, post: tvm.relay.Expr, node_map: tvm.ir.container.Map
    ) -> tvm.relay.Expr:
        params = self.params_class(post.op.body)
        params.ifm.tensor = post.args[0]
        channels_map = {
            "NHWC": 3,
        }

        activation_map = {"clip": "CLIP"}
        if params.activation:
            activation = activation_map[params.activation.op.name]
            clip_min = int(params.activation.attrs.a_min)
            clip_max = int(params.activation.attrs.a_max)
        else:
            activation = "NONE"
            clip_min = 0
            clip_max = 0

        # Activations requiring LUT is currently not supported, so setting it to an empty list
        lut = relay.const([], dtype="int8")

        # If ethosu.avgpool2d has strides which are not supported by the NPU, convert
        # ethosu.avgpool2d composite functions to ethosu_pooling operator with stride=[1, 1].
        # Since the spatial dimensions of ifm and the pooling kernel coincide and the padding
        # is [0, 0, 0, 0], the application of the pooling kernel will be done only once,
        # which will give us the desired output
        strides = params.strides
        if params.strides[0] > 3 or params.strides[1] > 3:
            strides = [1, 1]

        return ethosu_ops.ethosu_pooling(
            ifm=post.args[0],
            lut=lut,
            pooling_type=params.pooling_type,
            ifm_scale=params.ifm.q_params.scale_f32,
            ifm_zero_point=params.ifm.q_params.zero_point,
            ofm_scale=params.ofm.q_params.scale_f32,
            ofm_zero_point=params.ofm.q_params.zero_point,
            pool_shape=params.pool_shape,
            ofm_channels=params.ofm.shape[channels_map[str(params.ofm.layout)]],
            ofm_dtype=params.ofm.dtype,
            strides=strides,
            padding=params.padding,
            activation=activation,
            clip_min=clip_min,
            clip_max=clip_max,
            upscale="NONE",
            ifm_layout=str(params.ifm.layout),
            ofm_layout=str(params.ofm.layout),
        )


class MaxPoolingRewriter(PoolingRewriter):
    def __init__(self):
        super().__init__(
            params_class=ethosu_patterns.MaxPool2DParams,
            pattern=(
                wildcard().has_attr({"Composite": ethosu_patterns.MaxPool2DParams.composite_name})
            )(wildcard()),
        )


class AvgPoolingRewriter(PoolingRewriter):
    def __init__(self):
        super().__init__(
            params_class=ethosu_patterns.AvgPool2DParams,
            pattern=(
                wildcard().has_attr({"Composite": ethosu_patterns.AvgPool2DParams.composite_name})
            )(wildcard()),
        )


class BinaryElementwiseRewriter(DFPatternCallback):
    """Convert ethosu binary elementwise composite functions to
    ethosu_binary_elementwise operators"""

    def __init__(
        self,
        params_class: Type,
        pattern: CallPattern,
    ):
        super().__init__(require_type=True)
        self.params_class = params_class
        self.pattern = pattern

    @staticmethod
    def reshape_input(
        inputs: List["TensorParams"],
    ) -> List[tvm.relay.Expr]:
        """Reshape the inputs so that the following binary elementwise
        operator receives 4-dimensional inputs.

        Parameters
        ----------
        inputs: List[TensorParams]
            The inputs to reshape.

        Returns
        -------
        reshaped_inputs: List[tvm.relay.Expr]
            The new reshaped inputs.
        """
        reshaped_inputs = []
        for i in inputs:
            in_shape = i.shape
            if len(in_shape) < 4:
                pad_size = 4 - len(in_shape)
                new_shape = ([1] * pad_size) + in_shape
                new_call = relay.reshape(i.tensor, new_shape)
                reshaped_inputs.append(new_call)
            else:
                reshaped_inputs.append(i.tensor)
        return reshaped_inputs

    @staticmethod
    def reshape_output(output: tvm.relay.Expr, ifm_input_shape: List[int]) -> tvm.relay.Expr:
        """Reshape the output back to the original dimensionality.
        Since the NPU must have the brodcastable tensor as the
        second operand, the original shape of the first ifm must
        be the output shape.

        Parameters
        ----------
        output: tvm.relay.Expr
            The output to reshape.

        ifm_input_shape: List[int]
            The shape of the non-reshaped ifm tensor.

        Returns
        -------
        reshaped_output: tvm.relay.Expr
            The reshaped output expression.
        """
        if len(ifm_input_shape) == 4:
            return output
        reshaped_output = relay.reshape(output, ifm_input_shape)
        return reshaped_output

    def callback(
        self, pre: tvm.relay.Expr, post: tvm.relay.Expr, node_map: tvm.ir.container.Map
    ) -> tvm.relay.Expr:
        params = self.params_class(post.op.body)
        params.ifm.tensor = post.args[1] if params.reversed_operands else post.args[0]
        params.ifm2.tensor = post.args[0] if params.reversed_operands else post.args[1]

        activation_map = {"clip": "CLIP"}
        if params.activation:
            activation = activation_map[params.activation.op.name]
            clip_min = int(params.activation.attrs.a_min)
            clip_max = int(params.activation.attrs.a_max)
        else:
            activation = "NONE"
            clip_min = 0
            clip_max = 0

        # We don't yet support activation functions that need to get legalized to LUTs.
        lut = relay.const([], dtype="int8")

        inputs = [params.ifm, params.ifm2]
        inputs = self.reshape_input(inputs)

        ethosu_binary_elementwise = ethosu_ops.ethosu_binary_elementwise(
            ifm=inputs[0],
            ifm2=inputs[1],
            lut=lut,
            operator_type=params.operator_type,
            ifm_scale=float(params.ifm.q_params.scale_f32),
            ifm_zero_point=int(params.ifm.q_params.zero_point),
            ifm2_scale=float(params.ifm2.q_params.scale_f32),
            ifm2_zero_point=int(params.ifm2.q_params.zero_point),
            ofm_scale=float(params.ofm.q_params.scale_f32),
            ofm_zero_point=int(params.ofm.q_params.zero_point),
            ifm_channels=params.ifm.shape[-1] if params.ifm.shape else 1,
            ifm2_channels=params.ifm2.shape[-1] if params.ifm2.shape else 1,
            reversed_operands=params.reversed_operands,
            ofm_dtype=params.ofm.dtype,
            activation=activation,
            clip_min=clip_min,
            clip_max=clip_max,
            ifm_layout=str(params.ifm.layout),
            ifm2_layout=str(params.ifm2.layout),
            ofm_layout=str(params.ofm.layout),
        )
        output = self.reshape_output(ethosu_binary_elementwise, params.ifm.shape)
        return output


class AddRewriter(BinaryElementwiseRewriter):
    def __init__(self):
        super().__init__(
            params_class=ethosu_patterns.AddParams,
            pattern=(wildcard().has_attr({"Composite": ethosu_patterns.AddParams.composite_name}))(
                wildcard(), wildcard()
            ),
        )


class SubRewriter(BinaryElementwiseRewriter):
    def __init__(self):
        super().__init__(
            params_class=ethosu_patterns.SubParams,
            pattern=(wildcard().has_attr({"Composite": ethosu_patterns.SubParams.composite_name}))(
                wildcard(), wildcard()
            ),
        )


class MulRewriter(BinaryElementwiseRewriter):
    def __init__(self):
        super().__init__(
            params_class=ethosu_patterns.MulParams,
            pattern=(wildcard().has_attr({"Composite": ethosu_patterns.MulParams.composite_name}))(
                wildcard(), wildcard()
            ),
        )


class MinRewriter(BinaryElementwiseRewriter):
    def __init__(self):
        super().__init__(
            params_class=ethosu_patterns.MinParams,
            pattern=(wildcard().has_attr({"Composite": ethosu_patterns.MinParams.composite_name}))(
                wildcard(), wildcard()
            ),
        )


class MaxRewriter(BinaryElementwiseRewriter):
    def __init__(self):
        super().__init__(
            params_class=ethosu_patterns.MaxParams,
            pattern=(wildcard().has_attr({"Composite": ethosu_patterns.MaxParams.composite_name}))(
                wildcard(), wildcard()
            ),
        )


class ShlRewriter(BinaryElementwiseRewriter):
    def __init__(self):
        super().__init__(
            params_class=ethosu_patterns.ShlParams,
            pattern=(wildcard().has_attr({"Composite": ethosu_patterns.ShlParams.composite_name}))(
                wildcard(), wildcard()
            ),
        )


class StridedSliceRewriter(DFPatternCallback):
    """This pass brings the strided slice out of the partitioned function"""

    def __init__(self):
        super().__init__(require_type=True, rewrite_once=True)
        self.pattern = (
            wildcard().has_attr({"Composite": ethosu_patterns.StridedSliceParams.composite_name})
        )(wildcard())

    def callback(
        self, pre: tvm.relay.Expr, post: tvm.relay.Expr, node_map: tvm.ir.container.Map
    ) -> tvm.relay.Expr:

        slice_input = post.args[0]

        # TODO(lhutton1) For an unknown reason compilation will fail for strides of 4
        # dimensions, so we cannot use params.strides as this will sometimes give
        # strides as [1, 1, 1, 1]. Since we only support strides of 1, hardcoding this
        # value for now.
        strides = [1]

        params = ethosu_patterns.StridedSliceParams(post.op.body)
        strided_slice = relay.op.strided_slice(
            slice_input,
            params.begin,
            params.end,
            strides=strides,
            axes=params.axes,
            slice_mode=params.slice_mode,
        )
        return strided_slice


class ReshapeRewriter(DFPatternCallback):
    """This pass brings the reshape out of the partitioned function"""

    def __init__(self):
        super().__init__(require_type=True, rewrite_once=True)
        self.pattern = (
            wildcard().has_attr({"Composite": ethosu_patterns.ReshapeParams.composite_name})
        )(wildcard())

    def callback(
        self, pre: tvm.relay.Expr, post: tvm.relay.Expr, node_map: tvm.ir.container.Map
    ) -> tvm.relay.Expr:
        reshape_input = post.args[0]
        reshape_params = ethosu_patterns.ReshapeParams(post.op.body)
        new_shape = reshape_params.new_shape
        return relay.op.reshape(reshape_input, newshape=new_shape)


class NoOpRewriter(DFPatternCallback):
    """This pass adds an idenity operator to reshape and strided slice to avoid a no op
    without a consumer"""

    def __init__(self):
        super().__init__(require_type=True, rewrite_once=True)
        self.reshape = is_op("reshape")(wildcard())
        self.strided_slice = is_op("strided_slice")(wildcard())
        self.pattern = self.reshape | self.strided_slice

    def callback(
        self, pre: tvm.relay.Expr, post: tvm.relay.Expr, node_map: tvm.ir.container.Map
    ) -> tvm.relay.Expr:
        if pre.checked_type.dtype == "int32":
            return post
        return ethosu_ops.ethosu_identity(ifm=post, lut=relay.const([], dtype="int8"))


class UnaryElementwiseRewriter(DFPatternCallback):
    """
    Convert ethosu unary elementwise composite function to
    ethosu_unary_elementwise operators
    """

    def __init__(self, params_class: Type, pattern: CallPattern):
        super().__init__(require_type=True)
        self.params_class = params_class
        self.pattern = pattern

    def callback(
        self, pre: tvm.relay.Expr, post: tvm.relay.Expr, node_map: tvm.ir.container.Map
    ) -> tvm.relay.Expr:
        params = self.params_class(post.op.body)
        params.ifm.tensor = post.args[0]

        activation_map = {"clip": "CLIP"}
        if params.activation:
            activation = activation_map[params.activation.op.name]
            clip_min = int(params.activation.attrs.a_min)
            clip_max = int(params.activation.attrs.a_max)
        else:
            activation = "NONE"
            clip_min = 0
            clip_max = 0

        # We don't yet support activation functions that use LUT.
        lut = relay.const([], dtype="int8")

        unary_input_shape = params.ifm.shape
        # If the input tensor is not 4D, enter reshapes before and after the unary operator
        if len(params.ifm.shape) == 4:
            unary_input = params.ifm.tensor
        else:
            pad_size = 4 - len(unary_input_shape)
            unary_input_shape = ([1] * pad_size) + unary_input_shape
            unary_input = relay.op.reshape(params.ifm.tensor, newshape=unary_input_shape)

        ethosu_unary_elementwise = ethosu_ops.ethosu_unary_elementwise(
            ifm=unary_input,
            lut=lut,
            operator_type=params.operator_type,
            ifm_scale=float(params.ifm.q_params.scale_f32),
            ifm_zero_point=int(params.ifm.q_params.zero_point),
            ofm_scale=float(params.ofm.q_params.scale_f32),
            ofm_zero_point=int(params.ofm.q_params.zero_point),
            ofm_channels=unary_input_shape[3],
            activation=activation,
            clip_min=clip_min,
            clip_max=clip_max,
            ifm_layout=str(params.ifm.layout),
            ofm_layout=str(params.ofm.layout),
        )
        if len(params.ifm.shape) == 4:
            op = ethosu_unary_elementwise
        else:
            op = relay.op.reshape(ethosu_unary_elementwise, newshape=params.ifm.shape)
        return op


class AbsRewriter(UnaryElementwiseRewriter):
    def __init__(self):
        super().__init__(
            params_class=ethosu_patterns.AbsParams,
            pattern=(wildcard().has_attr({"Composite": ethosu_patterns.AbsParams.composite_name}))(
                wildcard()
            ),
        )


class MeanRewriter(DFPatternCallback):
    """Convert ethosu.mean composite functions to an equivalent legalization:
    - Case 1 (ifm qparams == ofm qparams): ethosu_pooling
    - Case 2 (else): ethosu_depthwise_conv2d
    """

    def __init__(self):
        super().__init__(require_type=True)
        self.pattern = (
            wildcard().has_attr({"Composite": ethosu_patterns.MeanParams.composite_name})
        )(wildcard())

    def callback(
        self, pre: tvm.relay.Expr, post: tvm.relay.Expr, node_map: tvm.ir.container.Map
    ) -> tvm.relay.Expr:
        params = ethosu_patterns.MeanParams(post.op.body)
        params.ifm.tensor = post.args[0]

        ifm_shape = params.ifm.shape
        ofm_shape = params.ofm.shape
        lut = relay.const([], "int8")
        axis = params.axis
        reduced_op = params.ifm.tensor

        # Enforce 4d input
        if len(ifm_shape) < 4:
            axis = [x + 1 for x in axis]
            if len(ifm_shape) == 3:
                ifm_shape = [1, params.height, params.width, ifm_shape[2]]
            else:
                ifm_shape = [1, params.height, params.width, 1]
            reduced_op = relay.reshape(reduced_op, ifm_shape)

        filter_height = ifm_shape[1] if 1 in axis else 1
        filter_width = ifm_shape[2] if 2 in axis else 1
        in_channels = out_channels = ifm_shape[-1]

        # If the height is greater than max kernel height, reshape the input
        # from [filter_height, filter_width] to [1, (filter_height*filter_width)]
        # only in the case the axis is [1, 2].
        if axis == [1, 2] and filter_height > 64:
            ifm_shape = (ifm_shape[0], 1, filter_height * filter_width, in_channels)
            filter_width = filter_height * filter_width
            filter_height = 1
            reduced_op = relay.reshape(reduced_op, ifm_shape)

        if (
            params.ifm.q_params.scale_f32 == params.ofm.q_params.scale_f32
            and params.ifm.q_params.zero_point == params.ofm.q_params.zero_point
        ):
            reduced_op = ethosu_ops.ethosu_pooling(
                ifm=reduced_op,
                lut=lut,
                pooling_type="AVG",
                ifm_scale=float(params.ifm.q_params.scale_f32),
                ifm_zero_point=0,
                ofm_scale=float(params.ofm.q_params.scale_f32),
                ofm_zero_point=0,
                pool_shape=(filter_height, filter_width),
                ofm_channels=out_channels,
                ofm_dtype=params.ofm.dtype,
                rounding_mode="TRUNCATE",
            )
        else:
            weight_scale = 1 / (filter_height * filter_width)
            weight_values = np.ones([out_channels, filter_height, filter_width, 1])
            bias = -1 * int(params.ifm.q_params.zero_point) * filter_height * filter_width

            scale_bias = vela_api.pack_biases(
                biases=np.ones([ifm_shape[-1]]) * bias,
                ifm_scale=params.ifm.q_params.scale_f32,
                ifm_dtype=np.dtype(params.ifm.dtype),
                weight_scales=np.array([weight_scale], dtype=np.float),
                ofm_scale=params.ofm.q_params.scale_f32,
                is_activation_tanh_or_sigmoid=False,
            )
            reduced_op = ethosu_ops.ethosu_depthwise_conv2d(
                ifm=reduced_op,
                weight=relay.const(weight_values, params.ifm.dtype),
                scale_bias=relay.const(scale_bias, "uint8"),
                lut=lut,
                ifm_scale=float(params.ifm.q_params.scale_f32),
                ifm_zero_point=0,
                weight_zero_point=0,
                ofm_scale=float(params.ofm.q_params.scale_f32),
                ofm_zero_point=int(params.ofm.q_params.zero_point),
                kernel_shape=(filter_height, filter_width),
                ofm_channels=out_channels,
                rounding_mode="NATURAL",
                ofm_dtype=params.ofm.dtype,
            )

        # Reshape to original ofm shape
        if len(ofm_shape) < 4:
            reduced_op = relay.reshape(reduced_op, ofm_shape)

        return reduced_op


class SumRewriter(DFPatternCallback):
    """
    Convert ethosu.sum composite functions to pooling operations
    """

    def __init__(self):
        super().__init__(require_type=True)
        self.pattern = (
            wildcard().has_attr({"Composite": ethosu_patterns.SumParams.composite_name})
        )(wildcard())

    def callback(
        self, pre: tvm.relay.Expr, post: tvm.relay.Expr, node_map: tvm.ir.container.Map
    ) -> tvm.relay.Expr:

        params = ethosu_patterns.SumParams(post.op.body)

        ifm_shape = params.ifm.shape
        ofm_shape = params.ofm.shape
        lut = relay.const([], "int8")
        reduced_op = post.args[0]

        # Enforce 4d input
        if len(ifm_shape) == 3:
            ifm_shape = [1, params.height, params.width, ifm_shape[2]]
            reduced_op = relay.reshape(reduced_op, ifm_shape)

        activation_map = {"clip": "CLIP"}
        if params.activation:
            activation = activation_map[params.activation.op.name]
            clip_min = int(params.activation.attrs.a_min)
            clip_max = int(params.activation.attrs.a_max)
        else:
            activation = "NONE"
            clip_min = 0
            clip_max = 0

        reduced_op = ethosu_ops.ethosu_pooling(
            ifm=reduced_op,
            lut=lut,
            pooling_type="SUM",
            ifm_scale=float(params.ifm.q_params.scale_f32),
            ifm_zero_point=int(params.ifm.q_params.zero_point),
            ofm_scale=float(params.ofm.q_params.scale_f32),
            ofm_zero_point=0,
            pool_shape=(1, 1),
            ofm_channels=1,
            ofm_dtype="int32",
            activation=activation,
            clip_min=clip_min,
            clip_max=clip_max,
            ifm_layout=params.ifm.layout,
            ofm_layout=params.ofm.layout,
            rounding_mode="NATURAL",
        )

        # Convert tensor dtype from int32 to int8
        scalar_tensor = relay.const(np.ones([1, 1, 1, 1], dtype="int32"), dtype="int32")
        reduced_op = ethosu_ops.ethosu_binary_elementwise(
            ifm=reduced_op,
            ifm2=scalar_tensor,
            lut=lut,
            operator_type="MUL",
            ifm_scale=0.0,
            ifm_zero_point=0,
            ifm2_scale=0.0,
            ifm2_zero_point=0,
            ofm_scale=0.0,
            ofm_zero_point=int(params.ofm.q_params.zero_point),
            ifm_channels=1,
            ifm2_channels=1,
            reversed_operands=False,
            ofm_dtype="int8",
        )

        # Reshape to original ofm shape
        if len(ofm_shape) < 4:
            reduced_op = relay.reshape(reduced_op, ofm_shape)

        return reduced_op


class ConcatRewriter(DFPatternCallback):
    """The newer versions of TFLite converters return a concatenate operator that concatenates
    tensors with same QNN params (if the QNN params of tensors were initially different,
    the converter adds a requantize node), so this rewriter replaces the QNN concatenate with
    "normal" concatenate"""

    def __init__(self):
        super().__init__(require_type=True, rewrite_once=True)
        self.pattern = (
            wildcard().has_attr({"Composite": ethosu_patterns.ConcatParams.composite_name})
        )(None)

    def callback(
        self, pre: tvm.relay.Expr, post: tvm.relay.Expr, node_map: tvm.ir.container.Map
    ) -> tvm.relay.Expr:
        # Find the tensors that are inputs to the concat and the scales and zero points
        concat_args = list()
        for arg in post.args:
            if isinstance(arg, (tvm.relay.expr.Call, tvm.relay.expr.TupleGetItem)):
                concat_args.append(arg)

        axis = post.op.body.attrs.axis
        concat = relay.op.concatenate(relay.Tuple(concat_args), axis=axis)
        return concat


class RequantizeRewriter(DFPatternCallback):
    """Convert ethos-u.requantize composite function to an identity operation."""

    def __init__(self):
        super().__init__(require_type=True)
        self.pattern = (
            wildcard().has_attr({"Composite": ethosu_patterns.RequantizeParams.composite_name})
        )(wildcard())

    def callback(
        self, pre: tvm.relay.Expr, post: tvm.relay.Expr, node_map: tvm.ir.container.Map
    ) -> tvm.relay.Expr:
        params = ethosu_patterns.RequantizeParams(post.op.body)
        params.ifm.tensor = post.args[0]

        lut = relay.const([], "int8")

        return ethosu_ops.ethosu_identity(
            ifm=params.ifm.tensor,
            lut=lut,
            ifm_scale=float(params.ifm.q_params.scale_f32),
            ifm_zero_point=int(params.ifm.q_params.zero_point),
            ofm_scale=float(params.ofm.q_params.scale_f32),
            ofm_zero_point=int(params.ofm.q_params.zero_point),
            rounding_mode="NATURAL",
        )


class Resize2dRewriter(DFPatternCallback):
    """
    Convert ethos-u.resize2d composite function to an equivalent operation that
    performs the relevant upsampling operation.

    Case 1: No upsampling (upscale factor of 1):
        Identity.
    Case 1: Nearest neighbor upsampling:
        1x1 pooling with 2x2 nearest neighbor upsampling.
    Case 2: Bilinear upsampling:
        2x2 average pool with 2x2 nearest neighbor upsampling.
    """

    def __init__(self):
        super().__init__(require_type=True)
        self.pattern = (
            wildcard().has_attr({"Composite": ethosu_patterns.Resize2dParams.composite_name})
        )(wildcard())

    def callback(
        self, pre: tvm.relay.Expr, post: tvm.relay.Expr, node_map: tvm.ir.container.Map
    ) -> tvm.relay.Expr:
        params = ethosu_patterns.Resize2dParams(post.op.body)
        params.ifm.tensor = post.args[0]

        lut = relay.const([], "int8")
        ifm_shape = params.ifm.shape
        in_channels = ifm_shape[-1]
        reduced_op = params.ifm.tensor
        current_size = np.array(ifm_shape[1:3])
        output_size = np.array(params.size)

        if (current_size == output_size).all():
            return ethosu_ops.ethosu_identity(
                reduced_op,
                lut,
                ifm_scale=float(params.ifm.q_params.scale_f32),
                ifm_zero_point=int(params.ifm.q_params.zero_point),
                ofm_scale=float(params.ofm.q_params.scale_f32),
                ofm_zero_point=int(params.ofm.q_params.zero_point),
            )

        padding = [0, 0, 0, 0]
        rounding_mode = "TFL"
        pool_shape = [1, 1]
        if params.method == "linear":
            pool_shape = [2, 2]
            rounding_mode = "NATURAL"
            if params.coordinate_transformation_mode == "asymmetric":
                # Use SAME padding.
                ypad = Resize2dRewriter.get_required_padding(ifm_shape[1])
                xpad = Resize2dRewriter.get_required_padding(ifm_shape[2])
                padding = [ypad // 2, xpad // 2, (ypad + 1) // 2, (xpad + 1) // 2]

        return ethosu_ops.ethosu_pooling(
            ifm=reduced_op,
            lut=lut,
            pooling_type="AVG",
            ifm_scale=float(params.ifm.q_params.scale_f32),
            ifm_zero_point=int(params.ifm.q_params.zero_point),
            ofm_scale=float(params.ofm.q_params.scale_f32),
            ofm_zero_point=int(params.ofm.q_params.zero_point),
            pool_shape=pool_shape,
            ofm_channels=in_channels,
            ofm_dtype=params.ofm.dtype,
            strides=[1, 1],
            padding=padding,
            upscale="NEAREST",
            rounding_mode=rounding_mode,
        )

    @staticmethod
    def get_required_padding(input_size: int, pool_size: int = 2) -> int:
        """Gets the amount of padding required needed to achieve
        'SAME' padding for a given axis."""
        needed_input = (input_size - 1) + pool_size
        total_padding = max(0, needed_input - input_size)
        return total_padding


class ExpandDimsRewriter(DFPatternCallback):
    """Legalize expand dims to a reshape operator."""

    def __init__(self):
        super().__init__(require_type=True, rewrite_once=True)
        self.pattern = (
            wildcard().has_attr({"Composite": ethosu_patterns.ExpandDimsParams.composite_name})
        )(None)

    def callback(
        self, pre: tvm.relay.Expr, post: tvm.relay.Expr, node_map: tvm.ir.container.Map
    ) -> tvm.relay.Expr:
        params = ethosu_patterns.ExpandDimsParams(post.op.body)
        return relay.op.reshape(post.args[0], newshape=params.output.shape)


class SqueezeRewriter(DFPatternCallback):
    """Legalize squeeze to a reshape operator."""

    def __init__(self):
        super().__init__(require_type=True, rewrite_once=True)
        self.pattern = (
            wildcard().has_attr({"Composite": ethosu_patterns.SqueezeParams.composite_name})
        )(None)

    def callback(
        self, pre: tvm.relay.Expr, post: tvm.relay.Expr, node_map: tvm.ir.container.Map
    ) -> tvm.relay.Expr:
        params = ethosu_patterns.SqueezeParams(post.op.body)
        return relay.op.reshape(post.args[0], newshape=params.output.shape)


class FullyConnectedRewriter(DFPatternCallback):
    """Legalize Fully Connected (with bias and clip) to an NPU operator"""

    def __init__(self):
        super().__init__(require_type=True)
        self.pattern = (
            wildcard().has_attr({"Composite": ethosu_patterns.FullyConnectedParams.composite_name})
        )(wildcard())

    def callback(self, pre, post, node_map):
        params = ethosu_patterns.FullyConnectedParams(post.op.body)
        params.ifm.tensor = post.args[0]

        # IFM reshapes
        ifm = post.args[0]
        if len(params.ifm.shape) != 4 or not params.ifm.shape[1] == params.ifm.shape[2] == 1:
            ifm = relay.reshape(ifm, (1, 1, 1, params.ifm.shape[-1]))

        # Weight transformations
        weights_values = params.weights.values
        weights_values_ohwi = np.expand_dims(weights_values, axis=(1, 2))
        if params.activation:
            activation = "CLIP"
            clip_min = int(params.activation.attrs.a_min)
            clip_max = int(params.activation.attrs.a_max)
        else:
            activation = "NONE"
            clip_min = 0
            clip_max = 0
        bias_values = (
            params.biases.tensor.data.asnumpy()
            if params.biases
            else np.zeros((params.ofm.shape[-1]))
        )
        scale_bias = vela_api.pack_biases(
            biases=bias_values,
            ifm_scale=params.ifm.q_params.scale_f32,
            ifm_dtype=np.dtype(params.ifm.dtype),
            weight_scales=params.weights.q_params.scale_f32,
            ofm_scale=params.ofm.q_params.scale_f32,
            is_activation_tanh_or_sigmoid=False,
        )
        ethosu_fc = ethosu_ops.ethosu_conv2d(
            ifm=ifm,
            weight=relay.const(weights_values_ohwi, params.weights.values.dtype),
            scale_bias=relay.const(scale_bias, "uint8"),
            lut=relay.const([], dtype="int8"),
            ifm_scale=float(params.ifm.q_params.scale_f32),
            ifm_zero_point=int(params.ifm.q_params.zero_point),
            weight_zero_point=int(params.weights.q_params.zero_point),
            ofm_scale=float(params.ofm.q_params.scale_f32),
            ofm_zero_point=int(params.ofm.q_params.zero_point),
            kernel_shape=[1, 1],
            ofm_channels=params.weights.shape[0],
            strides=(1, 1),
            padding=(0, 0, 0, 0),
            dilation=(1, 1),
            activation=activation,
            clip_min=clip_min,
            clip_max=clip_max,
            upscale="NONE",
            ifm_layout="NHWC",
            ofm_layout="NHWC",
        )

        if len(params.ofm.shape) != 4 or not params.ofm.shape[1] == params.ofm.shape[2] == 1:
            ethosu_fc = relay.reshape(ethosu_fc, params.ofm.shape)
        return ethosu_fc


class MatMulRewriter(DFPatternCallback):
    """Legalize matrix multiplication to an NPU operator"""

    def __init__(self):
        super().__init__(require_type=True)
        self.pattern = (
            wildcard().has_attr({"Composite": ethosu_patterns.MatMulParams.composite_name})
        )(wildcard(), wildcard())

    def callback(self, pre, post, node_map):
        params = ethosu_patterns.MatMulParams(post.op.body)
        ifm = post.args[0]
        ifm2 = post.args[1]
        lut = relay.const([], dtype="int8")
        activation_map = {"clip": "CLIP"}
        if params.activation:
            activation = activation_map[params.activation.op.name]
            clip_min = int(params.activation.attrs.a_min)
            clip_max = int(params.activation.attrs.a_max)
        else:
            activation = "NONE"
            clip_min = 0
            clip_max = 0

        # Reshape ifm to NHWC
        ifm = relay.reshape(ifm, (1, 1, *params.ifm.shape))
        # Split the second matrix to get columns
        columns = list(relay.op.split(ifm2, params.ofm.shape[-1], axis=0))

        res_columns = []
        for column in columns:
            ifm2 = relay.reshape(column, (1, 1, 1, params.ifm.shape[-1]))
            # Multiplying the first matrix by a column
            ethosu_binary_elementwise = ethosu_ops.ethosu_binary_elementwise(
                ifm=ifm,
                ifm2=ifm2,
                lut=lut,
                operator_type="MUL",
                ifm_zero_point=int(params.ifm.q_params.zero_point),
                ifm_scale=0.0,
                ifm2_zero_point=int(params.weights.q_params.zero_point),
                ifm2_scale=0.0,
                ofm_scale=0.0,
                ofm_zero_point=0,
                ifm_channels=params.ifm.shape[-1],
                ifm2_channels=params.ifm.shape[-1],
                reversed_operands=False,
                ofm_dtype="int32",
            )

            # Use reduce sum to get result column
            reduce_sum = ethosu_ops.ethosu_pooling(
                ifm=ethosu_binary_elementwise,
                lut=lut,
                pooling_type="SUM",
                ifm_zero_point=0,
                ifm_scale=float(params.weights.q_params.scale_f32)
                * float(params.ifm.q_params.scale_f32),
                ofm_scale=float(params.ofm.q_params.scale_f32),
                ofm_zero_point=0,
                pool_shape=(1, 1),
                ofm_channels=1,
                ofm_dtype="int32",
                activation=activation,
                clip_min=clip_min,
                clip_max=clip_max,
                rounding_mode="NATURAL",
            )

            # Convert tensor dtype from int32 to int8
            scalar_tensor = relay.const(np.ones([1, 1, 1, 1], dtype="int32"), dtype="int32")
            reduce_sum = ethosu_ops.ethosu_binary_elementwise(
                ifm=reduce_sum,
                ifm2=scalar_tensor,
                lut=lut,
                operator_type="MUL",
                ifm_scale=0.0,
                ifm_zero_point=0,
                ifm2_scale=0.0,
                ifm2_zero_point=0,
                ofm_scale=0.0,
                ofm_zero_point=int(params.ofm.q_params.zero_point),
                ifm_channels=1,
                ifm2_channels=1,
                reversed_operands=False,
                ofm_dtype="int8",
            )

            res_columns.append(reduce_sum)

        # Concatenate result columns
        concat = relay.op.concatenate(relay.Tuple(res_columns), axis=3)
        return relay.reshape(concat, params.ofm.shape)


class PadRewriter(DFPatternCallback):
    """Convert ethos-u.pad2d composite function to ethosu_depthwise_conv2d
    operator"""

    def __init__(self):
        super().__init__(require_type=True)
        self.pattern = (
            wildcard().has_attr({"Composite": ethosu_patterns.PadParams.composite_name})
        )(wildcard())

    def callback(
        self, pre: tvm.relay.Expr, post: tvm.relay.Expr, node_map: tvm.ir.container.Map
    ) -> tvm.relay.Expr:
        params = ethosu_patterns.PadParams(post.op.body)
        params.ifm.tensor = post.args[0]
        channels_map = {
            "NHWC": 3,
        }
        w_h, w_w = (1, 1)
        # OHWI format for the ethosu_depthwise_conv2d kernel weights
        weight_shape = (params.ifm.shape[-1], w_h, w_w, 1)
        weights = relay.const(np.full(weight_shape, 1), params.ifm.dtype)
        scale_bias = vela_api.pack_biases(
            biases=np.zeros(params.ifm.shape[-1]),
            ifm_scale=params.ifm.q_params.scale_f32,
            ifm_dtype=np.dtype(params.ifm.dtype),
            weight_scales=np.array(1.0, dtype=np.float32),
            ofm_scale=params.ofm.q_params.scale_f32,
            is_activation_tanh_or_sigmoid=False,
        )

        return ethosu_ops.ethosu_depthwise_conv2d(
            ifm=post.args[0],
            weight=weights,
            scale_bias=relay.const(scale_bias, "uint8"),
            lut=relay.const([], "int8"),
            ifm_scale=float(params.ifm.q_params.scale_f32),
            ifm_zero_point=int(params.ifm.q_params.zero_point.item()),
            weight_zero_point=0,
            ofm_scale=float(params.ofm.q_params.scale_f32),
            ofm_zero_point=int(params.ofm.q_params.zero_point.item()),
            kernel_shape=(w_h, w_w),
            ofm_channels=params.ofm.shape[channels_map[str(params.ofm.layout)]],
            strides=(1, 1),
            padding=params.padding,
            dilation=(1, 1),
            activation="NONE",
            clip_min=0,
            clip_max=0,
            upscale="NONE",
            ifm_layout=str(params.ifm.layout),
            ofm_layout=str(params.ofm.layout),
            ofm_dtype=str(params.ofm.dtype),
        )


class ChannelPadRewriter(DFPatternCallback):
    """Convert ethos-u.channel-pad composite function to the Relay concatenate operation"""

    def __init__(self):
        super().__init__(require_type=True)
        self.pattern = (
            wildcard().has_attr({"Composite": ethosu_patterns.ChannelPadParams.composite_name})
        )(wildcard())

    def callback(
        self, pre: tvm.relay.Expr, post: tvm.relay.Expr, node_map: tvm.ir.container.Map
    ) -> tvm.relay.Expr:
        params = ethosu_patterns.ChannelPadParams(post.op.body)
        params.ifm.tensor = post.args[0]

        concat_args = list()
        lut = relay.const([], dtype="int8")
        # pad channels before
        if params.ch_padding[0] > 0:
            shape1 = list(params.ifm.shape)
            shape1[3] = params.ch_padding[0].value
            pad_channels = relay.Constant(
                tvm.nd.array(
                    np.full(
                        shape=shape1,
                        fill_value=int(params.ifm.q_params.zero_point),
                        dtype=params.ifm.dtype,
                    )
                )
            )
            identity1 = ethosu_ops.ethosu_identity(
                ifm=pad_channels,
                lut=lut,
                ifm_scale=float(params.ifm.q_params.scale_f32),
                ifm_zero_point=int(params.ifm.q_params.zero_point),
                ofm_scale=float(params.ofm.q_params.scale_f32),
                ofm_zero_point=int(params.ofm.q_params.zero_point),
            )
            concat_args.append(identity1)

        identity2 = ethosu_ops.ethosu_identity(
            ifm=params.ifm.tensor,
            lut=lut,
            ifm_scale=float(params.ifm.q_params.scale_f32),
            ifm_zero_point=int(params.ifm.q_params.zero_point),
            ofm_scale=float(params.ofm.q_params.scale_f32),
            ofm_zero_point=int(params.ofm.q_params.zero_point),
        )
        concat_args.append(identity2)

        # pad channels after
        if params.ch_padding[1] > 0:
            shape3 = list(params.ifm.shape)
            shape3[3] = params.ch_padding[1].value
            pad_channels3 = relay.Constant(
                tvm.nd.array(
                    np.full(
                        shape=shape3,
                        fill_value=int(params.ifm.q_params.zero_point),
                        dtype=params.ifm.dtype,
                    )
                )
            )
            identity3 = ethosu_ops.ethosu_identity(
                ifm=pad_channels3,
                lut=lut,
                ifm_scale=float(params.ifm.q_params.scale_f32),
                ifm_zero_point=int(params.ifm.q_params.zero_point),
                ofm_scale=float(params.ofm.q_params.scale_f32),
                ofm_zero_point=int(params.ofm.q_params.zero_point),
            )
            concat_args.append(identity3)

        return relay.op.concatenate(relay.Tuple(concat_args), axis=3)


@util.create_npu_function_pass(opt_level=1)
class LegalizeEthosU:
    """This is the pass to call graph-rewrites to perform graph transformation
    in a way such that the operations are replaced with hardware/codegen supported
    operations.
    """

    def transform_npu_function(self, _, func: relay.Function) -> relay.Function:
        """This is the method that replaces the operations with hardware/codegen supported
        operations.
        """
        rewriters = [
            PartitionedSplitRewriter(),
            FullyConnectedRewriter(),
            MatMulRewriter(),
            SplitRewriter(),
            ChannelPadRewriter(),
            Conv2DRewriter(),
            Conv2DTransposeRewriter(),
            DepthwiseConv2DRewriter(),
            MaxPoolingRewriter(),
            AvgPoolingRewriter(),
            PadRewriter(),
            AddRewriter(),
            SubRewriter(),
            MulRewriter(),
            MinRewriter(),
            MaxRewriter(),
            ShlRewriter(),
            AbsRewriter(),
            TanhRewriter(),
            HardSwishRewriter(),
            LeakyReLURewriter(),
            MeanRewriter(),
            SumRewriter(),
            SoftmaxRewriter(),
            ConcatRewriter(),
            SigmoidRewriter(),
            RequantizeRewriter(),
            Resize2dRewriter(),
            ExpandDimsRewriter(),
            SqueezeRewriter(),
            ReshapeRewriter(),
            StridedSliceRewriter(),
            NoOpRewriter(),
        ]
        for rewriter in rewriters:
            func = rewrite(rewriter, func)

        return func

    def __call__(self, *args, **kwargs):
        # pylint is unable figure out the decorated
        # class is callable, thus adding this to
        # suppress the warning.
        pass
