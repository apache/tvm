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
"""This source will contain code to convert TIR, as produced by
the Relay to TIR compilation process, to Vela API calls to
generate command stream.
"""
from typing import NamedTuple
from enum import auto
from enum import Enum
import numpy as np  # type: ignore
import ethosu.vela.api as vapi  # type: ignore

import tvm
from tvm.relay.backend.contrib.ethosu import vela_api
from tvm.relay.backend.contrib.ethosu.tir import spec


class BufferType(Enum):
    """The buffer types the codegen supports"""

    constant = auto()
    input_or_output = auto()
    scratch = auto()
    input = auto()
    output = auto()


class BufferInfo(NamedTuple):
    """A data structure to hold metadata of the buffer"""

    # If the buffer holds constants, the values will contain that otherwise None
    values: np.ndarray
    shape: tvm.ir.container.Array
    dtype: np.dtype
    btype: BufferType


def extract_buffer_info(mod, param_dict):
    """
    This function is to read the tvm.IRModule that
    contains Relay to TIR compiled IRModule. Thereafter,
    this will extract the buffer information as the shape
    and constant data (if any).

    Parameters
    ----------
    mod : tvm.IRModule
        The NPU TIR IRModule.
    param_dict : dict
        A dictionary containing param idx --> const numpy.NDArray
    Returns
    -------
    dict
        a dictionary of buffer names --> BufferInfo
    """
    buffer_info = dict()
    # There should only be a single function
    assert len(mod.functions.items()) == 1
    primfunc = mod.functions.items()[0][1]
    for idx, const_data in param_dict.items():
        param = primfunc.params[idx]
        buffer_info[primfunc.buffer_map[param].data] = BufferInfo(
            const_data, const_data.shape, const_data.dtype, BufferType.constant
        )

    for param in primfunc.params:
        if primfunc.buffer_map[param].data not in buffer_info.keys():
            buffer_info[primfunc.buffer_map[param].data] = BufferInfo(
                None,
                primfunc.buffer_map[param].shape,
                primfunc.buffer_map[param].dtype,
                BufferType.input_or_output,
            )

    def populate_allocate_buffer_info(stmt):
        if isinstance(stmt, tvm.tir.stmt.Allocate):
            allocate = stmt
            buffer_info[allocate.buffer_var] = BufferInfo(
                None,
                allocate.extents,
                allocate.dtype,
                BufferType.scratch,
            )

    tvm.tir.stmt_functor.post_order_visit(primfunc.body, populate_allocate_buffer_info)

    return buffer_info


def _convert_clip_bounds(npu_op):
    """
    This function will convert the min and max value
    of clip activations to non quantized floats as
    expected by the API.
    Parameters
    ----------
    npu_op : ethosu.vela.api.NpuBlockOperation
    """
    clip_min_quant = npu_op.activation.min
    clip_max_quant = npu_op.activation.max
    clip_min_actual = (
        clip_min_quant - npu_op.ofm.quantization.zero_point
    ) * npu_op.ofm.quantization.scale_f32
    clip_max_actual = (
        clip_max_quant - npu_op.ofm.quantization.zero_point
    ) * npu_op.ofm.quantization.scale_f32
    npu_op.activation.min = clip_min_actual
    npu_op.activation.max = clip_max_actual


def translate_ethosu_conv2d(tir_extern_call):
    """This function will translate a tir extern_call
    as produced by Relay to TIR compilation.
    Parameters
    ----------
    tir_extern_call : tvm.tir.Call
        This should be an tir external call that has a agreed upon ordering
        for TIR Compiler. See Serial2DConvolution in
        tvm/relay/backend/contrib/ethosu/tir/spec.py for the ordering.

    Returns
    -------
    ethosu.vela.api.NpuConv2DOperation
        The vela object containing the params of ethosu_conv2d
    weights_zero_point : int
        The zero point of the weights
    """
    # We skip the first element as it is the extern_call function name
    serial_object = spec.create_serial_object(spec.Serial2DConvolution, tir_extern_call.args[1:])
    return _create_npu_op_conv2d(serial_object)


def _create_npu_op_conv2d(serial_2d_convolution):
    """This is a helper function to capture a list
    of arguments to create Vela NpuConv2DOperation object
    """
    npu_conv2d_op = vapi.NpuConv2DOperation()
    npu_conv2d_op.ifm = _create_npu_feature_map(serial_2d_convolution.ifm)
    npu_conv2d_op.ofm = _create_npu_feature_map(serial_2d_convolution.ofm)
    npu_conv2d_op.kernel = _create_npu_kernel(serial_2d_convolution.kernel)
    npu_conv2d_op.weights = [_create_npu_address_range(serial_2d_convolution.weight)]
    weights_zero_point = np.int64(serial_2d_convolution.weight_zero_point.value)
    npu_conv2d_op.biases = [_create_npu_address_range(serial_2d_convolution.scale_bias)]
    npu_conv2d_op.padding = _create_npu_padding(serial_2d_convolution.padding)

    npu_conv2d_op.activation = _create_npu_activation(serial_2d_convolution.activation)
    if (
        npu_conv2d_op.activation
        and npu_conv2d_op.activation.op_type == vapi.NpuActivationOp.NONE_OR_RELU
    ):
        _convert_clip_bounds(npu_conv2d_op)

    npu_conv2d_op.upscale = _create_npu_resampling_mode(serial_2d_convolution.upscale)
    target_accel_type = vela_api.get_target_accel_type()  # type: ignore
    block_config = vela_api.get_optimal_block_config(npu_conv2d_op, target_accel_type)
    npu_conv2d_op.block_config = block_config
    weights_shape_ohwi = [
        npu_conv2d_op.ofm.shape.depth,
        npu_conv2d_op.kernel.height,
        npu_conv2d_op.kernel.width,
        npu_conv2d_op.ifm.shape.depth,
    ]
    npu_conv2d_op.block_traversal = vela_api.calculate_block_traversal_mode(
        is_depthwise=False,
        weights_shape_ohwi=weights_shape_ohwi,
        ifm_bitdepth=npu_conv2d_op.ifm.data_type.size_in_bits(),
    )
    return npu_conv2d_op, weights_zero_point


def _create_npu_feature_map(serial_feature_map):
    """This is a helper function to capture a list
    of arguments to create Vela NpuFeatureMap object
    """
    layout_map = {"NHWC": vapi.NpuLayout.NHWC, "NHCWB16": vapi.NpuLayout.NHCWB16}
    datatype_map = {
        "uint8": vapi.NpuDataType.UINT8,
        "int8": vapi.NpuDataType.INT8,
        "uint16": vapi.NpuDataType.UINT16,
        "int16": vapi.NpuDataType.INT16,
        "int32": vapi.NpuDataType.INT32,
    }
    layout = str(serial_feature_map.layout.value)
    data_type = str(serial_feature_map.data_type.value)
    assert layout in layout_map.keys()
    assert data_type in datatype_map.keys()
    nfm = vapi.NpuFeatureMap()
    nfm.data_type = datatype_map[data_type]
    nfm.shape = vapi.NpuShape3D(
        int(serial_feature_map.height.value),
        int(serial_feature_map.width.value),
        int(serial_feature_map.channels.value),
    )
    nfm.tiles = vapi.NpuTileBox(
        int(serial_feature_map.tile_height_0.value),
        int(serial_feature_map.tile_height_1.value),
        int(serial_feature_map.tile_width_0.value),
        [
            serial_feature_map.tile_address_0,
            serial_feature_map.tile_address_1,
            serial_feature_map.tile_address_2,
            serial_feature_map.tile_address_3,
        ],
    )
    nfm.quantization = _create_npu_quantization(
        serial_feature_map.scale, serial_feature_map.zero_point
    )
    nfm.layout = layout_map[layout]
    nfm.strides = vapi.NpuShape3D(
        int(serial_feature_map.stride_h.value),
        int(serial_feature_map.stride_w.value),
        int(serial_feature_map.stride_c.value),
    )
    return nfm


def _create_npu_kernel(serial_kernel):
    """This is a helper function to capture a list
    of arguments to create Vela NpuKernel object
    """
    nknl = vapi.NpuKernel(
        w=int(serial_kernel.width.value),
        h=int(serial_kernel.height.value),
        stride_x=int(serial_kernel.stride_w.value),
        stride_y=int(serial_kernel.stride_h.value),
        dilation_x=int(serial_kernel.dilation_w.value),
        dilation_y=int(serial_kernel.dilation_h.value),
    )
    return nknl


def _create_npu_address_range(serial_address_range):
    """This is a helper function to capture a list
    of arguments to create Vela NpuAddressRange object
    """
    addr_range = vapi.NpuAddressRange(
        # region will be updated later
        region=0,
        address=serial_address_range.address,
        length=int(serial_address_range.length.value),
    )
    return addr_range


def _create_npu_quantization(
    scale,
    zero_point,
):
    """This is a helper function to capture a list
    of arguments to create Vela NpuQuantization object
    """
    # Scale could be an ndarray if per-channel quantization is available
    if not isinstance(scale, tvm.tir.expr.Load):
        if isinstance(scale.value, float):
            scale = np.single(scale.value)
        else:
            assert isinstance(scale.value.value, float)
            scale = np.single(scale.value.value)
    q_params = vapi.NpuQuantization(scale_f32=scale, zero_point=zero_point.value)
    return q_params


def _create_npu_weights_zero_point(
    zero_point,
):
    """This is a helper function to capture the weights zero point"""
    return zero_point.value


def _create_npu_padding(serial_padding):
    """This is a helper function to capture a list
    of arguments to create Vela NpuPadding object"""
    padding = vapi.NpuPadding(
        top=int(serial_padding.top.value),
        left=int(serial_padding.left.value),
        bottom=int(serial_padding.bottom.value),
        right=int(serial_padding.right.value),
    )
    return padding


def _create_npu_activation(serial_activation):
    """This is a helper function to capture a list
    of arguments to create Vela NpuActivation object"""
    if serial_activation.op == "NONE":
        return None
    if (
        serial_activation.op == "CLIP"
        and serial_activation.clip_min == 0
        and serial_activation.clip_max == 0
    ):
        return None
    op_map = {
        "CLIP": vapi.NpuActivationOp.NONE_OR_RELU,
        "TANH": vapi.NpuActivationOp.TANH,
        "SIGMOID": vapi.NpuActivationOp.SIGMOID,
    }
    op = str(serial_activation.op.value)
    assert op in op_map.keys()
    act_op = vapi.NpuActivation(op_map[op])
    act_op.min = int(serial_activation.clip_min.value)
    act_op.max = int(serial_activation.clip_max.value)
    return act_op


def _create_npu_resampling_mode(
    mode,
):
    """This is a helper function to capture a list
    of arguments to create Vela NpuResamplingMode object"""
    mode_map = {
        "NONE": vapi.NpuResamplingMode.NONE,
        "NEAREST": vapi.NpuResamplingMode.NEAREST,
        "TRANSPOSE": vapi.NpuResamplingMode.TRANSPOSE,
    }
    mode = str(mode.value)
    assert mode in mode_map.keys()
    return mode_map[mode]
