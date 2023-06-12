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
"""Extract parameters from the convolution operators in TIR."""
import math
from ethosu.vela import api as vapi
import tvm
from ..vela_api import SCALE_BIAS_LENGTH, get_accelerator_config
from .utils import get_outer_loops, get_op_attrs, get_base_address, get_loads, get_stores
from .dma import get_ifm_params, get_ofm_params
from .spec import SerialKernel, SerialAddressRange, SerialActivation, Serial2DConvolution


def get_conv2d_params(stmt, producers_consumers):
    """Get the parameters necessary to construct a call_extern for a 2D convolution.

    Parameters
    ----------
    stmt : tvm.tir.AttrStmt
        The outermost attribute statement of a convolution loop nest.
    producers_consumers: ProducersConsumers
        It associates pointers with the loop nest that produces
        their values and with the loop nest that consumes their values.

    Returns
    -------
    Serial2DConvolution
        The parameters needed to construct a 2D convolution.
    output_pointer : tvm.tir.Var
        The output pointer of the convolution operation.
    replace_pointer : tvm.tir.Var
        The output pointer of the DMA write operation, which is to replace
        the convolution output pointer.
    is_allocator : bool
        Whether this operator allocates its output.

    """
    accel_config = get_accelerator_config()

    attrs, body = get_op_attrs(stmt)
    _, _, _, _, _, inner = get_outer_loops(body, "NHWC")
    rh = inner
    rw = rh.body
    rc = rw.body
    # loads = [output, input, weights, scale_bias, scale_bias, LUT, LUT]
    loads = get_loads(rc.body)
    # stores = [output]
    stores = get_stores(rc.body)
    input_pointer = loads[1].buffer.data
    output_pointer = stores[0].buffer.data
    # Get feature map info
    serial_ifm, serial_padding = get_ifm_params(input_pointer, producers_consumers, stmt)
    serial_ofm, serial_block_config, replace_pointer, is_allocator = get_ofm_params(
        output_pointer, producers_consumers, stmt
    )
    # Get kernel info
    serial_kernel = SerialKernel(
        width=int(rw.extent),
        height=int(rh.extent),
        stride_w=int(attrs["stride_w"]),
        stride_h=int(attrs["stride_h"]),
        dilation_w=int(attrs["dilation_w"]),
        dilation_h=int(attrs["dilation_h"]),
    )
    # Get scale_bias info
    scale_bias_load = loads[3]
    scale_bias_base = [get_base_address(index) for index in scale_bias_load.indices]
    # Get weight info
    weight_load = loads[2]
    weight_base = [get_base_address(index) for index in weight_load.indices]
    channels = serial_ofm[3] if isinstance(serial_ofm[3], int) else serial_ofm[3].value

    if accel_config == vapi.NpuAccelerator.Ethos_U65_512:
        scale_bias_length = SCALE_BIAS_LENGTH * math.ceil(channels / 2)
        scale_bias2_length = SCALE_BIAS_LENGTH * math.floor(channels / 2)

        serial_scale_bias = SerialAddressRange(
            address=tvm.tir.BufferLoad(scale_bias_load.buffer, scale_bias_base),
            length=scale_bias_length,
        )
        serial_scale_bias2 = SerialAddressRange(
            address=tvm.tir.BufferLoad(
                scale_bias_load.buffer, [scale_bias_base[0] + scale_bias_length]
            ),
            length=scale_bias2_length,
        )

        weight_length = (
            channels * serial_kernel[0] * serial_kernel[1] * math.ceil(rc.extent.value / 2)
        )
        weight2_length = (
            channels * serial_kernel[0] * serial_kernel[1] * math.floor(rc.extent.value / 2)
        )

        serial_weight = SerialAddressRange(
            address=tvm.tir.BufferLoad(weight_load.buffer, weight_base),
            length=weight_length,
        )
        serial_weight2 = SerialAddressRange(
            address=tvm.tir.BufferLoad(weight_load.buffer, [weight_base[0] + weight_length]),
            length=weight2_length,
        )
    else:
        scale_bias_length = SCALE_BIAS_LENGTH * channels

        serial_scale_bias = SerialAddressRange(
            address=tvm.tir.BufferLoad(scale_bias_load.buffer, scale_bias_base),
            length=scale_bias_length,
        )
        # Insert -1s into the spec to denote the absence of the other pointer
        serial_scale_bias2 = SerialAddressRange(
            address=tvm.tir.IntImm("int8", -1),
            length=tvm.tir.IntImm("int8", -1),
        )

        weight_length = channels * serial_kernel[0] * serial_kernel[1] * rc.extent.value

        serial_weight = SerialAddressRange(
            address=tvm.tir.BufferLoad(weight_load.buffer, weight_base),
            length=weight_length,
        )
        serial_weight2 = SerialAddressRange(
            address=tvm.tir.IntImm("int8", -1),
            length=tvm.tir.IntImm("int8", -1),
        )
    # Get activation info
    serial_activation = SerialActivation(
        op=attrs["activation"], clip_min=attrs["clip_min"], clip_max=attrs["clip_max"]
    )
    return (
        Serial2DConvolution(
            ifm=serial_ifm,
            ofm=serial_ofm,
            kernel=serial_kernel,
            weight=serial_weight,
            weight2=serial_weight2,
            weight_zero_point=attrs["weight_zero_point"],
            scale_bias=serial_scale_bias,
            scale_bias2=serial_scale_bias2,
            padding=serial_padding,
            activation=serial_activation,
            rounding_mode=attrs["rounding_mode"],
            upscale=attrs["upscale"],
            block_config=serial_block_config,
        ),
        output_pointer,
        replace_pointer,
        is_allocator,
    )
