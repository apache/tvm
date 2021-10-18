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
"""Extract information from the depthwise convolution operators in TIR."""
from typing import Dict, Tuple
import tvm
from ..vela_api import SCALE_BIAS_LENGTH
from .utils import get_outer_loops, get_op_attrs, get_base_address, get_loads, get_stores
from .dma import get_ifm_params, get_ofm_params
from .spec import (
    SerialKernel,
    SerialAddressRange,
    SerialActivation,
    Serial2DDepthwise,
)


def get_depthwise_conv2d_params(
    stmt: tvm.tir.AttrStmt,
    producers: Dict[tvm.tir.Var, tvm.tir.AttrStmt],
    consumers: Dict[tvm.tir.Var, tvm.tir.AttrStmt],
) -> Tuple[Serial2DDepthwise, tvm.tir.Var, tvm.tir.Var]:
    """Get the parameters necessary to construct a call_extern for a depthwise_conv2d.

    Parameters
    ----------
    stmt : tvm.tir.AttrStmt
        The outermost attribute statement of a depthwise loop nest.
    producers : Dict[tvm.tir.Var, tvm.tir.AttrStmt]
        A dictionary to associate pointers with the loop nest
        that produces their values.
    consumers : Dict[tvm.tir.Var, tvm.tir.AttrStmt]
        A dictionary to associate pointers with the loop nest
        that consumes their values.

    Returns
    -------
    Serial2DDepthwise
        The parameters needed to construct a 2D depthwise.
    output_pointer : tvm.tir.Var
        The output pointer of the convolution operation.
    replace_pointer : tvm.tir.Var
        The output pointer of the DMA write operation, which is to replace
        the convolution output pointer.

    """
    attrs, body = get_op_attrs(stmt)
    _, _, _, _, _, inner = get_outer_loops(body, "NHWC")
    rh = inner
    rw = rh.body
    # loads = [output, input, weights, scale_bias, scale_bias]
    loads = get_loads(rw.body)
    # stores = [output]
    stores = get_stores(rw.body)
    input_pointer = loads[1].buffer_var
    output_pointer = stores[0].buffer_var
    # Get feature map info
    serial_ifm, serial_padding = get_ifm_params(input_pointer, producers)
    serial_ofm, replace_pointer = get_ofm_params(output_pointer, consumers)
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
    scale_bias_base = get_base_address(scale_bias_load.index)
    serial_scale_bias = SerialAddressRange(
        address=tvm.tir.Load("uint8", scale_bias_load.buffer_var, scale_bias_base),
        length=SCALE_BIAS_LENGTH * serial_ofm[3],
    )
    # Get weight info
    weight_load = loads[2]
    weight_base = get_base_address(weight_load.index)
    serial_weight = SerialAddressRange(
        address=tvm.tir.Load("uint8", weight_load.buffer_var, weight_base),
        length=serial_ofm[3] * serial_kernel[0] * serial_kernel[1],
    )
    # Get activation info
    serial_activation = SerialActivation(
        op=attrs["activation"], clip_min=attrs["clip_min"], clip_max=attrs["clip_max"]
    )

    return (
        Serial2DDepthwise(
            ifm=serial_ifm,
            ofm=serial_ofm,
            kernel=serial_kernel,
            weight=serial_weight,
            weight_zero_point=attrs["weight_zero_point"],
            scale_bias=serial_scale_bias,
            padding=serial_padding,
            activation=serial_activation,
            upscale="NONE",
        ),
        output_pointer,
        replace_pointer,
    )
