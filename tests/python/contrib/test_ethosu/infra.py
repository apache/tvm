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
This module provides infrastructure to verify the correctness of
the command stream produced.

Currently it will invoke vela to generate a vela-optimized tflite
in which the command stream is contained as a custom operator.
This class include methods to parse the custom operator to extract
the command stream and perform an equivalency check for single operator
test cases.
"""

import numpy
from enum import IntEnum

import tvm
from tvm import relay
import tvm.relay.backend.contrib.ethosu.op as ethosu_ops
from tvm.topi.nn.utils import get_pad_tuple


class AttachType(IntEnum):
    kGroupRoot = 1
    kInline = 2
    kInlinedAlready = 3
    kScope = 4
    kScanUpdate = 5


def generate_weights_data(shape, dtype):
    size = 1
    for dim in shape:
        size *= dim
    return (numpy.arange(size) % 255).reshape(shape).astype(dtype)


def get_convolutional_args(call, include_buffers=False, remove_constants=False):
    """A method to extract the arguments from conv2d or depthwise2d extern call."""
    args = call.args
    conv_args = []
    remove_indices = [0]

    if remove_constants:
        remove_indices += [41, 42, 44, 45]

    for i, arg in enumerate(args):
        if i in remove_indices:
            continue
        elif isinstance(arg, tvm.tir.expr.IntImm) or isinstance(arg, tvm.tir.expr.FloatImm):
            conv_args.append(arg.value)
        elif isinstance(arg, tvm.tir.expr.Load) and not include_buffers:
            conv_args.append(arg.index)
        else:
            conv_args.append(arg)

    return conv_args


def make_ethosu_conv2d(
    ifm,
    ifm_channels,
    ofm_channels,
    kernel_shape,
    padding,
    strides,
    dilation,
    activation="NONE",
    ifm_layout="NHWC",
    ofm_layout="NHWC",
    weight_dtype="int8",
):
    # conv params
    weight_shape = (ofm_channels, kernel_shape[0], kernel_shape[1], ifm_channels)
    padding = get_pad_tuple(padding, kernel_shape)

    scale_bias_data = generate_weights_data((weight_shape[0], 10), "uint8")
    scale_bias = relay.const(scale_bias_data, dtype="uint8")
    weight_data = generate_weights_data(weight_shape, "int8")
    weight = relay.const(weight_data, dtype=weight_dtype)
    conv = ethosu_ops.ethosu_conv2d(
        ifm,
        weight,
        scale_bias,
        lut=relay.const([], dtype="int8"),
        ifm_scale=0.5,
        ifm_zero_point=10,
        weight_zero_point=12,
        ofm_scale=0.25,
        ofm_zero_point=14,
        kernel_shape=kernel_shape,
        ofm_channels=ofm_channels,
        strides=strides,
        padding=padding,
        dilation=dilation,
        activation=activation,
        clip_min=10 if activation == "CLIP" else 0,
        clip_max=100 if activation == "CLIP" else 0,
        upscale="NONE",
        ifm_layout=ifm_layout,
        ofm_layout=ofm_layout,
    )
    return conv
