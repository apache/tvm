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
import tvm
from tvm import relay
from tvm.relay.backend.contrib.ethosu.tir.compiler import extract_constants, lower_to_te

import numpy as np


def create_te_graph(func):
    func, consts = extract_constants(func)
    mod = tvm.IRModule.from_expr(func)
    func = relay.transform.InferType()(mod)["main"]
    te_graph = lower_to_te(func)
    return te_graph, consts


def make_matrices(kernel, stride, dilation, padding, ifm_channels, ifm_layout, ofm_layout):
    kernel_h, kernel_w = kernel
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    nhwc_to_nhcwb16 = [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 1 / 16, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 16],
        [0, 0, 0, 0, 1],
    ]
    nhcwb16_to_nhwc = [
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 16, 0, 1, -16],
        [0, 0, 0, 0, 0, 1],
    ]
    ifm_matrix = [
        [1, 0, 0, 0, 0],
        [0, stride_h, 0, 0, (dilated_kernel_h - stride_h)],
        [0, 0, stride_w, 0, (dilated_kernel_w - stride_w)],
        [0, 0, 0, 0, ifm_channels],
        [0, 0, 0, 0, 1],
    ]
    weight_matrix = [
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, kernel_h],
        [0, 0, 0, 0, kernel_w],
        [0, 0, 0, 0, ifm_channels],
        [0, 0, 0, 0, 1],
    ]
    scale_bias_matrix = [
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 10],
        [0, 0, 0, 0, 1],
    ]
    if ofm_layout == "NHCWB16":
        ifm_matrix = np.matmul(ifm_matrix, nhcwb16_to_nhwc).tolist()
        weight_matrix = np.matmul(weight_matrix, nhcwb16_to_nhwc).tolist()
        scale_bias_matrix = np.matmul(scale_bias_matrix, nhcwb16_to_nhwc).tolist()
    if ifm_layout == "NHCWB16":
        ifm_matrix = np.matmul(nhwc_to_nhcwb16, ifm_matrix).tolist()

    ifm_offset = (
        [0, -padding[0], -padding[1], 0]
        if ifm_layout == "NHWC"
        else [0, -padding[0], 0, -padding[1], 0]
    )
    weight_offset = [0, 0, 0, 0]
    scale_bias_offset = [0, 0]
    return (
        ifm_matrix,
        ifm_offset,
        weight_matrix,
        weight_offset,
        scale_bias_matrix,
        scale_bias_offset,
    )
