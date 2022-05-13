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
ethosu_enabled = True
try:
    import ethosu.vela
except ImportError:
    ethosu_enabled = False

import tvm
from tvm import relay
import tvm.contrib.ethosu.cascader as cs
import numpy as np


def make_options(
    cascade_region: cs.MemoryRegion,
    max_proposals: int = 1,
    stripe_factors: int = 1,
    max_plan_size: int = 1,
    max_open_plans: int = 8,
    max_closed_plans: int = 32,
    always_copy_size: int = 1024,
    disable_pareto_plans: bool = False,
    disable_pareto_proposals: bool = False,
    enable_striping: bool = True,
):
    return cs.CascaderOptions(
        cascade_region=cascade_region,
        max_proposals=max_proposals,
        stripe_factors=stripe_factors,
        max_plan_size=max_plan_size,
        max_open_plans=max_open_plans,
        max_closed_plans=max_closed_plans,
        always_copy_size=always_copy_size,
        disable_pareto_plans=disable_pareto_plans,
        disable_pareto_proposals=disable_pareto_proposals,
        enable_striping=enable_striping,
    )


def make_simple_home_map(graph, var_region, const_region):
    home_map = {}
    for tensor in graph.tensor_order:
        if tensor.is_constant:
            home_map[tensor] = [const_region]
        else:
            home_map[tensor] = [var_region]

    return home_map


if ethosu_enabled:
    from tvm.relay.backend.contrib.ethosu.tir.compiler import extract_constants, lower_to_te
    from tvm.relay.backend.contrib.ethosu.te.common import get_layout_transform_matrices

    def create_te_graph(func):
        func, consts = extract_constants(func)
        mod = tvm.IRModule.from_expr(func)
        func = relay.transform.InferType()(mod)["main"]
        te_graph = lower_to_te(func)
        return te_graph, consts

    def make_matrices(
        op_type,
        kernel,
        stride,
        padding,
        ifm_layout,
        ofm_layout,
        dilation=(1, 1),
        ifm_channels=1,
        ofm_channels=1,
    ):
        kernel_h, kernel_w = kernel
        stride_h, stride_w = stride
        dilation_h, dilation_w = dilation
        dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
        dilated_kernel_w = (kernel_w - 1) * dilation_w + 1

        nhwc_to_nhcwb16, nhcwb16_to_nhwc = get_layout_transform_matrices(ofm_channels)

        if op_type == "ethosu_conv2d":
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
        elif op_type == "ethosu_depthwise_conv2d":
            ifm_matrix = [
                [1, 0, 0, 0, 0],
                [0, stride_h, 0, 0, (dilated_kernel_h - stride_h)],
                [0, 0, stride_w, 0, (dilated_kernel_w - stride_w)],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ]
            weight_matrix = [
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, kernel_h],
                [0, 0, 0, 0, kernel_w],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
            ]
        elif op_type == "ethosu_pooling":
            ifm_matrix = [
                [1, 0, 0, 0, 0],
                [0, stride_h, 0, 0, (dilated_kernel_h - stride_h)],
                [0, 0, stride_w, 0, (dilated_kernel_w - stride_w)],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ]
            weight_matrix = [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
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
