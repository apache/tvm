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
# pylint: disable=invalid-name,unused-variable,unused-argument
"""Winograd NCHW template for Adreno backend"""

import logging
from tvm import autotvm
from .conv2d_winograd_common import conv2d_winograd_comp, schedule_conv2d_winograd_impl


logger = logging.getLogger("conv2d_nchw_winograd")


@autotvm.register_topi_compute("conv2d_nchw_winograd.image2d")
def conv2d_nchw_winograd(cfg, data, kernel, strides, padding, dilation, out_dtype):
    return conv2d_nchw_winograd_comp(
        cfg, data, kernel, strides, padding, dilation, out_dtype, pre_computed=False
    )


@autotvm.register_topi_schedule("conv2d_nchw_winograd.image2d")
def schedule_conv2d_nchw_winograd(cfg, outs):
    return schedule_conv2d_winograd_impl(cfg, outs, tag="dummy_compute_at")


@autotvm.register_topi_compute("conv2d_nchw_winograd_without_weight_transform.image2d")
def conv2d_nchw_winograd_without_weight_transform(
    cfg, data, kernel, strides, padding, dilation, out_dtype
):
    return conv2d_nchw_winograd_comp(
        cfg, data, kernel, strides, padding, dilation, out_dtype, pre_computed=True
    )


@autotvm.register_topi_schedule("conv2d_nchw_winograd_without_weight_transform.image2d")
def schedule_conv2d_nchw_winograd_without_weight_transform(cfg, outs):
    return schedule_conv2d_winograd_impl(cfg, outs, tag="dummy_compute_at", pre_computed=True)


def conv2d_nchw_winograd_comp(
    cfg, data, kernel, strides, padding, dilation, out_dtype, pre_computed
):
    """Compute declaration for winograd

    Parameters
    ----------
    cfg: ConfigEntity
        The config for this template

    data: tvm.te.Tensor
        4-D or 5-D Data tensor with shape NCHW or NCHW4c

    kernel: tvm.te.Tensor
        4-D or 5-D tensor with shape OIHW or OIHW4o

    strides: int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    padding: int or a list/tuple of 2 or 4 ints
        padding size, or
        [pad_height, pad_width] for 2 ints, or
        [pad_top, pad_left, pad_bottom, pad_right] for 4 ints

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    out_dtype: str
        The output type. This is used for mixed precision.

    pre_computed: bool
        Flag if weights were pre computed if true or the weights should be
        computed in runtime

    Returns
    -------
    output: tvm.te.Tensor
        4-D or 5-D with shape NCHW or NCHW4c
    """
    return conv2d_winograd_comp(
        cfg, data, kernel, strides, padding, dilation, out_dtype, pre_computed, "NCHW"
    )
