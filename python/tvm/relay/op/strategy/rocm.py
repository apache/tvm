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
"""Definition of ROCm operator strategy."""
# pylint: disable=invalid-name,unused-argument,unused-wildcard-import,wildcard-import
import topi
from .generic import *
from .. import op as _op

@schedule_lrn.register("rocm")
def schedule_lrn_rocm(attrs, outs, target):
    """schedule LRN for rocm"""
    with target:
        return topi.rocm.schedule_lrn(outs)

@conv2d_strategy.register("rocm")
def conv2d_strategy_rocm(attrs, inputs, out_type, target):
    """conv2d rocm strategy"""
    strategy = _op.OpStrategy()
    data, kernel = inputs
    dilation_h, dilation_w = attrs.get_int_tuple("dilation")
    groups = attrs.groups
    layout = attrs.data_layout
    stride_h, stride_w = attrs.get_int_tuple("strides")
    kernel_layout = attrs.kernel_layout
    padding = attrs.get_int_tuple("padding")
    if dilation_h < 1 or dilation_w < 1:
        raise ValueError("dilation should be positive value")

    if groups == 1:
        if layout == "NCHW":
            # TODO(@vinx13, @icemelon9): Use conv2d_NCHWc_int8 when dtype is int8/uint8.
            assert kernel_layout == "OIHW"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.cuda.conv2d_nchw),
                wrap_topi_schedule(topi.cuda.schedule_conv2d_nchw),
                name="conv2d_nchw.cuda")
            _, _, kh, kw = get_const_tuple(kernel.shape)
            if 2 < kh < 8 and 2 < kw < 8 and kh == kw and stride_h == 1 and stride_w == 1 and \
                dilation_h == 1 and dilation_w == 1:
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.cuda.conv2d_nchw_winograd),
                    wrap_topi_schedule(topi.cuda.schedule_conv2d_nchw_winograd),
                    name="conv2d_nchw_winograd.cuda",
                    plevel=5)
        elif layout == "HWCN":
            assert kernel_layout == "HWIO"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.cuda.conv2d_hwcn),
                wrap_topi_schedule(topi.cuda.schedule_conv2d_hwcn),
                name="conv2d_hwcn.cuda")
        # TODO(@alexgl-github): Re-enable this after fix the conv2d_nhwc for cuda
        # elif layout == "NHWC":
        #     assert kernel_layout == "HWIO"
        #     strategy.add_implementation(
        #         wrap_compute_conv2d(topi.cuda.conv2d_nhwc),
        #         wrap_topi_schedule(topi.cuda.schedule_conv2d_nhwc),
        #         name="conv2d_nhwc.cuda")
        elif layout == "NCHW4c" and data.dtype in ["int8", "uint8"]:
            assert kernel_layout == "OIHW4o4i"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.cuda.conv2d_NCHWc_int8, True),
                wrap_topi_schedule(topi.cuda.schedule_conv2d_NCHWc_int8),
                name="conv2d_NCHWc_int8.cuda")
        else:
            raise RuntimeError("Unsupported conv2d layout {} for CUDA".format(layout))
        # add miopen implementation
        if "miopen" in target.libs and layout == "NCHW" and padding[0] == padding[2] and \
            padding[1] == padding[3]:
            strategy.add_implementation(
                wrap_compute_conv2d(topi.rocm.conv2d_nchw_miopen, True),
                wrap_topi_schedule(topi.rocm.schedule_conv2d_nchw_miopen),
                name="conv2d_nchw_miopen.rocm",
                plevel=15)
    elif is_depthwise_conv2d(data.shape, layout, kernel.shape, kernel_layout, groups):
        if layout == "NCHW":
            assert kernel_layout == "OIHW"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.cuda.depthwise_conv2d_nchw),
                wrap_topi_schedule(topi.cuda.schedule_depthwise_conv2d_nchw),
                name="depthwise_conv2d_nchw.cuda")
        elif layout == "NHWC":
            assert kernel_layout == "HWOI"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.nn.depthwise_conv2d_nhwc),
                wrap_topi_schedule(topi.cuda.schedule_depthwise_conv2d_nhwc),
                name="depthwise_conv2d_nhwc.cuda")
        else:
            raise RuntimeError("Unsupported depthwise_conv2d layout {}".format(layout))
    else: # group_conv2d
        if layout == 'NCHW':
            # TODO(@vinx13, @icemelon9): Use group_conv2d_NCHWc_int8 when dtype is int8/uint8.
            assert kernel_layout == "OIHW"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.cuda.group_conv2d_nchw, has_groups=True),
                wrap_topi_schedule(topi.cuda.schedule_group_conv2d_nchw),
                name="group_conv2d_nchw.cuda")
        elif layout == 'NCHW4c' and data.dtype in ["int8", "uint8"]:
            assert kernel_layout == "OIHW4o4i"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.cuda.group_conv2d_NCHWc_int8, True),
                wrap_topi_schedule(topi.cuda.schedule_group_conv2d_NCHWc_int8),
                name="group_conv2d_NCHWc_int8.cuda")
        else:
            raise RuntimeError("Unsupported group_conv2d layout {}".format(layout))
    return strategy

@dense_strategy.register("rocm")
def dense_strategy_rocm(attrs, inputs, out_type, target):
    """Dense strategy for ROCM"""
    assert len(inputs[0].shape) == 2 and len(inputs[1].shape) == 2, "Only support 2-dim dense"
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_dense(topi.rocm.dense),
        wrap_topi_schedule(topi.rocm.schedule_dense),
        name="dense.rocm")
    if target.target_name == "rocm" and "rocblas" in target.libs:
        assert out_type.dtype == inputs[0].dtype, "Mixed precision not supported."
        strategy.add_implementation(
            wrap_compute_dense(topi.rocm.dense_rocblas),
            wrap_topi_schedule(topi.rocm.schedule_dense_rocblas),
            name="dense_rocblas.rocm",
            plevel=15)
    return strategy
