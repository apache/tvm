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
from tvm import topi
from tvm.auto_scheduler import is_auto_scheduler_enabled
from tvm.te import SpecializedCondition
from tvm.contrib.thrust import can_use_rocthrust
from tvm.contrib import miopen

from .generic import *
from .. import op as _op
from .cuda import judge_winograd, naive_schedule


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
                name="conv2d_nchw.cuda",
            )
            _, _, kh, kw = get_const_tuple(kernel.shape)
            if (
                2 < kh < 8
                and 2 < kw < 8
                and kh == kw
                and stride_h == 1
                and stride_w == 1
                and dilation_h == 1
                and dilation_w == 1
            ):
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.cuda.conv2d_nchw_winograd),
                    wrap_topi_schedule(topi.cuda.schedule_conv2d_nchw_winograd),
                    name="conv2d_nchw_winograd.cuda",
                    plevel=5,
                )
        elif layout == "NHWC":
            assert kernel_layout == "HWIO"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.gpu.conv2d_nhwc),
                wrap_topi_schedule(topi.gpu.schedule_conv2d_nhwc),
                name="conv2d_nhwc.gpu",
            )
            N, H, W, _ = get_const_tuple(data.shape)
            KH, KW, CI, CO = get_const_tuple(kernel.shape)

            (_, judge_winograd_autotvm, judge_winograd_auto_scheduler,) = judge_winograd(
                N,
                H,
                W,
                KH,
                KW,
                CI,
                CO,
                padding,
                stride_h,
                stride_w,
                dilation_h,
                dilation_w,
                data.dtype,
                kernel.dtype,
                pre_flag=False,
            )

            if judge_winograd_autotvm:
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.cuda.conv2d_nhwc_winograd_direct),
                    wrap_topi_schedule(topi.cuda.schedule_conv2d_nhwc_winograd_direct),
                    name="conv2d_nhwc_winograd_direct.cuda",
                    plevel=5,
                )

            if is_auto_scheduler_enabled() and judge_winograd_auto_scheduler:
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.nn.conv2d_winograd_nhwc),
                    naive_schedule,  # this implementation should never be picked by autotvm
                    name="conv2d_nhwc.winograd",
                    plevel=15,
                )
        elif layout == "HWCN":
            assert kernel_layout == "HWIO"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.cuda.conv2d_hwcn),
                wrap_topi_schedule(topi.cuda.schedule_conv2d_hwcn),
                name="conv2d_hwcn.cuda",
            )
        elif layout == "NCHW4c" and data.dtype in ["int8", "uint8"]:
            assert kernel_layout == "OIHW4o4i"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.cuda.conv2d_NCHWc_int8, True),
                wrap_topi_schedule(topi.cuda.schedule_conv2d_NCHWc_int8),
                name="conv2d_NCHWc_int8.cuda",
            )
        else:
            raise RuntimeError("Unsupported conv2d layout {} for CUDA".format(layout))
        # add miopen implementation
        if (
            "miopen" in target.libs
            and layout == "NCHW"
            and padding[0] == padding[2]
            and padding[1] == padding[3]
        ):
            strategy.add_implementation(
                wrap_compute_conv2d(topi.rocm.conv2d_nchw_miopen, True),
                wrap_topi_schedule(topi.rocm.schedule_conv2d_nchw_miopen),
                name="conv2d_nchw_miopen.rocm",
                plevel=15,
            )
    elif is_depthwise_conv2d(data.shape, layout, kernel.shape, kernel_layout, groups):
        if layout == "NCHW":
            assert kernel_layout == "OIHW"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.cuda.depthwise_conv2d_nchw),
                wrap_topi_schedule(topi.cuda.schedule_depthwise_conv2d_nchw),
                name="depthwise_conv2d_nchw.cuda",
            )
        elif layout == "NHWC":
            assert kernel_layout == "HWOI"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.nn.depthwise_conv2d_nhwc),
                wrap_topi_schedule(topi.cuda.schedule_depthwise_conv2d_nhwc),
                name="depthwise_conv2d_nhwc.cuda",
            )
        else:
            raise RuntimeError("Unsupported depthwise_conv2d layout {}".format(layout))
    else:  # group_conv2d
        if layout == "NCHW":
            # TODO(@vinx13, @icemelon9): Use group_conv2d_NCHWc_int8 when dtype is int8/uint8.
            assert kernel_layout == "OIHW"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.cuda.group_conv2d_nchw, has_groups=True),
                wrap_topi_schedule(topi.cuda.schedule_group_conv2d_nchw),
                name="group_conv2d_nchw.cuda",
            )
        elif layout == "NCHW4c" and data.dtype in ["int8", "uint8"]:
            assert kernel_layout == "OIHW4o4i"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.cuda.group_conv2d_NCHWc_int8, True),
                wrap_topi_schedule(topi.cuda.schedule_group_conv2d_NCHWc_int8),
                name="group_conv2d_NCHWc_int8.cuda",
            )
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
        name="dense.rocm",
    )
    if target.kind.name == "rocm" and "rocblas" in target.libs:
        assert out_type.dtype == inputs[0].dtype, "Mixed precision not supported."
        strategy.add_implementation(
            wrap_compute_dense(topi.rocm.dense_rocblas),
            wrap_topi_schedule(topi.rocm.schedule_dense_rocblas),
            name="dense_rocblas.rocm",
            plevel=15,
        )
    return strategy


@batch_matmul_strategy.register("rocm")
def batch_matmul_strategy_rocm(attrs, inputs, out_type, target):
    """Batch matmul strategy for ROCM"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_batch_matmul(topi.cuda.batch_matmul),
        wrap_topi_schedule(topi.cuda.schedule_batch_matmul),
        name="batch_matmul.cuda",
        plevel=10,
    )
    if target.kind.name == "rocm" and "rocblas" in target.libs:
        assert out_type.dtype == inputs[0].dtype, "Mixed precision not supported."
        strategy.add_implementation(
            wrap_compute_batch_matmul(topi.rocm.batch_matmul_rocblas),
            wrap_topi_schedule(topi.rocm.schedule_batch_matmul_rocblas),
            name="batch_matmul_rocblas.rocm",
            plevel=12,
        )
    return strategy


@argsort_strategy.register(["rocm"])
def argsort_strategy_cuda(attrs, inputs, out_type, target):
    """argsort rocm strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_argsort(topi.cuda.argsort),
        wrap_topi_schedule(topi.cuda.schedule_argsort),
        name="argsort.rocm",
    )
    if can_use_rocthrust(target, "tvm.contrib.thrust.sort"):
        strategy.add_implementation(
            wrap_compute_argsort(topi.cuda.argsort_thrust),
            wrap_topi_schedule(topi.cuda.schedule_argsort),
            name="argsort_thrust.rocm",
            plevel=15,
        )
    return strategy


@scatter_strategy.register(["rocm"])
def scatter_cuda(attrs, inputs, out_type, target):
    """scatter rocm strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_scatter(topi.cuda.scatter),
        wrap_topi_schedule(topi.cuda.schedule_scatter),
        name="scatter.rocm",
        plevel=10,
    )

    rank = len(inputs[0].shape)

    with SpecializedCondition(rank == 1):
        if can_use_rocthrust(target, "tvm.contrib.thrust.stable_sort_by_key"):
            strategy.add_implementation(
                wrap_compute_scatter(topi.cuda.scatter_via_sort),
                wrap_topi_schedule(topi.cuda.schedule_scatter_via_sort),
                name="scatter_via_sort.rocm",
                plevel=9,  # use the sequential version by default
            )
    return strategy


@sort_strategy.register(["rocm"])
def sort_strategy_cuda(attrs, inputs, out_type, target):
    """sort rocm strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_sort(topi.cuda.sort),
        wrap_topi_schedule(topi.cuda.schedule_sort),
        name="sort.rocm",
    )
    if can_use_rocthrust(target, "tvm.contrib.thrust.sort"):
        strategy.add_implementation(
            wrap_compute_sort(topi.cuda.sort_thrust),
            wrap_topi_schedule(topi.cuda.schedule_sort),
            name="sort_thrust.cuda",
            plevel=15,
        )
    return strategy


@topk_strategy.register(["rocm"])
def topk_strategy_cuda(attrs, inputs, out_type, target):
    """topk rocm strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_topk(topi.cuda.topk),
        wrap_topi_schedule(topi.cuda.schedule_topk),
        name="topk.rocm",
    )

    if can_use_rocthrust(target, "tvm.contrib.thrust.sort"):
        strategy.add_implementation(
            wrap_compute_topk(topi.cuda.topk_thrust),
            wrap_topi_schedule(topi.cuda.schedule_topk),
            name="topk_thrust.rocm",
            plevel=15,
        )
    return strategy


@softmax_strategy.register(["rocm"])
def softmax_strategy_rocm(attrs, inputs, out_type, target):
    """rocm strategy for softmax"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_softmax(topi.nn.softmax),
        wrap_topi_schedule(topi.cuda.schedule_softmax),
        name="softmax.rocm",
    )
    if "miopen" in target.libs:
        strategy.add_implementation(
            wrap_compute_softmax(miopen.softmax),
            wrap_topi_schedule(topi.generic.schedule_extern),
            name="softmax.miopen",
            plevel=15,
        )
    return strategy


@log_softmax_strategy.register(["rocm"])
def log_softmax_strategy_rocm(attrs, inputs, out_type, target):
    """rocm strategy for log softmax"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_softmax(topi.nn.log_softmax),
        wrap_topi_schedule(topi.cuda.schedule_softmax),
        name="log_softmax.rocm",
    )
    if "miopen" in target.libs:
        strategy.add_implementation(
            wrap_compute_softmax(miopen.log_softmax),
            wrap_topi_schedule(topi.generic.schedule_extern),
            name="log_softmax.miopen",
            plevel=15,
        )
    return strategy
