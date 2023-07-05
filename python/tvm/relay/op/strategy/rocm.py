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
from tvm.te import SpecializedCondition
from tvm.contrib.thrust import can_use_rocthrust
from tvm.contrib import miopen

from .generic import *
from .. import op as _op
from .cuda import batch_matmul_strategy_cuda, conv2d_strategy_cuda, dense_strategy_cuda


@conv2d_strategy.register("rocm")
def conv2d_strategy_rocm(attrs, inputs, out_type, target):
    """conv2d rocm strategy"""
    groups = attrs.groups
    layout = attrs.data_layout
    padding = attrs.get_int_tuple("padding")

    strategy = conv2d_strategy_cuda(attrs, inputs, out_type, target)

    # add miopen implementation
    if (
        "miopen" in target.libs
        and groups == 1
        and layout == "NCHW"
        and padding[0] == padding[2]
        and padding[1] == padding[3]
    ):
        strategy.add_implementation(
            wrap_compute_conv2d(topi.rocm.conv2d_nchw_miopen, need_data_layout=True),
            wrap_topi_schedule(topi.rocm.schedule_conv2d_nchw_miopen),
            name="conv2d_nchw_miopen.rocm",
            plevel=50,
        )

    return strategy


@dense_strategy.register("rocm")
def dense_strategy_rocm(attrs, inputs, out_type, target):
    """Dense strategy for ROCM"""
    assert len(inputs[0].shape) == 2 and len(inputs[1].shape) == 2, "Only support 2-dim dense"
    strategy = dense_strategy_cuda(attrs, inputs, out_type, target)

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
    strategy = batch_matmul_strategy_cuda(attrs, inputs, out_type, target)

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


@scatter_elements_strategy.register(["rocm"])
def scatter_elements_cuda(attrs, inputs, out_type, target):
    """scatter rocm strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_scatter_elements(topi.cuda.scatter_elements),
        wrap_topi_schedule(topi.cuda.schedule_extern),
        name="scatter_elements.rocm",
        plevel=10,
    )

    rank = len(inputs[0].shape)

    with SpecializedCondition(rank == 1 and attrs.reduction == "update"):
        if can_use_rocthrust(target, "tvm.contrib.thrust.stable_sort_by_key"):
            strategy.add_implementation(
                wrap_compute_scatter_elements(topi.cuda.scatter_via_sort),
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
