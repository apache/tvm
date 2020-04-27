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
"""Definition of CUDA/GPU operator strategy."""
# pylint: disable=invalid-name,unused-argument,wildcard-import,unused-wildcard-import
import topi
import tvm
from tvm.te import SpecializedCondition
from tvm.contrib import nvcc
from .generic import *
from .. import op as _op
from .... import get_global_func

@schedule_injective.register(["cuda", "gpu"])
def schedule_injective_cuda(attrs, outs, target):
    """schedule injective ops for cuda"""
    with target:
        return topi.cuda.schedule_injective(outs)

@schedule_reduce.register(["cuda", "gpu"])
def schedule_reduce_cuda(attrs, outs, target):
    """schedule reduction ops for cuda"""
    with target:
        return topi.cuda.schedule_reduce(outs)

@schedule_concatenate.register(["cuda", "gpu"])
def schedule_concatenate_cuda(attrs, outs, target):
    """schedule concatenate for cuda"""
    with target:
        return topi.cuda.schedule_injective(outs)

@schedule_pool.register(["cuda", "gpu"])
def schedule_pool_cuda(attrs, outs, target):
    """schedule pooling ops for cuda"""
    with target:
        return topi.cuda.schedule_pool(outs, attrs.layout)

@schedule_pool_grad.register(["cuda", "gpu"])
def schedule_pool_grad_cuda(attrs, outs, target):
    """schedule pooling gradient ops for cuda"""
    with target:
        return topi.cuda.schedule_pool_grad(outs)

@schedule_adaptive_pool.register(["cuda", "gpu"])
def schedule_adaptive_pool_cuda(attrs, outs, target):
    """schedule adaptive pooling ops for cuda"""
    with target:
        return topi.cuda.schedule_adaptive_pool(outs)

@softmax_strategy.register(["cuda", "gpu"])
def softmax_strategy_cuda(attrs, inputs, out_type, target):
    """softmax cuda strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_softmax(topi.nn.softmax),
        wrap_topi_schedule(topi.cuda.schedule_softmax),
        name="softmax.cuda")
    if target.target_name == "cuda" and "cudnn" in target.libs:
        strategy.add_implementation(
            wrap_compute_softmax(topi.cuda.softmax_cudnn),
            wrap_topi_schedule(topi.cuda.schedule_softmax_cudnn),
            name="softmax.cudnn",
            plevel=15)
    return strategy

@schedule_log_softmax.register(["cuda", "gpu"])
def schedule_log_softmax_cuda(attrs, outs, target):
    """scheudle log_softmax for cuda"""
    with target:
        return topi.cuda.schedule_softmax(outs)

@schedule_lrn.register(["cuda", "gpu"])
def schedule_lrn_cuda(attrs, outs, target):
    """schedule LRN for cuda"""
    with target:
        return topi.cuda.schedule_lrn(outs)

@conv2d_strategy.register(["cuda", "gpu"])
def conv2d_strategy_cuda(attrs, inputs, out_type, target):
    """conv2d cuda strategy"""
    strategy = _op.OpStrategy()
    data, kernel = inputs
    stride_h, stride_w = attrs.get_int_tuple("strides")
    dilation_h, dilation_w = attrs.get_int_tuple("dilation")
    padding = attrs.get_int_tuple("padding")
    groups = attrs.groups
    layout = attrs.data_layout
    kernel_layout = attrs.kernel_layout
    if dilation_h < 1 or dilation_w < 1:
        raise ValueError("dilation should be positive value")

    if groups == 1:
        if layout == "NCHW":
            assert kernel_layout == "OIHW"
            if data.dtype in ('int8', 'uint8') and kernel.dtype in ('int8', 'uint8'):
                assert data.dtype == kernel.dtype
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.cuda.conv2d_nchw_int8),
                    wrap_topi_schedule(topi.cuda.schedule_conv2d_nchw_int8),
                    name="conv2d_nchw_int8.cuda")
            else:
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
        elif layout == "NHWC":
            assert kernel_layout == "HWIO"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.cuda.conv2d_nhwc),
                wrap_topi_schedule(topi.cuda.schedule_conv2d_nhwc),
                name="conv2d_nhwc.cuda")
            N, _, _, _ = get_const_tuple(data.shape)
            _, _, CI, CO = get_const_tuple(kernel.shape)
            if target.target_name == "cuda":
                if nvcc.have_tensorcore(tvm.gpu(0).compute_version):
                    if (N % 16 == 0 and CI % 16 == 0 and CO % 16 == 0) or \
                            (N % 8 == 0 and CI % 16 == 0 and CO % 32 == 0) or \
                            (N % 32 == 0 and CI % 16 == 0 and CO % 8 == 0):
                        strategy.add_implementation(
                            wrap_compute_conv2d(topi.cuda.conv2d_nhwc_tensorcore),
                            wrap_topi_schedule(topi.cuda.schedule_conv2d_nhwc_tensorcore),
                            name="conv2d_nhwc_tensorcore.cuda",
                            plevel=20)
        elif layout == "NCHW4c" and data.dtype in ["int8", "uint8"]:
            assert kernel_layout == "OIHW4o4i"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.cuda.conv2d_NCHWc_int8, True),
                wrap_topi_schedule(topi.cuda.schedule_conv2d_NCHWc_int8),
                name="conv2d_NCHWc_int8.cuda")
        else:
            raise RuntimeError("Unsupported conv2d layout {} for CUDA".format(layout))
        # add cudnn implementation
        if target.target_name == "cuda" and "cudnn" in target.libs:
            if layout in ["NCHW", "NHWC"] and padding[0] == padding[2] and \
                    padding[1] == padding[3]:
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.cuda.conv2d_cudnn,
                                        need_data_layout=True,
                                        has_groups=True),
                    wrap_topi_schedule(topi.cuda.schedule_conv2d_cudnn),
                    name="conv2d_cudnn.cuda",
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
        # add cudnn implementation, if any
        cudnn_impl = False
        if target.target_name == "cuda" and "cudnn" in target.libs:
            if layout in ["NCHW", "NHWC"] and padding[0] == padding[2] and \
                    padding[1] == padding[3]:
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.cuda.conv2d_cudnn,
                                        need_data_layout=True,
                                        has_groups=True),
                    wrap_topi_schedule(topi.cuda.schedule_conv2d_cudnn),
                    name="conv2d_cudnn.cuda",
                    plevel=15)
                cudnn_impl = True

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
        elif not cudnn_impl:
            raise RuntimeError("Unsupported group_conv2d layout {}".format(layout))
    return strategy

@conv2d_winograd_without_weight_transfrom_strategy.register(["cuda", "gpu"])
def conv2d_winograd_without_weight_transfrom_strategy_cuda(attrs, inputs, out_type, target):
    """conv2d_winograd_without_weight_transfrom cuda strategy"""
    dilation = attrs.get_int_tuple("dilation")
    groups = attrs.get_int("groups")
    layout = attrs.data_layout
    assert dilation == (1, 1), "Do not support dilate now"
    assert groups == 1, "Do not supoort arbitrary group number"
    strategy = _op.OpStrategy()
    if layout == "NCHW":
        strategy.add_implementation(
            wrap_compute_conv2d(topi.cuda.conv2d_nchw_winograd_without_weight_transform),
            wrap_topi_schedule(
                topi.cuda.schedule_conv2d_nchw_winograd_without_weight_transform),
            name="conv2d_nchw_winograd_without_weight_transform.cuda")
    else:
        raise RuntimeError("Unsupported conv2d_winograd_without_weight_transfrom layout {}".
                           format(layout))
    return strategy

@deformable_conv2d_strategy.register(["cuda", "gpu"])
def deformable_conv2d_strategy_cuda(attrs, inputs, out_type, target):
    """deformable_conv2d cuda strategy"""
    layout = attrs.data_layout
    assert layout == "NCHW"
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_deformable_conv2d(topi.cuda.deformable_conv2d_nchw),
        wrap_topi_schedule(topi.cuda.schedule_deformable_conv2d_nchw),
        name="deformable_conv2d_nchw.cuda")
    return strategy

@conv2d_transpose_strategy.register(["cuda", "gpu"])
def conv2d_transpose_strategy_cuda(attrs, inputs, out_type, target):
    """conv2d_transpose cuda strategy"""
    layout = attrs.data_layout
    dilation = get_const_tuple(attrs.dilation)
    groups = attrs.groups
    assert layout == "NCHW", "only support nchw for now"
    assert dilation == (1, 1), "not support dilate now"
    assert groups == 1, "only support groups == 1 for now"
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_conv2d_transpose(topi.cuda.conv2d_transpose_nchw),
        wrap_topi_schedule(topi.cuda.schedule_conv2d_transpose_nchw),
        name="conv2d_transpose_nchw.cuda")
    return strategy

@conv3d_strategy.register(["cuda", "gpu"])
def conv3d_strategy_cuda(attrs, inputs, out_type, target):
    """conv3d cuda strategy"""
    strategy = _op.OpStrategy()
    data, kernel = inputs
    layout = attrs.data_layout
    _, stride_h, stride_w = attrs.get_int_tuple("strides")
    _, dilation_h, dilation_w = attrs.get_int_tuple("dilation")
    assert layout in ["NCDHW", "NDHWC"], "Not support this layout {} yet".format(layout)
    if layout == "NCDHW":
        strategy.add_implementation(wrap_compute_conv3d(topi.cuda.conv3d_ncdhw),
                                    wrap_topi_schedule(topi.cuda.schedule_conv3d_ncdhw),
                                    name="conv3d_ncdhw.cuda",
                                    plevel=10)
        _, _, _, kh, kw = get_const_tuple(kernel.shape)
        if 2 < kh < 8 and 2 < kw < 8 and kh == kw and \
            stride_h == 1 and stride_w == 1 and \
            dilation_h == 1 and dilation_w == 1:
            strategy.add_implementation(
                wrap_compute_conv3d(topi.cuda.conv3d_ncdhw_winograd),
                wrap_topi_schedule(topi.cuda.schedule_conv3d_ncdhw_winograd),
                name="conv3d_ncdhw_winograd.cuda",
                plevel=5)
    else:  # layout == "NDHWC":
        strategy.add_implementation(
            wrap_compute_conv3d(topi.cuda.conv3d_ndhwc),
            wrap_topi_schedule(topi.cuda.schedule_conv3d_ndhwc),
            name="conv3d_ndhwc.cuda",
            plevel=10)
        N, _, _, _, _ = get_const_tuple(data.shape)
        _, _, _, CI, CO = get_const_tuple(kernel.shape)
        if target.target_name == "cuda":
            if nvcc.have_tensorcore(tvm.gpu(0).compute_version):
                if (N % 16 == 0 and CI % 16 == 0 and CO % 16 == 0) or \
                (N % 8 == 0 and CI % 16 == 0 and CO % 32 == 0) or \
                (N % 32 == 0 and CI % 16 == 0 and CO % 8 == 0):
                    strategy.add_implementation(
                        wrap_compute_conv3d(topi.cuda.conv3d_ndhwc_tensorcore),
                        wrap_topi_schedule(topi.cuda.schedule_conv3d_ndhwc_tensorcore),
                        name="conv3d_ndhwc_tensorcore.cuda",
                        plevel=20)

    if target.target_name == "cuda" and "cudnn" in target.libs:
        strategy.add_implementation(wrap_compute_conv3d(topi.cuda.conv3d_cudnn, True),
                                    wrap_topi_schedule(topi.cuda.schedule_conv3d_cudnn),
                                    name="conv3d_cudnn.cuda",
                                    plevel=15)
    return strategy

@conv3d_winograd_without_weight_transfrom_strategy.register(["cuda", "gpu"])
def conv3d_winograd_without_weight_transfrom_strategy_cuda(attrs, inputs, out_type, target):
    """conv3d_winograd_without_weight_transfrom cuda strategy"""
    dilation = attrs.get_int_tuple("dilation")
    groups = attrs.get_int("groups")
    layout = attrs.data_layout
    assert dilation == (1, 1, 1), "Do not support dilate now"
    assert groups == 1, "Do not supoort arbitrary group number"
    strategy = _op.OpStrategy()
    if layout == "NCDHW":
        strategy.add_implementation(
            wrap_compute_conv3d(topi.cuda.conv3d_ncdhw_winograd_without_weight_transform),
            wrap_topi_schedule(
                topi.cuda.schedule_conv3d_ncdhw_winograd_without_weight_transform),
            name="conv3d_ncdhw_winograd_without_weight_transform.cuda")
    else:
        raise RuntimeError("Unsupported conv3d_winograd_without_weight_transfrom layout {}".
                           format(layout))
    return strategy

@conv1d_strategy.register(["cuda", "gpu"])
def conv1d_strategy_cuda(attrs, inputs, out_type, target):
    """conv1d cuda strategy"""
    layout = attrs.data_layout
    dilation = get_const_tuple(attrs.dilation)
    if dilation[0] < 1:
        raise ValueError("dilation should be a positive value")
    strategy = _op.OpStrategy()
    if layout == "NCW":
        strategy.add_implementation(wrap_compute_conv1d(topi.cuda.conv1d_ncw),
                                    wrap_topi_schedule(topi.cuda.schedule_conv1d_ncw),
                                    name="conv1d_ncw.cuda")
    elif layout == "NWC":
        strategy.add_implementation(wrap_compute_conv1d(topi.cuda.conv1d_nwc),
                                    wrap_topi_schedule(topi.cuda.schedule_conv1d_nwc),
                                    name="conv1d_nwc.cuda")
    else:
        raise ValueError("Unsupported conv1d layout {}".format(layout))
    return strategy

@conv1d_transpose_strategy.register(["cuda", "gpu"])
def conv1d_transpose_strategy_cuda(attrs, inputs, out_type, target):
    """conv1d_transpose cuda strategy"""
    strategy = _op.OpStrategy()
    layout = attrs.data_layout
    dilation = get_const_tuple(attrs.dilation)
    groups = attrs.groups
    assert layout == "NCW", "conv1d_transpose ncw only supported"
    assert dilation == (1,), "conv1d_transpose dilation is not supported"
    assert groups == 1, "conv1d_transpose groups == 1 only supported"
    strategy.add_implementation(wrap_compute_conv1d_transpose(topi.cuda.conv1d_transpose_ncw),
                                wrap_topi_schedule(topi.cuda.schedule_conv1d_transpose_ncw),
                                name="conv1d_transpose_ncw.cuda")
    return strategy

@dense_strategy.register(["cuda", "gpu"])
def dense_strategy_cuda(attrs, inputs, out_type, target):
    """dense cuda strategy"""
    strategy = _op.OpStrategy()
    data, weights = inputs
    b, i = get_const_tuple(data.shape)
    o, _ = get_const_tuple(weights.shape)
    if out_type.dtype == "int8":
        strategy.add_implementation(
            wrap_compute_dense(topi.cuda.dense_int8),
            wrap_topi_schedule(topi.cuda.schedule_dense_int8),
            name="dense_int8.cuda")
    else:
        strategy.add_implementation(
            wrap_compute_dense(topi.cuda.dense_small_batch),
            wrap_topi_schedule(topi.cuda.schedule_dense_small_batch),
            name="dense_small_batch.cuda")
        with SpecializedCondition(b >= 32):
            strategy.add_implementation(
                wrap_compute_dense(topi.cuda.dense_large_batch),
                wrap_topi_schedule(topi.cuda.schedule_dense_large_batch),
                name="dense_large_batch.cuda",
                plevel=5)
        if target.target_name == "cuda":
            if nvcc.have_tensorcore(tvm.gpu(0).compute_version):
                if(i % 16 == 0 and b % 16 == 0 and o % 16 == 0) \
                        or (i % 16 == 0 and b % 8 == 0 and o % 32 == 0) \
                        or (i % 16 == 0 and b % 32 == 0 and o % 8 == 0):
                    strategy.add_implementation(
                        wrap_compute_dense(topi.cuda.dense_tensorcore),
                        wrap_topi_schedule(topi.cuda.schedule_dense_tensorcore),
                        name="dense_tensorcore.cuda",
                        plevel=20)
    if target.target_name == "cuda" and "cublas" in target.libs:
        strategy.add_implementation(
            wrap_compute_dense(topi.cuda.dense_cublas),
            wrap_topi_schedule(topi.cuda.schedule_dense_cublas),
            name="dense_cublas.cuda",
            plevel=15)
    return strategy

@batch_matmul_strategy.register(["cuda", "gpu"])
def batch_matmul_strategy_cuda(attrs, inputs, out_type, target):
    """batch_matmul cuda strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_batch_matmul(topi.nn.batch_matmul),
        wrap_topi_schedule(topi.cuda.schedule_batch_matmul),
        name="batch_matmul.cuda",
        plevel=10)
    if target.target_name == "cuda" and "cublas" in target.libs:
        strategy.add_implementation(
            wrap_compute_batch_matmul(topi.cuda.batch_matmul_cublas),
            wrap_topi_schedule(topi.generic.schedule_extern),
            name="batch_matmul_cublas.cuda",
            plevel=15)
    return strategy

@argsort_strategy.register(["cuda", "gpu"])
def argsort_strategy_cuda(attrs, inputs, out_type, target):
    """argsort cuda strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_argsort(topi.cuda.argsort),
        wrap_topi_schedule(topi.cuda.schedule_argsort),
        name="argsort.cuda")
    if get_global_func("tvm.contrib.thrust.sort", allow_missing=True):
        strategy.add_implementation(wrap_compute_argsort(topi.cuda.argsort_thrust),
                                    wrap_topi_schedule(topi.cuda.schedule_argsort),
                                    name="argsort_thrust.cuda",
                                    plevel=15)
    return strategy

@topk_strategy.register(["cuda", "gpu"])
def topk_strategy_cuda(attrs, inputs, out_type, target):
    """topk cuda strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(wrap_compute_topk(topi.cuda.topk),
                                wrap_topi_schedule(topi.cuda.schedule_topk),
                                name="topk.cuda")
    if get_global_func("tvm.contrib.thrust.sort", allow_missing=True):
        strategy.add_implementation(wrap_compute_topk(topi.cuda.topk_thrust),
                                    wrap_topi_schedule(topi.cuda.schedule_topk),
                                    name="topk_thrust.cuda",
                                    plevel=15)
    return strategy

@multibox_prior_strategy.register(["cuda", "gpu"])
def multibox_prior_strategy_cuda(attrs, inputs, out_type, target):
    """multibox_prior cuda strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_multibox_prior(topi.cuda.multibox_prior),
        wrap_topi_schedule(topi.cuda.schedule_multibox_prior),
        name="multibox_prior.cuda")
    return strategy

@multibox_transform_loc_strategy.register(["cuda", "gpu"])
def multibox_transform_loc_strategy_cuda(attrs, inputs, out_type, target):
    """multibox_transform_loc cuda strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_multibox_transform_loc(topi.cuda.multibox_transform_loc),
        wrap_topi_schedule(topi.cuda.schedule_multibox_transform_loc),
        name="multibox_transform_loc.cuda")
    return strategy

@get_valid_counts_strategy.register(["cuda", "gpu"])
def get_valid_counts_strategy_cuda(attrs, inputs, out_type, target):
    """get_valid_counts cuda strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_get_valid_counts(topi.cuda.get_valid_counts),
        wrap_topi_schedule(topi.cuda.schedule_get_valid_counts),
        name="get_valid_counts.cuda")
    return strategy

@nms_strategy.register(["cuda", "gpu"])
def nms_strategy_cuda(attrs, inputs, out_type, target):
    """nms cuda strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_nms(topi.cuda.non_max_suppression),
        wrap_topi_schedule(topi.cuda.schedule_nms),
        name="nms.cuda")
    return strategy

@roi_align_strategy.register(["cuda", "gpu"])
def roi_align_strategy_cuda(attrs, inputs, out_type, target):
    """roi_align cuda strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(wrap_compute_roi_align(topi.vision.rcnn.roi_align_nchw),
                                wrap_topi_schedule(topi.cuda.schedule_roi_align),
                                name="roi_align_nchw.cuda")
    return strategy

@schedule_roi_pool.register(["cuda", "gpu"])
def schedule_roi_pool_cuda(attrs, outs, target):
    """schedule roi_pool for cuda"""
    with target:
        return topi.cuda.schedule_roi_pool(outs)

@proposal_strategy.register(["cuda", "gpu"])
def proposal_strategy_cuda(attrs, inputs, out_type, target):
    """proposal cuda strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(wrap_compute_proposal(topi.cuda.proposal),
                                wrap_topi_schedule(topi.cuda.schedule_proposal),
                                name="proposal.cuda")
    return strategy
