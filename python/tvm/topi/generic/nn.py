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
# pylint: disable=invalid-name,unused-argument
"""Generic nn operators"""
from tvm import te
from .default import default_schedule as _default_schedule


def schedule_conv1d_ncw(outs):
    """Schedule for conv1d_ncw

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of conv1d_ncw
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


def schedule_conv1d_nwc(outs):
    """Schedule for conv1d_nwc

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of conv1d_nwc
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


def schedule_group_conv1d_ncw(outs):
    """Schedule for group_conv1d_ncw

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of group_conv1d_ncw
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


def schedule_group_conv1d_nwc(outs):
    """Schedule for group_conv1d_nwc

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of group_conv1d_nwc
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


def schedule_conv2d_hwcn(outs):
    """Schedule for conv2d_hwcn

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of conv2d_hwcn
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


def schedule_conv2d_nchw(outs):
    """Schedule for conv2d_nchw

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of conv2d_nchw
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


def schedule_conv2d_nhwc_pack(outs):
    """Schedule for conv2d_nhwc_pack

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of conv2d_nhwc_pack
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


def schedule_conv2d_nhwc(outs):
    """Schedule for conv2d_nhwc

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of conv2d_nchw
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


def schedule_conv2d_NCHWc(outs):
    """Schedule for conv2d_NCHW[x]c

    Parameters
    ----------
    outs : Array of Tensor
        The computation graph description of conv2d_NCHWc
        in the format of an array of tensors.
        The number of filter, i.e., the output channel.

    Returns
    -------
    sch : Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


def schedule_conv2d_NCHWc_int8(outs):
    """Schedule for conv2d_NCHW[x]c_int8

    Parameters
    ----------
    outs : Array of Tensor
        The computation graph description of conv2d_NCHWc_int8
        in the format of an array of tensors.
        The number of filter, i.e., the output channel.

    Returns
    -------
    sch : Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


def schedule_conv2d_winograd_weight_transform(outs):
    """Schedule for weight transformation of winograd

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of this operator
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    # Typically this is computed in PreCompute pass
    # so we make a schedule here for cpu llvm
    s = te.create_schedule([x.op for x in outs])
    output = outs[0]
    _, G = s[output].op.input_tensors
    s[G].compute_inline()
    eps, nu, co, ci = s[output].op.axis
    r_kh, r_kw = s[output].op.reduce_axis
    s[output].reorder(co, ci, r_kh, r_kw, eps, nu)
    for axis in [r_kh, r_kw, eps, nu]:
        s[output].unroll(axis)
    s[output].parallel(co)
    return s


def schedule_conv2d_gemm_weight_transform(outs):
    """Schedule for weight transformation of gemm

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of this operator
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    # Typically this is computed in PreCompute pass
    s = te.create_schedule([x.op for x in outs])
    return s


def schedule_conv3d_winograd_weight_transform(outs):
    """Schedule for weight transformation of 3D winograd

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of this operator
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    # Typically this is computed in PreCompute pass
    # so we make a schedule here for cpu llvm
    s = te.create_schedule([x.op for x in outs])
    output = outs[0]
    _, G = s[output].op.input_tensors
    s[G].compute_inline()
    transform_depth = len(s[output].op.reduce_axis) == 3
    if transform_depth:
        omg, eps, nu, ci, co = s[output].op.axis
        r_kd, r_kh, r_kw = s[output].op.reduce_axis
        s[output].reorder(co, ci, omg, eps, nu, r_kd, r_kh, r_kw)
        for axis in [r_kd, r_kh, r_kw]:
            s[output].unroll(axis)
    else:
        eps, nu, d, ci, co = s[output].op.axis
        r_kh, r_kw = s[output].op.reduce_axis
        s[output].reorder(co, ci, d, eps, nu, r_kh, r_kw)
        for axis in [r_kh, r_kw]:
            s[output].unroll(axis)
    s[output].parallel(co)
    return s


def schedule_conv2d_winograd_without_weight_transform(outs):
    """Schedule for winograd without weight transformation

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of this operator
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


def schedule_conv2d_winograd_nnpack_weight_transform(outs):
    """Schedule for weight transformation of winograd
     Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of this operator
          in the format of an array of tensors.
     Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    # Typically this is computed in PreCompute pass
    s = te.create_schedule([x.op for x in outs])
    return s


def schedule_conv3d_ncdhw(outs):
    """Schedule for conv3d_ncdhw

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of conv2d_nchw
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


def schedule_conv3d_ndhwc(outs):
    """Schedule for conv3d_ndhwc

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of conv3d_ndhwc
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


def schedule_conv3d_transpose_ncdhw(outs):
    """Schedule for conv3d_transpose_ncdhw

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of conv3d_transpose_ncdhw
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


def schedule_conv2d_transpose_nchw(outs):
    """Schedule for conv2d_transpose_nchw

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of conv2d_transpose_nchw
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


def schedule_conv1d_transpose_ncw(outs):
    """Schedule for conv1d_transpose_ncw

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of conv2d_transpose_ncw
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


def schedule_depthwise_conv2d_nchw(outs):
    """Schedule for depthwise_conv2d_nchw

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of depthwise_conv2d_nchw
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


def schedule_depthwise_conv2d_nhwc(outs):
    """Schedule for depthwise_conv2d_nhwc
    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of depthwise_conv2d_nhwc
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


def schedule_depthwise_conv2d_NCHWc(outs):
    """Schedule for depthwise_conv2d_NCHWc
    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of depthwise_conv2d_nhwc
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


def schedule_group_conv2d_nchw(outs):
    """Schedule for group_conv2d_nchw

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of group_conv2d_nchw
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


def schedule_group_conv2d_transpose_nchw(outs):
    """Schedule for group_conv2d_transpose_nchw

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of group_conv2d_nhwc
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


def schedule_group_conv2d_nhwc(outs):
    """Schedule for group_conv2d_nhwc

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of group_conv2d_nhwc
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


def schedule_deformable_conv2d_nchw(outs):
    """Schedule for deformable_conv2d_nchw

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of deformable_conv2d_nchw
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


def schedule_deformable_conv2d_nhwc(outs):
    """Schedule for deformable_conv2d_nhwc.
    We only use the default schedule here and rely on auto_scheduler.

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of deformable_conv2d_nhwc
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


def schedule_bitserial_conv2d_nchw(outs):
    """Schedule for bitserial_conv2d_nchw

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of bitserial_conv2d_nchw
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


def schedule_bitserial_conv2d_nhwc(outs):
    """Schedule for bitserial_conv2d_nhwc

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of bitserial_conv2d_nchw
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


def schedule_bitserial_dense(outs):
    """Schedule for bitserial_dense
    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of bitserial_dense
          in the format of an array of tensors.
    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


def schedule_reduce(outs):
    """Schedule for reduction

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of reduce
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, True)


def schedule_softmax(outs):
    """Schedule for softmax

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of softmax
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


def schedule_fast_softmax(outs):
    """Schedule for fast_softmax

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of fast_softmax
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


def schedule_matmul(outs):
    """Schedule for matmul

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of matmul
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


def schedule_dense(outs):
    """Schedule for dense

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of dense
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


def schedule_pool(outs, layout):
    """Schedule for pool

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of pool
          in the format of an array of tensors.

    layout: str
        Data layout.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


def schedule_pool_grad(outs):
    """Schedule for pool_grad

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of pool
          in the format of an array of tensors.
    """
    return _default_schedule(outs, False)


def schedule_adaptive_pool(outs):
    """Schedule for adaptive pool

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of adaptive pool
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


def schedule_binarize_pack(outs):
    """Schedule for binarize_pack

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of binarize_pack
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


def schedule_bitpack(outs):
    """Schedule for bitpack
    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of bitpack
        in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


def schedule_binary_dense(outs):
    """Schedule for binary_dense

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of binary_dense
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


def schedule_lrn(outs):
    """Schedule for lrn

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of lrn
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


def schedule_sparse_dense(outs):
    """Schedule for sparse_dense

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of sparse_dense
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


def schedule_sparse_transpose(outs):
    """Schedule for sparse_transpose

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of sparse_transpose
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


def schedule_sparse_conv2d(outs):
    """Schedule for sparse_conv2d

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of sparse_conv2d
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


def schedule_batch_matmul(outs):
    """Schedule for batch_matmul

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of sparse_transpose
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


def schedule_batch_norm(outs):
    """Schedule for batch_norm

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of sparse_transpose
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


def schedule_correlation_nchw(outs):
    """Schedule for correlation_nchw

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of correlation_nchw
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)
