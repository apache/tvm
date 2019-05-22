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
from __future__ import absolute_import as _abs
import tvm
from .. import cpp

def _default_schedule(outs, auto_inline):
    """Default schedule for llvm."""
    target = tvm.target.current_target(allow_none=False)
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    if target.target_name != "llvm":
        raise RuntimeError("schedule not registered for '%s'" % target)
    s = tvm.create_schedule([x.op for x in outs])
    if auto_inline:
        x = outs[0]
        tvm.schedule.AutoInlineInjective(s)
        s[x].fuse(s[x].op.axis)
    return s


@tvm.target.generic_func
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


@tvm.target.generic_func
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


@tvm.target.generic_func
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


@tvm.target.generic_func
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


@tvm.target.generic_func
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
    # Typically this is computed in nnvm PreCompute pass
    # so we make a schedule here for cpu llvm
    s = tvm.create_schedule([x.op for x in outs])
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


@tvm.target.generic_func
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


@tvm.target.generic_func
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
    # Typically this is computed in nnvm PreCompute pass
    s = tvm.create_schedule([x.op for x in outs])
    return s

@tvm.target.generic_func
def schedule_conv2d_winograd_nnpack_without_weight_transform(outs):
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


@tvm.target.generic_func
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


@tvm.target.generic_func
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


@tvm.target.generic_func
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


@tvm.target.generic_func
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


@tvm.target.generic_func
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


@tvm.target.generic_func
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


@tvm.target.generic_func
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


@tvm.target.generic_func
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


@tvm.target.generic_func
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


@tvm.target.override_native_generic_func("schedule_reduce")
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


@tvm.target.override_native_generic_func("schedule_softmax")
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


@tvm.target.override_native_generic_func("schedule_dense")
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


@tvm.target.override_native_generic_func("schedule_pool")
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


@tvm.target.override_native_generic_func("schedule_adaptive_pool")
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


@tvm.target.override_native_generic_func("schedule_binarize_pack")
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


@tvm.target.override_native_generic_func("schedule_binary_dense")
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


@tvm.target.generic_func
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
    target = tvm.target.current_target(allow_none=False)
    cpp_target = cpp.TEST_create_target(target.target_name)
    return cpp.generic.default_schedule(cpp_target, outs, False)

@tvm.target.generic_func
def schedule_l2_normalize(outs):
    """Schedule for l2 normalize

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of l2 normalize
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    target = tvm.target.current_target(allow_none=False)
    cpp_target = cpp.TEST_create_target(target.target_name)
    return cpp.generic.default_schedule(cpp_target, outs, False)

@tvm.target.generic_func
def schedule_batch_matmul(outs):
    target = tvm.target.current_target(allow_none=False)
    cpp_target = cpp.TEST_create_target(target.target_name)
    return cpp.generic.default_schedule(cpp_target, outs, False)
