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
"""HLS nn operators"""
from __future__ import absolute_import as _abs
import tvm
from tvm import te
from .. import tag


def _schedule_conv2d(outs):
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    tvm.te.schedule.AutoInlineInjective(s)

    def traverse(OP):
        """Internal traverse function"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_injective(OP.tag):
            if OP not in s.outputs:
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
                if isinstance(tensor.op, tvm.te.ComputeOp):
                    traverse(tensor.op)
        # schedule conv2d
        elif OP.tag.find("conv2d") >= 0:
            Conv2d = OP.output(0)
            if not Conv2d.op in s.outputs:
                Out = outs[0].op.output(0)
                s[Conv2d].compute_at(s[Out], s[Out].op.axis[1])
        else:
            raise RuntimeError(f"Unsupported operator: {OP.tag}")

    traverse(outs[0].op)

    px, x = s[outs[0]].split(outs[0].op.axis[0], nparts=1)
    s[outs[0]].bind(px, te.thread_axis("pipeline"))
    return s


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
    return _schedule_conv2d(outs)


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
    return _schedule_conv2d(outs)


def schedule_conv2d_NCHWc(outs):
    """Schedule for conv2d_NCHW[x]c

    Parameters
    ----------
    outs : Array of Tensor
        The computation graph description of conv2d_NCHWc
        in the format of an array of tensors.

    Returns
    -------
    sch : Schedule
        The computation schedule for the op.
    """
    return _schedule_conv2d(outs)


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
    return _schedule_conv2d(outs)


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
    return _schedule_conv2d(outs)


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
    return _schedule_conv2d(outs)


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
    return _schedule_conv2d(outs)


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
    return _schedule_conv2d(outs)


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
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    tvm.te.schedule.AutoInlineInjective(s)

    def traverse(OP):
        """Internal traverse function"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(OP.tag):
            if OP not in s.outputs:
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
                if isinstance(tensor.op, tvm.te.ComputeOp):
                    traverse(tensor.op)
        elif OP.tag in ["comm_reduce", "comm_reduce_idx"]:
            if OP.tag == "comm_reduce":
                Reduce = OP.output(0)
            else:
                Reduce = OP.input_tensors[0]
            if not Reduce.op in s.outputs:
                Out = outs[0].op.output(0)
                s[Reduce].compute_at(s[Out], s[Out].op.axis[0])
        else:
            raise RuntimeError(f"Unsupported operator: {OP.tag}")

    traverse(outs[0].op)

    fused = s[outs[0]].fuse()
    px, x = s[outs[0]].split(fused, nparts=1)
    s[outs[0]].bind(px, te.thread_axis("pipeline"))
    return s


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
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    tvm.te.schedule.AutoInlineInjective(s)

    softmax = outs[0]

    op_tag = softmax.op.tag
    if op_tag == "softmax_output":
        expsum = softmax.op.input_tensors[1]
        exp = softmax.op.input_tensors[0]
        max_elem = s[exp].op.input_tensors[1]
    elif op_tag == "log_softmax_output":
        exp = None
        max_elem = softmax.op.input_tensors[1]
        expsum = softmax.op.input_tensors[2]
    else:
        raise ValueError(
            f"Tag is expected to be softmax_output or log_softmax_output. Got {op_tag}"
        )

    if exp is not None:
        s[exp].compute_at(s[softmax], s[softmax].op.axis[1])

    s[expsum].compute_at(s[softmax], s[softmax].op.axis[1])
    s[max_elem].compute_at(s[softmax], s[softmax].op.axis[1])

    px, x = s[softmax].split(softmax.op.axis[0], nparts=1)
    s[softmax].bind(px, te.thread_axis("pipeline"))
    return s


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
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    tvm.te.schedule.AutoInlineInjective(s)

    def traverse(OP):
        """Internal traverse function"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(OP.tag):
            if OP not in s.outputs:
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
                if isinstance(tensor.op, tvm.te.ComputeOp):
                    traverse(tensor.op)
        # schedule dense
        elif OP.tag == "dense":
            Dense = OP.output(0)
            if not Dense.op in s.outputs:
                Out = outs[0].op.output(0)
                s[Dense].compute_at(s[Out], s[Out].op.axis[1])
        else:
            raise RuntimeError(f"Unsupported operator: {OP.tag}")

    traverse(outs[0].op)

    px, x = s[outs[0]].split(outs[0].op.axis[0], nparts=1)
    s[outs[0]].bind(px, te.thread_axis("pipeline"))
    return s


def schedule_pool(outs, layout):
    """Schedule for pool

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of pool
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    tvm.te.schedule.AutoInlineInjective(s)

    def traverse(OP):
        """Internal traverse function"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(OP.tag):
            if OP not in s.outputs:
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
                if isinstance(tensor.op, tvm.te.ComputeOp):
                    traverse(tensor.op)
        # schedule pool
        elif OP.tag.startswith("pool"):
            Pool = OP.output(0)
            if not Pool.op in s.outputs:
                Out = outs[0].op.output(0)
                s[Pool].compute_at(s[Out], s[Out].op.axis[1])
        else:
            raise RuntimeError(f"Unsupported operator: {OP.tag}")

    traverse(outs[0].op)

    px, x = s[outs[0]].split(outs[0].op.axis[0], nparts=1)
    s[outs[0]].bind(px, te.thread_axis("pipeline"))
    return s


def schedule_adaptive_pool(outs):
    """Schedule for adaptive_pool

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of adaptive_pool
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    tvm.te.schedule.AutoInlineInjective(s)

    def traverse(OP):
        """Internal traverse function"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(OP.tag):
            if OP not in s.outputs:
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
                if isinstance(tensor.op, tvm.te.ComputeOp):
                    traverse(tensor.op)
        # schedule global_pool
        elif OP.tag.startswith("adaptive_pool"):
            Pool = OP.output(0)
            if not Pool.op in s.outputs:
                Out = outs[0].op.output(0)
                s[Pool].compute_at(s[Out], s[Out].op.axis[1])
        else:
            raise RuntimeError(f"Unsupported operator: {OP.tag}")

    traverse(outs[0].op)

    px, x = s[outs[0]].split(outs[0].op.axis[0], nparts=1)
    s[outs[0]].bind(px, te.thread_axis("pipeline"))
    return s
