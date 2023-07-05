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
# pylint: disable=invalid-name,too-many-locals,unused-variable
"""x86 nn operators"""
from tvm import te
from ..utils import traverse_inline
from .injective import schedule_injective_from_existing


def _schedule_softmax(softmax_op, s, outs):
    op_tag = softmax_op.tag
    if op_tag == "softmax_output":
        exp = softmax_op.input_tensors[0]
        expsum = softmax_op.input_tensors[1]
        max_elem = s[exp].op.input_tensors[1]
        delta = None
        axis = int(softmax_op.attrs["axis"])
    elif op_tag == "fast_softmax_output":
        exp = softmax_op.input_tensors[0]
        expsum = softmax_op.input_tensors[1]
        delta = s[exp].op.input_tensors[0]
        max_elem = s[delta].op.input_tensors[1]
        axis = int(softmax_op.attrs["axis"])
    elif op_tag == "log_softmax_output":
        exp = None
        delta = None
        max_elem = softmax_op.input_tensors[1]
        expsum = softmax_op.input_tensors[2]
        axis = int(softmax_op.attrs["axis"])
    else:
        raise ValueError(
            f"Tag is expected to be softmax_output or log_softmax_output. Got {op_tag}"
        )

    output = outs[0]

    def _schedule(output_op, softmax_op):
        # only parallelize outer dimensions up to axis
        outer_axes = [output_op.axis[i] for i in range(0, axis)]
        fused_outer_axes = s[output_op].fuse(*outer_axes)
        s[output_op].parallel(fused_outer_axes)

        if softmax_op != output_op:
            # fuse softmax output with following elemwise ops.
            s[softmax_op].compute_at(s[output_op], fused_outer_axes)

        # move computations with the same outer dimensions under the same root
        s[max_elem].compute_at(s[output_op], fused_outer_axes)
        s[expsum].compute_at(s[output_op], fused_outer_axes)

        if delta is not None:
            s[exp].compute_inline()
            s[delta].compute_inline()
        if exp is not None:
            s[exp].compute_at(s[output_op], fused_outer_axes)

    if list(output.shape) == list(softmax_op.output(0).shape):
        _schedule(output.op, softmax_op)
    else:
        # This case can happen, for example, if the 4D input to softmax
        # is in the NCHW layout while the fused elemwise op takes the NCHWc layout.
        # Since we parallelize over outer axes up to the "axis" parameter of softmax,
        # softmax and the fused op need to be in the same layout if we want to
        # fuse them under the same parallel loop.
        # This case can be removed if softmax supported AlterLayout.
        schedule_injective_from_existing(s, output)
        _schedule(softmax_op, softmax_op)


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

    def _callback(op):
        if "softmax" in op.tag:
            _schedule_softmax(op, s, outs)

    traverse_inline(s, outs[0].op, _callback)
    return s


def schedule_batch_norm(outs):
    """Schedule for batch_norm

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of batch_norm
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    s = te.create_schedule([x.op for x in outs])
    # only parallelize outer dimensions up to axis
    output_op = outs[0].op
    axis = output_op.axis
    outer_axes = [output_op.axis[i] for i in range(0, len(axis) - 1)]
    fused_outer_axes = s[output_op].fuse(*outer_axes)
    s[output_op].parallel(fused_outer_axes)
    # when scale or center is enabled
    if "divide" not in output_op.name:
        div = output_op.input_tensors[0]
        substract = s[div].op.input_tensors[0]
        s[div].compute_inline()
        s[substract].compute_inline()
    return s
