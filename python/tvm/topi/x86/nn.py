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
        axis = 1
    else:
        raise ValueError(
            "Tag is expected to be softmax_output or log_softmax_output. \
                         Got {0}".format(
                op_tag
            )
        )

    # only parallelize outer dimensions up to axis
    outer_axes = [s[softmax_op].op.axis[i] for i in range(0, axis)]
    fused_outer_axes = s[softmax_op].fuse(*outer_axes)
    s[softmax_op].parallel(fused_outer_axes)

    # move computations with the same outer dimensions under the same root
    s[max_elem].compute_at(s[softmax_op], fused_outer_axes)
    s[expsum].compute_at(s[softmax_op], fused_outer_axes)

    if delta is not None:
        s[exp].compute_inline()
        s[delta].compute_inline()
    if exp is not None:
        s[exp].compute_at(s[softmax_op], fused_outer_axes)

    if softmax_op != outs[0].op:
        # fuse softmax output with following elemwise ops.
        output = outs[0]
        outer_axes = [s[output].op.axis[i] for i in range(0, axis)]
        fused_outer_axes = s[output].fuse(*outer_axes)
        s[output].parallel(fused_outer_axes)
        s[softmax_op].compute_at(s[output], fused_outer_axes)


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
