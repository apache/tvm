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
from __future__ import absolute_import as _abs
import tvm
from .. import generic

@generic.schedule_softmax.register(["cpu"])
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
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    softmax = outs[0]
    s = tvm.create_schedule([x.op for x in outs])

    op_tag = softmax.op.tag
    if op_tag == 'softmax_output':
        exp = softmax.op.input_tensors[0]
        expsum = softmax.op.input_tensors[1]
        max_elem = s[exp].op.input_tensors[1]
        axis = int(softmax.op.attrs['axis'])
    elif op_tag == 'log_softmax_output':
        exp = None
        max_elem = softmax.op.input_tensors[1]
        expsum = softmax.op.input_tensors[2]
        axis = 1
    else:
        raise ValueError('Tag is expected to be softmax_output or log_softmax_output. \
                         Got {0}'.format(op_tag))

    # only parallelize outer dimensions up to axis
    outer_axes = [s[softmax].op.axis[i] for i in range(0, axis)]
    fused_outer_axes = s[softmax].fuse(*outer_axes)
    s[softmax].parallel(fused_outer_axes)

    # move computations with the same outer dimensions under the same root
    s[max_elem].compute_at(s[softmax], fused_outer_axes)
    s[expsum].compute_at(s[softmax], fused_outer_axes)

    if exp != None:
        s[exp].compute_at(s[softmax], fused_outer_axes)

    return s
