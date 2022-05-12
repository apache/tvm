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
# pylint: disable=invalid-name
"""scheduler functions for cuda backend"""
from __future__ import absolute_import as _abs

import tvm
from tvm import te
from ..utils import traverse_inline


def schedule_lrn(outs):
    """Schedule for LRN

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of LRN
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    max_threads = int(tvm.target.Target.current(allow_none=False).max_num_threads)

    def _callback(op):
        if "sqr_sum" in op.tag:
            pad = op.input_tensors[0]
            s[pad].compute_inline()
            fused_axis = s[outs[0]].fuse(*s[outs[0]].op.axis)
            bx, tx = s[outs[0]].split(fused_axis, factor=max_threads)
            s[outs[0]].bind(bx, te.thread_axis("blockIdx.x"))
            s[outs[0]].bind(tx, te.thread_axis("threadIdx.x"))
            s[op].compute_at(s[outs[0]], tx)

    traverse_inline(s, outs[0].op, _callback)
    return s
