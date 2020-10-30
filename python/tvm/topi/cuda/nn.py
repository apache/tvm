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

from tvm import te
from .. import cpp


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
    return cpp.cuda.schedule_lrn(outs)


def loads_per_thread(dtype):
    """Number elements per load per thread"""
    s = regex.search("[0-9]+", dtype)
    assert s is not None
    byts = int(s.group()) // 8
    return 16 // byts


def schedule_embed_grad(outs):
    """Schedule for embed_grad

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of embed_grad
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    s = te.create_schedule([outs[0].op])
    # this should be autotuned, but we can't with hybrid script
    vec_size = loads_per_thread(outs[0].dtype)
    warp_size = int(tvm.target.Target.current(allow_none=False).thread_warp_size)
    num_warps = 4
    out = s.outputs[0].output(0)
    i, j = s[out].op.axis
    jo, ji = s[out].split(j, factor=vec_size)
    s[out].vectorize(ji)
    joo, joi = s[out].split(jo, factor=warp_size)
    s[out].bind(joi, te.thread_axis("threadIdx.x"))
    _, jooi = s[out].split(joo, factor=num_warps)
    s[out].bind(jooi, te.thread_axis("threadIdx.y"))
    s[out].bind(i, te.thread_axis("blockIdx.x"))

    return s
