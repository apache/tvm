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
"""cuda batch_matmul operators"""
from __future__ import absolute_import as _abs
import tvm

from .. import generic
from ..util import traverse_inline, get_const_tuple, get_max_power2_factor


@generic.schedule_batch_matmul.register(["cuda", "gpu"])
def schedule_batch_matmul(outs):
    """Schedule for batch_matmul

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of batch_matmul
          in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for the op.
    """
    s = tvm.create_schedule([x.op for x in outs])

    def _schedule(op):
        C = op.output(0)
        A, B = s[C].op.input_tensors
        _, M, N = get_const_tuple(C.shape)
        AA = s.cache_read(A, "shared", [C])
        AL = s.cache_read(AA, "local", [C])
        BB = s.cache_read(B, "shared", [C])
        BL = s.cache_read(BB, "local", [C])
        CC = s.cache_write(C, "local")

        b, y, x = s[C].op.axis
        y_bn = get_max_power2_factor(M, 64)
        x_bn = get_max_power2_factor(N, 64)
        by, y = s[C].split(y, y_bn)
        bx, x = s[C].split(x, x_bn)
        y_nthreads = min(y_bn, 8)
        x_nthreads = min(x_bn, 8)
        ty, yi = s[C].split(y, nparts=y_nthreads)
        tx, xi = s[C].split(x, nparts=x_nthreads)
        thread_x = tvm.thread_axis((0, x_nthreads), "threadIdx.x")
        thread_y = tvm.thread_axis((0, y_nthreads), "threadIdx.y")

        s[C].reorder(b, by, bx, ty, tx, yi, xi)
        s[C].bind(b, tvm.thread_axis("blockIdx.z"))
        s[C].bind(by, tvm.thread_axis("blockIdx.y"))
        s[C].bind(bx, tvm.thread_axis("blockIdx.x"))
        s[C].bind(ty, thread_y)
        s[C].bind(tx, thread_x)
        s[C].pragma(yi, "auto_unroll_max_step", 16)

        s[CC].compute_at(s[C], tx)
        _, yi, xi = s[CC].op.axis
        k, = s[CC].op.reduce_axis
        ko, ki = s[CC].split(k, 8)
        s[CC].reorder(ko, ki, yi, xi)
        s[CC].pragma(ki, "auto_unroll_max_step", 16)

        s[AA].compute_at(s[CC], ko)
        s[AL].compute_at(s[CC], ki)
        s[BB].compute_at(s[CC], ko)
        s[BL].compute_at(s[CC], ki)
        _, y, k = s[AA].op.axis
        ty, yi = s[AA].split(y, nparts=y_nthreads)
        tx, ki = s[AA].split(k, nparts=x_nthreads)
        s[AA].reorder(ty, tx, yi, ki)
        s[AA].bind(ty, thread_y)
        s[AA].bind(tx, thread_x)
        s[AA].pragma(yi, "auto_unroll_max_step", 16)

        _, x, k = s[BB].op.axis
        ty, xi = s[BB].split(x, nparts=y_nthreads)
        tx, ki = s[BB].split(k, nparts=x_nthreads)
        s[BB].bind(ty, thread_y)
        s[BB].bind(tx, thread_x)
        s[BB].reorder(ty, tx, xi, ki)
        s[BB].pragma(xi, "auto_unroll_max_step", 16)

    def _callback(op):
        if "batch_matmul" in op.tag:
            _schedule(op)

    traverse_inline(s, outs[0].op, _callback)
    return s
