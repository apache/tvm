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
"""x86 batch_matmul operators"""
from __future__ import absolute_import as _abs
import tvm
from tvm.contrib import cblas
from topi.nn import batch_matmul, batch_matmul_default
from .. import generic
from ..util import traverse_inline, get_const_tuple, get_max_power2_factor

@batch_matmul.register(["cpu"])
def batch_matmul_x86(x, y):
    """Computes batch matrix multiplication of `x` and `y` when `x` and `y` are
    data in batch.

    Parameters
    ----------
    x : tvm.Tensor
        3-D with shape [batch, M, K]

    y : tvm.Tensor
        3-D with shape [batch, N, K]

    Returns
    -------
    output : tvm.Tensor
        3-D with shape [batch, M, N]
    """
    target = tvm.target.current_target()
    if "cblas" in target.libs:
        return cblas.batch_matmul(x, y, False, True)
    return batch_matmul_default(x, y)

@generic.schedule_batch_matmul.register(["cpu"])
def schedule_batch_matmul(outs):
    """Schedule for batch_matmul

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of batch_matmul
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    target = tvm.target.current_target()
    if "cblas" in target.libs:
        return generic.schedule_extern(outs)

    s = tvm.create_schedule([x.op for x in outs])

    def _callback(op):
        if "batch_matmul" in op.tag:
            C = op.output(0)
            A, B = s[C].op.input_tensors
            _, M, N = get_const_tuple(C.shape)
            k, = s[C].op.reduce_axis
            ko, ki = s[C].split(k, 16)
            CC = s.rfactor(C, ki)

            b, y, x = s[C].op.axis
            y_bn = get_max_power2_factor(M, 8)
            x_bn = get_max_power2_factor(N, 8)
            yo, yi = s[C].split(y, y_bn)
            xo, xi = s[C].split(x, x_bn)
            s[C].reorder(b, yo, xo, yi, xi)
            bxyo = s[C].fuse(b, yo, xo)
            s[C].parallel(bxyo)
            s[C].fuse(yi, xi)

            s[CC].compute_at(s[C], bxyo)
            _, _, y, x = s[CC].op.axis
            s[CC].fuse(y, x)
            s[CC].vectorize(s[CC].op.axis[0])
            s[C].pragma(bxyo, 'auto_unroll_max_step', 16)

    traverse_inline(s, outs[0].op, _callback)
    return s
