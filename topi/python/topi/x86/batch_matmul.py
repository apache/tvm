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

from tvm import autotvm
from tvm.autotvm.task.space import SplitEntity
from .. import generic, nn
from ..util import traverse_inline, get_const_tuple, get_max_power2_factor

def _default_dense_pack_config(cfg, M, N, K):
    tile_y = get_max_power2_factor(M, 8)
    tile_x = get_max_power2_factor(N, 8)
    tile_k = get_max_power2_factor(N, 16)

    cfg["tile_y"] = SplitEntity([M // tile_y, tile_y])
    cfg["tile_x"] = SplitEntity([N // tile_x, tile_x])
    cfg["tile_k"] = SplitEntity([K // tile_k, tile_k])


@autotvm.register_topi_compute(nn.batch_matmul, "cpu", "direct")
def _decl(cfg, x, y):
    return nn.batch_matmul(x, y)


@autotvm.register_topi_schedule(generic.schedule_batch_matmul, 'cpu', ['direct'])
def schedule_batch_matmul(cfg, outs):
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
    s = tvm.create_schedule([x.op for x in outs])

    def _callback(op):
        if "batch_matmul" in op.tag:
            C = op.output(0)
            A = s[C].op.input_tensors[0]
            _, M, N = get_const_tuple(C.shape)
            _, _, K = get_const_tuple(A.shape)
            b, y, x = s[C].op.axis
            k, = s[C].op.reduce_axis
            cfg.define_split("tile_k", k, num_outputs=2,
                             filter=lambda item: item.size[-1] <= 64)
            cfg.define_split("tile_y", y, num_outputs=2,
                             filter=lambda item: item.size[-1] <= 32)
            cfg.define_split("tile_x", x, num_outputs=2,
                             filter=lambda item: item.size[-1] <= 32)

            if cfg.is_fallback:
                _default_dense_pack_config(cfg, M, N, K)

            ko, ki = cfg["tile_k"].apply(s, C, k)
            CC = s.rfactor(C, ki)
            yo, yi = cfg["tile_y"].apply(s, C, y)
            xo, xi = cfg["tile_x"].apply(s, C, x)
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
