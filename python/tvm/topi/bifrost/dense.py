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
# pylint: disable=invalid-name,unused-variable
"""dense schedule on ARM Mali Biforst GPU"""
from tvm import te
from tvm import autotvm

from .. import nn
from ..utils import traverse_inline


@autotvm.register_topi_compute("dense.bifrost")
def dense(_, data, weight, bias=None, out_dtype=None):
    """Dense operator on Biforst"""
    return nn.dense(data, weight, bias, out_dtype)


@autotvm.register_topi_schedule("dense.bifrost")
def schedule_dense(cfg, outs):
    """Schedule for dense operator.

    Parameters
    ----------
    cfg: ConfigEntity
        The config entity for this template
    outs: Array of Tensor
        The computation graph description of dense
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for dense.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == "dense":
            vec_size = [1, 2, 4, 8, 16]
            max_unroll = 32

            dense_out = op.output(0)
            output = outs[0]

            y, x = s[output].op.axis
            c = s[dense_out].op.reduce_axis[0]

            ##### space definition begin #####
            cfg.define_split("tile_y", y, num_outputs=3)
            cfg.define_split("tile_x", x, num_outputs=3)
            cfg.define_split("c_unroll", c, num_outputs=2, max_factor=64)

            # fallback support
            if cfg.is_fallback:
                ref_log = autotvm.tophub.load_reference_log("mali", "rk3399", "dense.bifrost")
                cfg.fallback_with_reference_log(ref_log)
            ##### space definition end #####

            if dense_out.op in s.outputs:
                dense_out = s.cache_write(output, "local")

            by, ty, yi = cfg["tile_y"].apply(s, output, y)
            bx, tx, xi = cfg["tile_x"].apply(s, output, x)

            s[output].bind(by, te.thread_axis("blockIdx.y"))
            s[output].bind(bx, te.thread_axis("blockIdx.x"))
            s[output].bind(ty, te.thread_axis("threadIdx.y"))
            s[output].bind(tx, te.thread_axis("threadIdx.x"))

            if cfg["tile_y"].size[-1] < max_unroll:
                s[output].unroll(yi)
            if cfg["tile_x"].size[-1] in vec_size:
                s[output].vectorize(xi)
            s[dense_out].compute_at(s[output], tx)

            k = s[dense_out].op.reduce_axis[0]
            y, x = s[dense_out].op.axis
            k, k_unroll = cfg["c_unroll"].apply(s, dense_out, k)
            s[dense_out].reorder(k, k_unroll, y, x)
            s[dense_out].unroll(k_unroll)
            if cfg["tile_y"].size[-1] < max_unroll:
                s[dense_out].unroll(y)
            if cfg["tile_x"].size[-1] in vec_size:
                s[dense_out].vectorize(x)

    traverse_inline(s, outs[0].op, _callback)
    return s


def fuse_and_bind(s, tensor, axis=None, num_thread=None):
    """fuse all the axis and bind to GPU threads"""
    axis = axis or s[tensor].op.axis
    fused = s[tensor].fuse(*axis)
    bx, tx = s[tensor].split(fused, num_thread)
    s[tensor].bind(bx, te.thread_axis("blockIdx.x"))
    s[tensor].bind(tx, te.thread_axis("threadIdx.x"))
    return bx, tx
