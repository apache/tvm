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
# pylint: disable=invalid-name, no-value-for-parameter
"""Direct implementation of dense."""

from tvm import te
from tvm.topi.utils import traverse_inline, get_const_tuple

from .micro_kernel.gemm import (
    intrin_gemm_MxKxN,
    gemm_MxKxN_impl,
)
from .... import tag


def dense_dsp_compute(cfg, data, weight, bias=None, out_dtype=None):
    """Defines the v7e-m DSP instructions of dense."""
    M, K = get_const_tuple(data.shape)
    N, _ = get_const_tuple(weight.shape)

    cfg.define_split("tile_x", M, policy="factors", num_outputs=2)
    cfg.define_split("tile_y", N, policy="factors", num_outputs=2)
    cfg.define_split("tile_k", K, policy="factors", num_outputs=2)

    k = te.reduce_axis((0, K), "k")
    C = te.compute(
        (M, N),
        lambda x, y: te.sum(
            data[x, k].astype(out_dtype) * weight[y, k].astype(out_dtype),
            axis=k,
        ),
        name="dense",
        tag="dense_dsp",
    )

    if bias is not None:
        C = te.compute((M, N), lambda i, j: C[i, j] + bias[j].astype(out_dtype), tag=tag.BROADCAST)
    return C


def dense_dsp_schedule(cfg, outs):
    """Schedule function for v7e-m DSP instructions of dense."""
    sched = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "dense" not in op.tag:
            return

        output = op.output(0)
        dense = op

        data = dense.input_tensors[0]

        M = cfg["tile_x"].size[-1]
        N = cfg["tile_y"].size[-1]
        K = cfg["tile_k"].size[-1]

        x, y = sched[dense].op.axis
        k = sched[dense].op.reduce_axis[0]

        x_o, x_i = cfg["tile_x"].apply(sched, dense, x)
        y_o, y_i = cfg["tile_y"].apply(sched, dense, y)
        k_o, k_i = cfg["tile_k"].apply(sched, dense, k)

        sched[dense].reorder(x_o, y_o, k_o, x_i, y_i, k_i)

        gemm, uniq_id = intrin_gemm_MxKxN(M, K, N, data.dtype, output.dtype, stride_w=1)
        sched[output].tensorize(x_i, gemm)
        sched[output].pragma(x_o, "import_c", gemm_MxKxN_impl(M, K, N, uniq_id))

    traverse_inline(sched, outs[-1].op, _callback)
    return sched
