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

"""Sparse operators"""
from tvm import te
from tvm import autotvm
from tvm.autotvm.task.space import SplitEntity
from ..util import traverse_inline
from .. import nn


@autotvm.register_topi_compute("sparse_dense.cuda")
def sparse_dense(cfg, data, weight_data, weight_indices, weight_indptr):
    """
    Computes sparse-dense matrix multiplication of `data` and
    `(weight_data, weight_indices, weight_indptr).T`

    Parameters
    ----------
    cfg: ConfigEntity
        The config for this template

    data : tvm.te.Tensor
        2-D with shape [M, K], float32

    weight_data : tvm.te.Tensor
        1-D with shape [nnz] (CSR) or
        3-D with shape [num_blocks, bs_r, bs_c] (BSR)

    weight_indices : tvm.te.Tensor
        1-D with shape [nnz] (CSR) or
        1-D with shape [num_blocks] (BSR)

    weight_indptr : tvm.te.Tensor
        1-D with shape [N + 1] (CSR) or
        1-D with shape [(N + 1) // bs_r] (BSR)

    Returns
    -------
    output : tvm.te.Tensor
        2-D with shape [M, N]
    """
    # pylint:disable=unused-argument
    return nn.sparse_dense(data, weight_data, weight_indices, weight_indptr)


@autotvm.register_topi_schedule("sparse_dense.cuda")
def schedule_sparse_dense(cfg, outs):
    """Create schedule for sparse dense"""
    # pylint:disable=invalid-name
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == "sparse_dense_bsrmm":
            y_bsrmm = op.input_tensors[0]
            assert y_bsrmm.op.tag == "sparse_dense_bsrmm_block"
            out = s.outputs[0].output(0)

            if op not in s.outputs:
                y_reshape = op.output(0)
                s[y_reshape].compute_at(s[out], s[out].op.axis[1])

            (_, c) = s[y_bsrmm].op.reduce_axis

            (m_o, n_o) = s[out].op.axis
            s[out].bind(m_o, te.thread_axis("blockIdx.x"))
            s[out].bind(n_o, te.thread_axis("blockIdx.y"))
            s[y_bsrmm].compute_at(s[out], n_o)

            thread_x = te.thread_axis("threadIdx.x")

            cfg.define_split("tile_c", c, num_outputs=2)
            if cfg.is_fallback:
                cfg["tile_c"] = SplitEntity([-1, 8])
            _, ci = cfg["tile_c"].apply(s, y_bsrmm, c)

            y_bsrmm_factored = s.rfactor(y_bsrmm, ci)
            tx = s[y_bsrmm].op.reduce_axis[0]
            s[y_bsrmm].bind(tx, thread_x)
            s[y_bsrmm_factored].compute_at(s[y_bsrmm], tx)
            s[y_bsrmm].set_store_predicate(thread_x.var.equal(0))
            s[out].set_store_predicate(thread_x.var.equal(0))

    traverse_inline(s, outs[0].op, _callback)
    return s
