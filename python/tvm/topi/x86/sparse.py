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

"""sparse_dense schedule on x86"""
from tvm import te

from ..utils import traverse_inline, get_const_int
from .utils import get_simd_32bit_lanes


def schedule_sparse_dense(outs):
    """Create schedule for sparse dense"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        simd_width = get_simd_32bit_lanes()
        if op.tag == "sparse_dense_sp_lhs_csrmm" or op.tag == "sparse_dense_sp_lhs_csrmm":
            (y_o, y_i) = s[op].split(s[op].op.axis[1], 2)
            fused = s[op].fuse(s[op].op.axis[0], y_o)
            s[op].parallel(fused)
            s[op].vectorize(y_i)
        elif op.tag == "sparse_dense_sp_rhs_bsrmm" or op.tag == "sparse_dense_sp_rhs_bsrmm":
            y_bsrmm = op.input_tensors[0]
            assert (
                y_bsrmm.op.tag == "sparse_dense_sp_rhs_bsrmm_block"
                or y_bsrmm.op.tag == "sparse_dense_sp_lhs_bsrmm_block"
            )
            y_reshape = op
            (m, num_blocks, b_r) = s[y_bsrmm].op.axis
            bs_r = get_const_int(b_r.dom.extent)
            (elem_idx, c) = s[y_bsrmm].op.reduce_axis
            s[y_bsrmm].reorder(num_blocks, m, elem_idx, b_r, c)
            s[y_bsrmm].vectorize(b_r)
            (m_o, n_o) = s[y_reshape].op.axis
            (noo, noi) = s[y_reshape].split(n_o, bs_r)
            s[y_bsrmm].compute_at(s[y_reshape], noi)
            s[y_reshape].vectorize(noi)
            if op != s[outs[0]].op:
                (y_o, y_i) = s[outs[0].op].split(s[outs[0].op].op.axis[1], 2 * simd_width)
                s[y_reshape].compute_at(s[outs[0]], y_o)
                s[outs[0].op].parallel(y_o)
                s[outs[0].op].vectorize(y_i)
            else:
                m_o_noo = s[y_reshape].fuse(m_o, noo)
                s[y_reshape].parallel(m_o_noo)

    traverse_inline(s, outs[0].op, _callback)
    return s
