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
from tvm.topi.utils import traverse_inline

from .micro_kernel.gemm import (
    intrin_gemm_MxKxN,
    gemm_MxKxN_impl,
)


def dense_dsp_schedule(outs):
    """Schedule function for v7e-m DSP instructions of dense."""
    sched = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "dense" not in op.tag:
            return

        # extract tensors
        output = op.output(0)
        dense = op
        data_vec = dense.input_tensors[0]
        M, K = data_vec.shape
        N, _ = dense.input_tensors[1].shape

        n, _ = sched[dense].op.axis
        no, ni = sched[dense].split(n, nparts=1)

        gemm, uniq_id = intrin_gemm_MxKxN(M, K, N, data_vec.dtype, output.dtype)
        sched[output].tensorize(ni, gemm)
        sched[output].pragma(no, "import_c", gemm_MxKxN_impl(M, K, N, uniq_id))

    traverse_inline(sched, outs[-1].op, _callback)
    return sched
