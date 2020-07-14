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
# pylint: disable=invalid-name, no-member, too-many-locals, too-many-arguments, too-many-statements, singleton-comparison
# pylint: disable=bad-continuation, unused-argument
"""Transform operator"""
import tvm
from tvm import te


def invert_permutation_ir(data, out):
    """Low level IR to get invert_permutation.

    Parameters
    ----------
    data : Buffer
        Input data. 1-D Buffer with shape [elem_num].

    out : Buffer
        1D buffer for invert permutation result with the same shape with data.

    Returns
    -------
    stmt : Stmt
        The result IR statement.
    """
    elem_num = data.shape[0]

    ib = tvm.tir.ir_builder.create()
    data = ib.buffer_ptr(data)
    out = ib.buffer_ptr(out)

    max_threads = int(tvm.target.Target.current(
        allow_none=False).max_num_threads)
    nthread_tx = max_threads
    nthread_bx = elem_num // max_threads + 1
    tx = te.thread_axis("threadIdx.x")
    bx = te.thread_axis("blockIdx.x")
    ib.scope_attr(tx, "thread_extent", nthread_tx)
    ib.scope_attr(bx, "thread_extent", nthread_bx)
    tid = bx * max_threads + tx

    with ib.if_scope(tid < elem_num):
        r_ind = data[tid]
        out[r_ind] = tid
    return ib.get()

def invert_permutation(data):
    """Compute definition of invert_permutation.

    for an output tensor y and an input tensor x, this operation computes the following:
    y[x[i]] = i for i in [0, 1, ..., len(x) - 1]

    Parameters
    ----------
    data : tvm.te.Tensor
        1-D tensor

    Returns
    -------
    out : tvm.te.Tensor
    """
    data_buf = tvm.tir.decl_buffer(
        data.shape, data.dtype, "data_buf", data_alignment=8)
    out_buf = tvm.tir.decl_buffer(
        data.shape, data.dtype, "out_buf", data_alignment=8)

    out = te.extern([data.shape,], [data,],
                    lambda ins, outs: invert_permutation_ir(ins[0], outs[0]),
                    in_buffers=[data_buf,],
                    out_buffers=[out_buf,],
                    name="invert_permutation",
                    tag="invert_permutation_gpu")

    return out
