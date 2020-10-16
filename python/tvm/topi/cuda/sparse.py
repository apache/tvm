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
import numpy as np
import scipy.sparse as sp

import tvm
from tvm import relay, te

from .. import nn
from ..util import traverse_inline


def sparse_dense(data, weight_data, weight_indices, weight_indptr):
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


def schedule_sparse_dense(outs):
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

            y_bsrmm_factored = s.rfactor(y_bsrmm, c)
            tx = s[y_bsrmm].op.reduce_axis[0]
            s[y_bsrmm].bind(tx, thread_x)
            s[y_bsrmm_factored].compute_at(s[y_bsrmm], tx)
            s[y_bsrmm].set_store_predicate(thread_x.var.equal(0))
            s[out].set_store_predicate(thread_x.var.equal(0))

    traverse_inline(s, outs[0].op, _callback)
    return s


def schedule_cuda_transpose(s, out):
    """Schedule for transpose on the gpu.

    Roughly follows this:
    https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/, but
    without the padding for shared memory. For better performance, we could
    rewrite it in tir to add the padding.
    """

    def _callback(op):
        # pylint: disable=invalid-name
        m, n = s[op].op.axis
        warp_size = int(tvm.target.Target.current(allow_none=False).thread_warp_size)
        no, ni = s[op].split(n, factor=warp_size)
        mo, mi = s[op].split(m, factor=warp_size)
        s[op].reorder(mo, no, mi, ni)
        s[op].bind(mo, te.thread_axis("blockIdx.x"))
        s[op].bind(no, te.thread_axis("blockIdx.y"))
        c = s.cache_read(op.input_tensors[0], "shared", op)
        s[c].compute_at(s[op], no)
        thread_x = te.thread_axis("threadIdx.x")
        thread_y = te.thread_axis("threadIdx.y")
        s[op].bind(ni, thread_x)
        # This is a hack to make the scheduling language realize that this axis
        # can be scheduled.
        a, _ = s[c].split(s[c].op.axis[1], factor=1)
        s[c].bind(a, thread_x)
        # Use 4 warps per block. Slightly faster than 1 warp per block
        ao, _ = s[op].split(mi, nparts=4)
        s[op].bind(ao, thread_y)
        ao, _ = s[c].split(s[c].op.axis[0], nparts=4)
        s[c].bind(ao, thread_y)

    traverse_inline(s, out.op, _callback)


def sparse_dense_tir(data, w_data, w_indices, w_indptr):
    """Compute data * w^T.

    Actually computes (w * data^T) ^ T as data needs to be in column-major
    format for performance reasons.

    Good resources:
    Yang, Carl, Aydın Buluç, and John D. Owens. "Design principles for sparse
    matrix multiplication on the GPU." European Conference on Parallel
    Processing. Springer, Cham, 2018. <- This code is basically row-split from here.
    Gale, Trevor, et al. "Sparse GPU Kernels for Deep Learning." arXiv preprint
    arXiv:2006.10901 (2020).


    Profile with
    `/opt/nvidia/nsight-compute/2020.1.2/ncu -k default_function_kernel1
    --section '.*' -s 1 -c 1 venv/bin/python3 test_topi_sparse.py manual`
    with either default_function_kernel0 for the transpose or
    default_function_kernel1 for the multiply.
    """

    def ceil_div(a, b):
        return (a + (b - 1)) // b

    def gen_ir(data, w_data, w_indices, w_indptr, out):
        # pylint: disable=invalid-name
        # TODO(tkonolige): use tensorcores for block multiply
        # TODO(tkonolige): use vectorize on loads
        # TODO(tkonolige): seperate implementation if M is small
        # TODO(tkonolige): seperate implementation for large block sizes
        ib = tvm.tir.ir_builder.create()

        warp_size = int(tvm.target.Target.current(allow_none=False).thread_warp_size)
        m = data.shape[1]
        nb = w_indptr.shape[0] - 1
        nnzb = w_data.shape[0]
        # treat csr like block size 1 bsr
        if len(w_data.shape) == 1:
            bs_n = 1
            bs_k = 1
        else:
            bs_n = w_data.shape[1]
            bs_k = w_data.shape[2]
        bs_m = bs_n
        mb = m // bs_m
        mi = warp_size
        assert (
            mb >= mi
        ), "Number of block rows in dense matrix must be larger than warp size: {} vs {}.".format(
            warp_size, m
        )
        mo = ceil_div(mb, mi)
        ni = 1  # TODO(tkonolige): how do I compute the number of warps per block?
        no = ceil_div(nb, ni)
        rowlength_bi = warp_size

        bx = te.thread_axis("blockIdx.x")
        ib.scope_attr(bx, "thread_extent", mo)
        by = te.thread_axis("blockIdx.y")
        ib.scope_attr(by, "thread_extent", no)
        tx = te.thread_axis("threadIdx.x")
        ib.scope_attr(tx, "thread_extent", warp_size)
        warp = te.thread_axis("threadIdx.y")
        ib.scope_attr(warp, "thread_extent", ni)

        out_ptr = ib.buffer_ptr(out)
        data_ptr = ib.buffer_ptr(data)
        w_data_ptr = ib.buffer_ptr(w_data, shape=(nnzb, bs_n, bs_k))
        w_indices_ptr = ib.buffer_ptr(w_indices)
        w_indptr_ptr = ib.buffer_ptr(w_indptr)

        n_index = by * ni + warp
        m_index = bx * mi + tx
        row_start = w_indptr_ptr[n_index]

        # Guaranteed to be evenly divisible
        rowlength_bo = ceil_div(w_indptr_ptr[n_index + 1] - row_start, rowlength_bi)

        # thread local storage for bs_m x bs_n block
        block = ib.allocate(data.dtype, (bs_m, bs_n), name="block", scope="local")
        indices = ib.allocate(w_indices.dtype, (rowlength_bi,), name="indices", scope="warp")
        data_cache = ib.allocate(data.dtype, (mi, bs_m, bs_k), name="data_cache", scope="local")
        w_data_cache = ib.allocate(
            w_data.dtype, (rowlength_bi, bs_n, bs_k), name="w_data_cache", scope="warp"
        )

        # zero block
        with ib.for_range(0, bs_m, name="x", for_type="unroll") as x:
            with ib.for_range(0, bs_n, name="y", for_type="unroll") as y:
                block[x, y] = 0.0
        # compute into thread local storage using warp_size chunks
        with ib.for_range(0, rowlength_bo, name="bb") as bb:
            elem_idx = bb * rowlength_bi + tx
            # Cache indices. Guaranteed to be multiple of warp_size.
            indices[elem_idx] = w_indices_ptr[row_start + elem_idx]
            # cache dense matrix
            # each thread has a row
            # TODO: ideally we could vectorize this
            with ib.for_range(0, rowlength_bi, name="bi") as bi:
                with ib.for_range(0, bs_m, name="x", for_type="unroll") as x:
                    with ib.for_range(0, bs_k, name="z", for_type="unroll") as z:
                        # This memory acces should be out of bounds when
                        # m_index >= mb (which occurs when the dense matrix
                        # rows % 32 != 0), but it seems to work just fine...
                        data_cache[bi, x, z] = data_ptr[indices[bi] * bs_k + z, m_index * bs_m + x]
            # cache w_data
            elem_idx = bb * rowlength_bi + tx
            with ib.for_range(0, bs_n, name="y", for_type="unroll") as y:
                with ib.for_range(0, bs_k, name="z", for_type="unroll") as z:
                    w_data_cache[tx, y, z] = w_data_ptr[row_start + elem_idx, y, z]
            with ib.for_range(0, mi, name="i") as i:
                # thread local block matmul
                with ib.for_range(0, bs_m, name="x", for_type="unroll") as x:
                    with ib.for_range(0, bs_n, name="y", for_type="unroll") as y:
                        with ib.for_range(0, bs_k, name="z", for_type="unroll") as z:
                            block[x, y] += data_cache[i, x, z] * w_data_cache[i, y, z]
        # store results
        with ib.for_range(0, bs_m, name="x", for_type="unroll") as x:
            with ib.for_range(0, bs_n, name="y", for_type="unroll") as y:
                with ib.if_scope(m_index < mb):
                    with ib.if_scope(n_index < nb):
                        # It doesn't seem like we would be getting coelesced
                        # writes here, but it doesn't seem to matter
                        out_ptr[m_index * bs_m + x, n_index * bs_n + y] = block[x, y]

        return ib.get()

    data_t = tvm.topi.transpose(data)
    # handle csr
    if len(w_data.shape) == 1:
        blocksize = 1
    else:
        blocksize = w_data.shape[1]
    out_shape = (data_t.shape[1], (w_indptr.shape[0] - 1) * blocksize)
    out_buf = tvm.tir.decl_buffer(out_shape, data.dtype, "out_buf")
    out = te.extern(
        [out_shape],
        [data_t, w_data, w_indices, w_indptr, data],
        lambda ins, outs: gen_ir(ins[0], ins[1], ins[2], ins[3], outs[0]),
        dtype=data.dtype,
        out_buffers=[out_buf],
        name="sparse_dense_gpu",
        tag="sparse_dense_gpu",
    )
    return out


def sparse_dense_padded(data, weight_data, weight_indices, weight_indptr):
    """
    Computes sparse-dense matrix multiplication of `data` and
    `(weight_data, weight_indices, weight_indptr).T`

    This variation uses a padded matrix where all row lengths are a multiple of the warp size.

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
    return sparse_dense_tir(data, weight_data, weight_indices, weight_indptr)


def schedule_sparse_dense_padded(outs):
    """Create schedule for sparse dense"""
    # XXX: this will fail if we don't include the data_t Tensor in the schedule
    # ops. Maybe create_schedule should do some analysis so this isn't
    # necessary
    data_t = outs[0].op.input_tensors[0]
    s = te.create_schedule([outs[0].op, data_t.op])
    schedule_cuda_transpose(s, outs[0].op.input_tensors[0])
    return s


def pad_sparse_matrix(matrix, blocksize):
    """Pad rows of sparse matrix matrix so that they are a multiple of blocksize."""
    assert isinstance(matrix, sp.bsr_matrix)
    new_entries = np.zeros(matrix.shape[0], dtype=matrix.indptr.dtype)
    bsr = matrix.blocksize[0]
    for i in range(matrix.shape[0] // bsr):
        row_length = matrix.indptr[i + 1] - matrix.indptr[i]
        if row_length % blocksize != 0:
            new_entries[i] = blocksize - (row_length % blocksize)
    additional = np.sum(new_entries)
    indices = np.zeros(matrix.indices.shape[0] + additional, dtype=matrix.indices.dtype)
    data = np.zeros(
        (matrix.data.shape[0] + additional, matrix.data.shape[1], matrix.data.shape[2]),
        dtype=matrix.data.dtype,
    )

    n = matrix.shape[0] // bsr
    indptr = np.zeros(n + 1, dtype=matrix.indptr.dtype)
    indptr[: matrix.indptr.shape[0]] = matrix.indptr

    for i in range(matrix.shape[0] // bsr):
        indptr[i + 1] = indptr[i] + new_entries[i] + (matrix.indptr[i + 1] - matrix.indptr[i])
        indices[indptr[i] : indptr[i + 1] - new_entries[i]] = matrix.indices[
            matrix.indptr[i] : matrix.indptr[i + 1]
        ]
        data[indptr[i] : indptr[i + 1] - new_entries[i], :, :] = matrix.data[
            matrix.indptr[i] : matrix.indptr[i + 1], :, :
        ]

    return sp.bsr_matrix((data, indices, indptr), matrix.shape)


@nn.sparse_dense_alter_layout.register(["cuda", "gpu"])
def _alter_sparse_dense_layout(_attrs, inputs, _tinfos, _out_type):
    """With cuda, we modify use alter_op_layout to swap the default
    sparse_dense implementation for one that operates on a padded matrix. We
    also padd the matrix.
    """
    if (
        isinstance(inputs[1], relay.Constant)
        and isinstance(inputs[2], relay.Constant)
        and isinstance(inputs[3], relay.Constant)
    ):
        sparse_matrix = sp.bsr_matrix(
            (inputs[1].data.asnumpy(), inputs[2].data.asnumpy(), inputs[3].data.asnumpy())
        )
        warp_size = int(tvm.target.Target.current(allow_none=False).thread_warp_size)
        sparse_matrix = pad_sparse_matrix(sparse_matrix, warp_size)
        return relay.nn._make.sparse_dense_padded(
            inputs[0],
            relay.Constant(tvm.nd.array(sparse_matrix.data)),
            relay.Constant(tvm.nd.array(sparse_matrix.indices)),
            relay.Constant(tvm.nd.array(sparse_matrix.indptr)),
        )
    return None
