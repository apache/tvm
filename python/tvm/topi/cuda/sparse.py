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
from ..utils import traverse_inline, get_const_tuple, prod, get_const_int, ceil_div
from .transform import schedule_transpose_from_existing


def sparse_dense(data, weight_data, weight_indices, weight_indptr, sparse_lhs=False):
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
    return nn.sparse_dense(data, weight_data, weight_indices, weight_indptr, sparse_lhs)


def schedule_sparse_dense(outs):
    """Create schedule for sparse dense"""
    # pylint:disable=invalid-name
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == "sparse_dense_sp_rhs_bsrmm" or op.tag == "sparse_dense_sp_lhs_bsrmm":
            y_bsrmm = op.input_tensors[0]
            assert (
                y_bsrmm.op.tag == "sparse_dense_sp_rhs_bsrmm_block"
                or y_bsrmm.op.tag == "sparse_dense_sp_lhs_bsrmm_block"
            )
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
        elif op.tag == "sparse_dense_sp_lhs_csrmm" or op.tag == "sparse_dense_sp_rhs_csrmm":
            out = op.output(0)
            const_size = get_const_int(prod(out.shape))
            fused = s[out].fuse(*s[out].op.axis)
            bx, tx = s[out].split(fused, factor=const_size)
            s[out].bind(tx, te.thread_axis("threadIdx.x"))
            s[out].bind(bx, te.thread_axis("blockIdx.x"))

    traverse_inline(s, outs[0].op, _callback)
    return s


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

    def gen_ir(data, w_data, w_indices, w_indptr, out):
        # pylint: disable=invalid-name, simplifiable-if-statement
        # TODO(tkonolige): use tensorcores for block multiply
        # TODO(tkonolige): use vectorize on loads
        # TODO(tkonolige): seperate implementation if M is small
        # TODO(tkonolige): seperate implementation for large block sizes
        ib = tvm.tir.ir_builder.create()

        if tvm.target.Target.current(allow_none=False).kind.name == "cuda":
            use_warp_storage = True
        else:
            # TVMs warp shuffle intrinsics are slow on ROCM because they use
            # LDS (shared memory) to do the shuffling. Instead, we could use
            # ROCM's support for accessing neighboring threads memory, but we
            # those intrinsics aren't accessible from TVM. For now, we just use
            # shared memory. We also default to shared memory on platforms
            # where we do not know how warp storage performs.
            use_warp_storage = False

        warp_size = int(tvm.target.Target.current(allow_none=False).thread_warp_size)
        m = data.shape[1]
        nb = w_indptr.shape[0] - 1
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
            warp_size, mb
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
        w_data_ptr = ib.buffer_ptr(w_data)
        w_indices_ptr = ib.buffer_ptr(w_indices)
        w_indptr_ptr = ib.buffer_ptr(w_indptr)

        n_index = by * ni + warp
        m_index = bx * mi + tx
        row_start = w_indptr_ptr[n_index]

        # Guaranteed to be evenly divisible
        rowlength_bo = ceil_div(w_indptr_ptr[n_index + 1] - row_start, rowlength_bi)

        # thread local storage for bs_m x bs_n block
        block = ib.allocate(data.dtype, (bs_m, bs_n), name="block", scope="local")
        data_cache = ib.allocate(data.dtype, (mi, bs_m, bs_k), name="data_cache", scope="local")
        if use_warp_storage:
            indices = ib.allocate(w_indices.dtype, (rowlength_bi,), name="indices", scope="warp")
            w_data_cache = ib.allocate(
                w_data.dtype, (rowlength_bi, bs_n, bs_k), name="w_data_cache", scope="warp"
            )
        else:
            indices = ib.allocate(
                w_indices.dtype, (ni, rowlength_bi), name="indices", scope="shared"
            )
            w_data_cache = ib.allocate(
                w_data.dtype, (ni, rowlength_bi, bs_n, bs_k), name="w_data_cache", scope="shared"
            )

        # zero block
        with ib.for_range(0, bs_m, name="x", kind="unroll") as x:
            with ib.for_range(0, bs_n, name="y", kind="unroll") as y:
                block[x, y] = 0.0
        # compute into thread local storage using warp_size chunks
        with ib.for_range(0, rowlength_bo, name="bb") as bb:
            elem_idx = bb * rowlength_bi + tx
            # Cache indices. Guaranteed to be multiple of warp_size.
            if use_warp_storage:
                indices[tx] = w_indices_ptr[row_start + elem_idx]
            else:
                indices[warp, tx] = w_indices_ptr[row_start + elem_idx]
            # cache dense matrix
            # each thread has a row
            # TODO: ideally we could vectorize this
            with ib.for_range(0, rowlength_bi, name="bi") as bi:
                with ib.for_range(0, bs_m, name="x", kind="unroll") as x:
                    with ib.for_range(0, bs_k, name="z", kind="unroll") as z:
                        # This memory acces should be out of bounds when
                        # m_index >= mb (which occurs when the dense matrix
                        # rows % 32 != 0), but it seems to work just fine...
                        if use_warp_storage:
                            ind = indices[bi]
                        else:
                            ind = indices[warp, bi]
                        data_cache[bi, x, z] = data_ptr[ind * bs_k + z, m_index * bs_m + x]
            # cache w_data
            elem_idx = bb * rowlength_bi + tx
            with ib.for_range(0, bs_n, name="y", kind="unroll") as y:
                with ib.for_range(0, bs_k, name="z", kind="unroll") as z:
                    data_indices = [row_start + elem_idx] + (
                        [y, z] if len(w_data.shape) > 1 else []
                    )
                    cache_indices = [tx, y, z] if use_warp_storage else [warp, tx, y, z]
                    w_data_cache[cache_indices] = w_data_ptr[data_indices]
            with ib.for_range(0, mi, name="i") as i:
                # thread local block matmul
                with ib.for_range(0, bs_m, name="x", kind="unroll") as x:
                    with ib.for_range(0, bs_n, name="y", kind="unroll") as y:
                        with ib.for_range(0, bs_k, name="z", kind="unroll") as z:
                            if use_warp_storage:
                                w = w_data_cache[i, y, z]
                            else:
                                w = w_data_cache[warp, i, y, z]
                            block[x, y] += data_cache[i, x, z] * w
        # store results
        with ib.for_range(0, bs_m, name="x", kind="unroll") as x:
            with ib.for_range(0, bs_n, name="y", kind="unroll") as y:
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


def is_valid_for_sparse_dense_padded(data, weight_data):
    """
    Check whether input is applicable for sparse_dense_padded op.
    If not we should fall back to default scheduling.
    """
    # pylint:disable=invalid-name
    warp_size = int(tvm.target.Target.current(allow_none=False).thread_warp_size)
    # If there are multiple alter_ops in a model, the first alteration does not
    # run type inference for the subsequent ones. In this case, we don't have
    # the shape information, so we run the inferencer manually.
    try:
        m = get_const_tuple(data.checked_type.shape)[1]
    except ValueError:
        data_infered = relay.transform.InferType()(tvm.IRModule.from_expr(data))["main"]
        m = get_const_tuple(data_infered.ret_type.shape)[1]
    if len(weight_data.shape) == 1:
        bs_m = 1
    else:
        bs_m = weight_data.shape[1]

    mb = m // bs_m
    if mb >= warp_size:
        return True
    return False


def sparse_dense_padded(data, weight_data, weight_indices, weight_indptr, sparse_lhs=False):
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
    # TODO(ANSHUMAN87): Handle for sparse_lhs case too
    assert not sparse_lhs, "Currently only sparse weight is supported."
    return sparse_dense_tir(data, weight_data, weight_indices, weight_indptr)


def schedule_sparse_dense_padded(outs):
    """Create schedule for sparse dense"""
    # XXX: this will fail if we don't include the data_t Tensor in the schedule
    # ops. Maybe create_schedule should do some analysis so this isn't
    # necessary
    data_t = outs[0].op.input_tensors[0]
    s = te.create_schedule([outs[0].op, data_t.op])
    schedule_transpose_from_existing(s, outs[0].op.input_tensors[0])
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


@nn.sparse_dense_alter_layout.register(["cuda", "gpu", "rocm"])
def _alter_sparse_dense_layout(_attrs, inputs, _tinfos, _out_type):
    """With cuda, we modify use alter_op_layout to swap the default
    sparse_dense implementation for one that operates on a padded matrix. We
    also pad the matrix.
    """
    # TODO(ANSHUMAN87): Handle for sparse_lhs case too
    if (
        isinstance(inputs[1], relay.Constant)
        and isinstance(inputs[2], relay.Constant)
        and isinstance(inputs[3], relay.Constant)
        and is_valid_for_sparse_dense_padded(inputs[0], inputs[1].data.numpy())
    ):
        if len(inputs[1].data.numpy().shape) == 1:
            sparse_matrix = sp.csr_matrix(
                (inputs[1].data.numpy(), inputs[2].data.numpy(), inputs[3].data.numpy())
            ).tobsr()
        else:
            sparse_matrix = sp.bsr_matrix(
                (inputs[1].data.numpy(), inputs[2].data.numpy(), inputs[3].data.numpy())
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
