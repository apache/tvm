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
from __future__ import absolute_import
import tvm
from tvm import te, auto_scheduler

from ..utils import get_const_tuple


def sparse_dense_sp_rhs(data, weight_data, weight_indices, weight_indptr):
    """
    Computes sparse-dense matrix multiplication of `data` and
    `(weight_data, weight_indices, weight_indptr).T`

    Parameters
    ----------
    data : tvm.te.Tensor
        2-D with shape [M, K]

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
    assert len(weight_data.shape) in (1, 3)
    if len(weight_data.shape) == 1:
        func = _sparse_dense_sp_rhs_csrmm
    if len(weight_data.shape) == 3:
        func = _sparse_dense_sp_rhs_bsrmm
    return func(data, weight_data, weight_indices, weight_indptr)


def sparse_dense_sp_lhs(data_data, data_indices, data_indptr, weight):
    """
    Computes sparse-dense matrix multiplication of
    `(data_data, data_indices, data_indptr)` and `weight.T`

    Parameters
    ----------
    data_data:
        1-D with shape [nnz] (CSR) or
        3-D with shape [num_blocks, bs_r, bs_c] (BSR)

    data_indices:
        1-D with shape [nnz] (CSR) or
        1-D with shape [num_blocks] (BSR)

    data_indptr:
        1-D with shape [M + 1] (CSR) or
        1-D with shape [(M + 1) // bs_r] (BSR)

    weight:
        2-D with shape [N, K]

    Returns
    -------
    output : tvm.te.Tensor
        2-D with shape [M, N]
    """
    assert len(data_data.shape) in (1, 3)
    if len(data_data.shape) == 1:
        func = _sparse_dense_sp_lhs_csrmm
    if len(data_data.shape) == 3:
        func = _sparse_dense_sp_lhs_bsrmm
    return func(data_data, data_indices, data_indptr, weight)


# pylint: disable=no-else-return,inconsistent-return-statements
def sparse_dense(dense_data, sparse_data, sparse_indices, sparse_indptr, sparse_lhs=False):
    """
    Computes sparse-dense matrix multiplication of `data` and
    `(weight_data, weight_indices, weight_indptr).T`, if sparse_lhs=False
    or
    Computes sparse-dense matrix multiplication of
    `(data_data, data_indices, data_indptr)` and `weight.T`, if sparse_lhs=True

    Parameters
    ----------
    dense_data : tvm.te.Tensor
        2-D with shape [M, K]

    sparse_data : tvm.te.Tensor
        1-D with shape [nnz] (CSR) or
        3-D with shape [num_blocks, bs_r, bs_c] (BSR)

    sparse_indices : tvm.te.Tensor
        1-D with shape [nnz] (CSR) or
        1-D with shape [num_blocks] (BSR)

    sparse_indptr : tvm.te.Tensor
        1-D with shape [N + 1] (CSR) or
        1-D with shape [(N + 1) // bs_r] (BSR)

    sparse_lhs : bool, optional
        Indicates whether lhs or rhs matrix is sparse. Default value is False.

    Returns
    -------
    output : tvm.te.Tensor
        2-D with shape [M, N]
    """
    if sparse_lhs:
        return sparse_dense_sp_lhs(sparse_data, sparse_indices, sparse_indptr, dense_data)
    else:
        return sparse_dense_sp_rhs(dense_data, sparse_data, sparse_indices, sparse_indptr)


def _sparse_dense_sp_lhs_csrmm(data_data, data_indices, data_indptr, weight):
    oshape = (get_const_tuple(data_indptr.shape)[0] - 1, get_const_tuple(weight.shape)[0])

    def f(row, i):
        row_start = data_indptr[row]
        row_end = data_indptr[row + 1]
        row_elems = row_end - row_start
        elem_idx = te.reduce_axis((0, row_elems), name="elem_idx")
        elem = row_start + elem_idx
        a_val = data_data[elem]
        weight_val = weight[i, data_indices[elem]]
        return te.sum(a_val * weight_val, axis=elem_idx)

    return te.compute(oshape, f, tag="sparse_dense_sp_lhs_csrmm")


def _sparse_dense_sp_rhs_csrmm(data, weight_data, weight_indices, weight_indptr):
    oshape = (get_const_tuple(data.shape)[0], get_const_tuple(weight_indptr.shape)[0] - 1)

    def f(i, row):
        row_start = weight_indptr[row]
        row_end = weight_indptr[row + 1]
        row_elems = row_end - row_start
        elem_idx = te.reduce_axis((0, row_elems), name="elem_idx")
        elem = row_start + elem_idx
        a_val = weight_data[elem]
        weight_val = data[i, weight_indices[elem]]
        return te.sum(a_val * weight_val, axis=elem_idx)

    return te.compute(oshape, f, tag="sparse_dense_sp_rhs_csrmm")


def _sparse_dense_sp_lhs_bsrmm(data_data, data_indices, data_indptr, weight):
    (m, _) = get_const_tuple(weight.shape)
    (_, bs_r, bs_c) = get_const_tuple(data_data.shape)
    (num_blocks_plus_1,) = get_const_tuple(data_indptr.shape)
    num_blocks = num_blocks_plus_1 - 1

    def _compute_block(nb_j, j, i):
        row_start = data_indptr[nb_j]
        row_end = data_indptr[nb_j + 1]
        row_elems = row_end - row_start
        elem_idx = te.reduce_axis((0, row_elems), name="elem_idx")
        block_offset = row_start + elem_idx
        c = te.reduce_axis((0, bs_c), name="c")
        block_j = data_indices[block_offset]
        block_ij_val = data_data[block_offset][j][c]
        x_val = weight[i, bs_c * block_j + c]
        return te.sum(block_ij_val * x_val, axis=[elem_idx, c])

    idxd = tvm.tir.indexdiv
    idxm = tvm.tir.indexmod

    bsrmm_block = te.compute(
        (num_blocks, bs_r, m), _compute_block, tag="sparse_dense_sp_lhs_bsrmm_block"
    )
    return te.compute(
        (num_blocks * bs_r, m),
        lambda m, n: bsrmm_block[idxd(m, bs_r), idxm(m, bs_r), n],
        tag="sparse_dense_sp_lhs_bsrmm",
    )


def _sparse_dense_sp_rhs_bsrmm(data, weight_data, weight_indices, weight_indptr):
    (m, k) = get_const_tuple(data.shape)
    (_, bs_r, bs_c) = get_const_tuple(weight_data.shape)
    (num_blocks_plus_1,) = get_const_tuple(weight_indptr.shape)
    num_blocks = num_blocks_plus_1 - 1

    def _compute_block(i, nb_j, j):
        row_start = weight_indptr[nb_j]
        row_end = weight_indptr[nb_j + 1]
        row_elems = row_end - row_start
        elem_idx = te.reduce_axis((0, row_elems), name="elem_idx")
        block_offset = row_start + elem_idx
        c = te.reduce_axis((0, bs_c), name="c")
        block_j = weight_indices[block_offset]
        block_ij_val = weight_data[block_offset][j][c]
        x_val = data[i, bs_c * block_j + c]
        return te.sum(block_ij_val * x_val, axis=[elem_idx, c])

    idxd = tvm.tir.indexdiv
    idxm = tvm.tir.indexmod

    bsrmm_block = te.compute(
        (m, num_blocks, bs_r),
        _compute_block,
        tag="sparse_dense_sp_rhs_bsrmm_block",
        attrs={"FLOP": 2 * m * num_blocks * bs_r * k},
    )
    return te.compute(
        (m, num_blocks * bs_r),
        lambda m, n: bsrmm_block[m, idxd(n, bs_r), idxm(n, bs_r)],
        tag="sparse_dense_sp_rhs_bsrmm",
    )


def sparse_transpose(sparse_data, sparse_indices, sparse_indptr):
    """
    Transpose a square sparse matrix,
    `A` is an n-by-n sparse matrix in the CSR format.
    ** Currently only support Square Matrices **

    Parameters
    ----------
    sparse_data : tvm.te.Tensor
        1-D with shape [nonzeros]

    sparse_indices : tvm.te.Tensor
        1-D with shape [nonzeros], dtype of 'int32'

    sparse_indptr : tvm.te.Tensor
        1-D with shape [n+1], dtype of 'int32'

    Returns
    -------
    out_data : tvm.te.Tensor
        1-D with shape [nonzeros]

    out_indices : tvm.te.Tensor
        1-D with shape [nonzeros], dtype of 'int32'

    out_indptr : tvm.te.Tensor
        1-D with shape [n+1], dtype of 'int32'
    """
    assert len(sparse_data.shape) == 1, "error in data dimension"
    assert len(sparse_indices.shape) == 1, "error in indices dimension"
    assert len(sparse_indptr.shape) == 1, "error in indptr dimension"

    nnz = get_const_tuple(sparse_data.shape)[0]
    n = get_const_tuple(sparse_indptr.shape)[0] - 1
    output_shape = [(nnz,), (nnz,), (n + 1,)]

    # TODO: Add BSR transpose support

    output_data, output_indices, output_indptr = te.extern(
        shape=output_shape,
        inputs=[sparse_data, sparse_indices, sparse_indptr],
        fcompute=lambda ins, outs: _csr_transpose_ir(
            ins[0], ins[1], ins[2], outs[0], outs[1], outs[2]
        ),
        tag="sparse_transpose_csr",
        dtype=[sparse_data.dtype, "int32", "int32"],
        name="out",
    )

    return [output_data, output_indices, output_indptr]


def _csr_transpose_ir(data, indices, indptr, out_data, out_indices, out_indptr):
    """define ir for csr_transpose"""
    irb = tvm.tir.ir_builder.create()

    data_ptr = irb.buffer_ptr(data)
    indices_ptr = irb.buffer_ptr(indices)
    indptr_ptr = irb.buffer_ptr(indptr)

    out_data_ptr = irb.buffer_ptr(out_data)
    out_indices_ptr = irb.buffer_ptr(out_indices)
    out_indptr_ptr = irb.buffer_ptr(out_indptr)

    n = get_const_tuple(indptr.shape)[0] - 1
    nnz = get_const_tuple(data.shape)[0]

    with irb.for_range(0, n, kind="parallel", name="col") as col:
        out_indptr_ptr[col] = 0

    with irb.for_range(0, nnz, kind="serial", name="nz_idx") as nz_idx:
        out_indptr_ptr[indices_ptr[nz_idx]] += 1

    cumsum = irb.allocate("int32", (1,), name="cumsum", scope="local")
    temp = irb.allocate("int32", (1,), name="temp", scope="local")
    cumsum[0] = 0
    with irb.for_range(0, n, kind="serial", name="col") as col:
        temp[0] = out_indptr_ptr[col]
        out_indptr_ptr[col] = cumsum[0]
        cumsum[0] += temp[0]

    out_indptr_ptr[n] = nnz

    with irb.for_range(0, n, kind="serial", name="row") as row:
        offset = indptr_ptr[row]
        diff = indptr_ptr[row + 1] - indptr_ptr[row]
        with irb.for_range(0, diff, kind="serial", name="idx") as idx:
            real_idx = offset + idx
            col = indices_ptr[real_idx]
            dest = out_indptr_ptr[col]

            out_indices_ptr[dest] = row
            out_data_ptr[dest] = data_ptr[real_idx]
            out_indptr_ptr[col] += 1

    last = irb.allocate("int32", (1,), name="last", scope="local")
    temp2 = irb.allocate("int32", (1,), name="temp2", scope="local")
    last[0] = 0
    with irb.for_range(0, n, kind="serial", name="col") as col:
        temp2[0] = out_indptr_ptr[col]
        out_indptr_ptr[col] = last[0]
        last[0] = temp2[0]

    return irb.get()


@tvm.target.generic_func
def sparse_dense_alter_layout(_attrs, _inputs, _tinfos, _out_type):
    """Change Sparse Dense layout.

    This is used for modifying the inputs weights so they are more amenable for
    the target.

    Parameters
    ----------
    attrs : tvm.ir.Attrs
        Attributes of current convolution
    inputs : tvm.relay.Expr
        Grouped input symbols
    tinfos : list
        Input shape and dtype
    out_type: type
        The output type

    Note
    ----
    Unlike other TOPI functions, this function operates on both graph level and operator level.
    """
    return None


@auto_scheduler.register_task_input_check_func
def try_get_sparse_input(args):
    """Analyze the input data from the given args.

    Parameters
    ----------
    args : List[Tensor]
        Input/output Tensor of a TVM subgraph.

    Returns
    -------
    Dict[Tensor, str] :
        Map from the input Tensor to its buffer name.

    Notes
    -----
    The buffer name is specially designed, and these buffer should be provided in
    `SearchTask(..., task_inputs={...})`.
    """
    sparse_prefix = sparse_data = sparse_indices = sparse_indptr = None

    def _process_inputs(input_tensors, m, n, prefix_init):
        nonlocal sparse_prefix
        nonlocal sparse_data
        nonlocal sparse_indices
        nonlocal sparse_indptr

        assert len(input_tensors) == 4
        unsure_tensors = list(input_tensors)
        # Get the Dense data
        dense_data = None
        for tensor in unsure_tensors:
            if len(tensor.shape) == 2:
                assert dense_data is None
                dense_data = tensor
                assert m == dense_data.shape[0]
                k = dense_data.shape[1]
        unsure_tensors.remove(dense_data)

        # Get the Sparse data
        sparse_data = None
        for tensor in unsure_tensors:
            if len(tensor.shape) == 3:
                assert sparse_data is None
                sparse_data = tensor
                block_size, bs_r, bs_c = sparse_data.shape
        unsure_tensors.remove(sparse_data)

        # Get the Sparse indptr & indices
        sparse_indices = None
        for tensor in unsure_tensors:
            assert len(tensor.shape) == 1
            if tensor.shape[0] == block_size:
                assert sparse_indices is None
                sparse_indices = tensor
        unsure_tensors.remove(sparse_indices)
        assert len(unsure_tensors) == 1
        sparse_indptr = unsure_tensors[0]

        # Generate the sparse_prefix
        density = 1.0
        for i in sparse_data.shape:
            density *= i
        density /= k * n
        density = density.value
        sparse_prefix = "%s_%d_%d_%d_%d_%d_%d_" % (
            prefix_init,
            n,
            k,
            bs_r,
            bs_c,
            sparse_indices.shape[0],
            sparse_indptr.shape[0],
        )

    visited = set()

    def _traverse(t):
        # We cannot directly add tensors to the set, because the comparison of
        # two tensors with ndim=0 is ambiguous.
        assert t.handle is not None
        if t.handle.value in visited:
            return

        if isinstance(t.op, te.ComputeOp):
            # TODO(jcf94): Currently only support to one sparse op, add more support here
            if t.op.tag == "sparse_dense_sp_rhs_bsrmm":
                m, n = t.shape
                assert len(t.op.input_tensors) == 1
                block_tensor = t.op.input_tensors[0]
                _process_inputs(block_tensor.op.input_tensors, m, n, "sparse_dense_bsr")
            if sparse_prefix is not None:
                # Early stop if we find a sparse_prefix
                # Notice: If any workload has more than one sparse input, this may get problem
                return
            for x in t.op.input_tensors:
                _traverse(x)
        visited.add(t.handle.value)

    try:
        for arg in args:
            _traverse(arg)
    # pylint: disable=broad-except
    except Exception:
        return {}

    if sparse_data is None or sparse_indices is None or sparse_indptr is None:
        return {}

    sparse_input_map = {}
    sparse_input_map[sparse_data] = sparse_prefix + "W_data"
    sparse_input_map[sparse_indices] = sparse_prefix + "W_indices"
    sparse_input_map[sparse_indptr] = sparse_prefix + "W_indptr"

    return sparse_input_map


def _sparse_conv2d_bsr_compute_nhwc(data, weight_data, weight_indices, weight_indptr):
    (m, h, w, k) = get_const_tuple(data.shape)  # pylint: disable=C0103
    if len(weight_data.shape) == 2:
        _, bs_r = get_const_tuple(weight_data.shape)
    elif len(weight_data.shape) == 3:
        _, bs_r, bs_c = get_const_tuple(weight_data.shape)
    (num_blocks_plus_1,) = get_const_tuple(weight_indptr.shape)
    num_blocks = num_blocks_plus_1 - 1

    def _compute_block(i, h, w, nb_j, j):  # pylint: disable=C0103
        row_start = weight_indptr[nb_j]
        row_end = weight_indptr[nb_j + 1]
        row_elems = row_end - row_start
        elem_idx = te.reduce_axis((0, row_elems), name="elem_idx")
        block_offset = row_start + elem_idx
        block_j = weight_indices[block_offset]
        if len(weight_data.shape) == 3:
            c = te.reduce_axis((0, bs_c), name="c")
            block_ij_val = weight_data[block_offset][j][c]
            x_val = data[i, h, w, bs_c * block_j + c]
            return te.sum(block_ij_val * x_val, axis=[elem_idx, c])
        else:
            block_ij_val = weight_data[block_offset][j]
            x_val = data[i, h, w, block_j]
            return te.sum(block_ij_val * x_val, axis=[elem_idx])

    idxd = tvm.tir.indexdiv
    idxm = tvm.tir.indexmod

    bsrmm_block = te.compute(
        (m, h, w, num_blocks, bs_r),
        _compute_block,
        tag="sparse_conv2d_sp_bsrmm_block",
        attrs={"FLOP": 2 * m * num_blocks * bs_r * k * h * w},
    )
    return te.compute(
        (m, h, w, num_blocks * bs_r),
        lambda m, h, w, n: bsrmm_block[m, h, w, idxd(n, bs_r), idxm(n, bs_r)],
        tag="sparse_conv2d_sp_bsrmm",
        name="sparse_conv2d",
        attrs={"layout": "NHWC"},
    )


def _sparse_conv2d_bsr_compute_nchw(data, weight_data, weight_indices, weight_indptr):
    (m, k, h, w) = get_const_tuple(data.shape)  # pylint: disable=C0103
    if len(weight_data.shape) == 2:
        _, bs_r = get_const_tuple(weight_data.shape)
    elif len(weight_data.shape) == 3:
        _, bs_r, bs_c = get_const_tuple(weight_data.shape)
    (num_blocks_plus_1,) = get_const_tuple(weight_indptr.shape)
    num_blocks = num_blocks_plus_1 - 1

    def _compute_block(i, nb_j, j, h, w):  # pylint: disable=C0103
        row_start = weight_indptr[nb_j]
        row_end = weight_indptr[nb_j + 1]
        row_elems = row_end - row_start
        elem_idx = te.reduce_axis((0, row_elems), name="elem_idx")
        block_offset = row_start + elem_idx
        block_j = weight_indices[block_offset]
        if len(weight_data.shape) == 3:
            c = te.reduce_axis((0, bs_c), name="c")
            block_ij_val = weight_data[block_offset][j][c]
            x_val = data[i, bs_c * block_j + c, h, w]
            return te.sum(block_ij_val * x_val, axis=[elem_idx, c])
        else:
            block_ij_val = weight_data[block_offset][j]
            x_val = data[i, block_j, h, w]
            return te.sum(block_ij_val * x_val, axis=[elem_idx])

    idxd = tvm.tir.indexdiv
    idxm = tvm.tir.indexmod

    bsrmm_block = te.compute(
        (m, num_blocks, bs_r, h, w),
        _compute_block,
        tag="sparse_conv2d_sp_bsrmm_block",
        attrs={"FLOP": 2 * m * num_blocks * bs_r * k * h * w},
    )
    return te.compute(
        (m, num_blocks * bs_r, h, w),
        lambda m, n, h, w: bsrmm_block[m, idxd(n, bs_r), idxm(n, bs_r), h, w],
        tag="sparse_conv2d_sp_bsrmm",
        name="sparse_conv2d",
        attrs={"layout": "NCHW"},
    )


def sparse_conv2d(
    dense_data, sparse_data, sparse_indices, sparse_indptr, layout="NHWC", kernel_size=1
):
    """
    Computes sparse-conv2d(1*1) of ``data`` and
    ``(weight_data, weight_indices, weight_indptr)``

    Parameters
    ----------
    dense_data : tvm.te.Tensor
        4-D with shape ``[M, H, W, K]`` (layout=NHWC)

        4-D with shape ``[M, K, H, W]`` (layout=NCHW)

    sparse_data : tvm.te.Tensor
        2-D with shape ``[num_blocks, bs_r]`` (BSR)

        3-D with shape ``[num_blocks, bs_r, bs_c]`` (BSR)

    sparse_indices : tvm.te.Tensor
        1-D with shape ``[num_blocks]`` (BSR)

    sparse_indptr : tvm.te.Tensor
        1-D with shape ``[(N + 1) // bs_r]`` (BSR)

    layout : str
        layout of data

    Returns
    -------
    output : tvm.te.Tensor
        4-D with shape [M, H, W, N] (layout=NHWC)
        4-D with shape [M, N, H ,W] (layout=NCHW)
    """
    if kernel_size == 1:
        if layout == "NHWC":
            return _sparse_conv2d_bsr_compute_nhwc(
                dense_data, sparse_data, sparse_indices, sparse_indptr
            )
        elif layout == "NCHW":
            return _sparse_conv2d_bsr_compute_nchw(
                dense_data, sparse_data, sparse_indices, sparse_indptr
            )
    else:
        raise ValueError(f"Unsupport Layout {layout}")


@auto_scheduler.register_task_input_check_func
def try_get_conv2d_sparse_input(args):
    """Analyze the input data from the given args.

    Parameters
    ----------
    args : List[Tensor]
        Input/output Tensor of a TVM subgraph.

    Returns
    -------
    Dict[Tensor, str] :
        Map from the input Tensor to its buffer name.

    Notes
    -----
    The buffer name is specially designed, and these buffer should be provided in
    `SearchTask(..., task_inputs={...})`.
    """
    sparse_prefix = sparse_data = sparse_indices = sparse_indptr = None

    def _process_inputs(input_tensors, m, h, w, n, prefix_init, layout):  # pylint: disable=C0103
        nonlocal sparse_prefix
        nonlocal sparse_data
        nonlocal sparse_indices
        nonlocal sparse_indptr

        assert len(input_tensors) == 4
        unsure_tensors = list(input_tensors)
        # Get the Dense data
        dense_data = None
        for tensor in unsure_tensors:
            if len(tensor.shape) == 4:
                assert dense_data is None
                dense_data = tensor
                if layout == "NHWC":
                    assert m == dense_data.shape[0]
                    assert h == dense_data.shape[1]
                    assert w == dense_data.shape[2]
                    k = dense_data.shape[3]
                elif layout == "NCHW":
                    assert m == dense_data.shape[0]
                    assert h == dense_data.shape[2]
                    assert w == dense_data.shape[3]
                    k = dense_data.shape[1]
        unsure_tensors.remove(dense_data)
        # Get the Sparse data
        sparse_data = None
        for tensor in unsure_tensors:
            if len(tensor.shape) == 3:
                assert sparse_data is None
                sparse_data = tensor
                block_size, bs_r, bs_c = sparse_data.shape
            if len(tensor.shape) == 2:
                assert sparse_data is None
                sparse_data = tensor
                block_size, bs_r = sparse_data.shape
                bs_c = 1
        unsure_tensors.remove(sparse_data)
        # Get the Sparse indptr & indices
        sparse_indices = None
        for tensor in unsure_tensors:
            assert len(tensor.shape) == 1
            if tensor.shape[0] == block_size:
                assert sparse_indices is None
                sparse_indices = tensor
        unsure_tensors.remove(sparse_indices)
        assert len(unsure_tensors) == 1
        sparse_indptr = unsure_tensors[0]
        # Generate the sparse_prefix
        density = 1.0
        for i in sparse_data.shape:
            density *= i
        density /= k * n
        density = density.value
        sparse_prefix = "%s_%d_%d_%d_%d_%d_%d_" % (
            prefix_init,
            n,
            k,
            bs_r,
            bs_c,
            sparse_indices.shape[0],
            sparse_indptr.shape[0],
        )

    visited = set()

    def _traverse(t):
        # We cannot directly add tensors to the set, because the comparison of
        # two tensors with ndim=0 is ambiguous.
        assert t.handle is not None
        if t.handle.value in visited:
            return

        if isinstance(t.op, te.ComputeOp):
            if t.op.tag == "sparse_conv2d_sp_bsrmm":
                m, h, w, n = t.shape  # pylint: disable=C0103
                assert len(t.op.input_tensors) == 1
                block_tensor = t.op.input_tensors[0]
                _process_inputs(
                    block_tensor.op.input_tensors,
                    m,
                    h,
                    w,
                    n,
                    "sparse_conv2d_bsr",
                    t.op.attrs["layout"],
                )
            if sparse_prefix is not None:
                # Early stop if we find a sparse_prefix
                # Notice: If any workload has more than one sparse input, this may get problem
                return
            for x in t.op.input_tensors:
                _traverse(x)
        visited.add(t.handle.value)

    try:
        for arg in args:
            _traverse(arg)
    # pylint: disable=broad-except
    except Exception:
        return {}

    if sparse_data is None or sparse_indices is None or sparse_indptr is None:
        return {}

    sparse_input_map = {}
    sparse_input_map[sparse_data] = sparse_prefix + "W_data"
    sparse_input_map[sparse_indices] = sparse_prefix + "W_indices"
    sparse_input_map[sparse_indptr] = sparse_prefix + "W_indptr"

    return sparse_input_map


def sparse_add(dense_data, sparse_data, sparse_indices, sparse_indptr):
    """
    Computes sparse-dense addition

    Parameters
    ----------
    dense_data : tvm.te.Tensor
        2-D with shape [M, N]

    sparse_data : tvm.te.Tensor
        1-D with shape [nnz] (CSR)

    sparse_indices : tvm.te.Tensor
        1-D with shape [nnz] (CSR)

    sparse_indptr : tvm.te.Tensor
        1-D with shape [M + 1] (CSR)

    Returns
    -------
    output : tvm.te.Tensor
        2-D with shape [M, N]
    """
    # TODO(ANSHUMAN87): support BSR format too
    assert len(sparse_data.shape) == 1, "only CSR format is supported"
    return _sparse_add_csr(dense_data, sparse_data, sparse_indices, sparse_indptr)


def _sparse_add_csr(dense_data_inp, sparse_data_inp, sparse_indices_inp, sparse_indptr_inp):
    oshape = get_const_tuple(dense_data_inp.shape)

    def _csr_add_ir(dense_data, sparse_data, sparse_indices, sparse_indptr, out_data):
        irb = tvm.tir.ir_builder.create()
        dense_data_ptr = irb.buffer_ptr(dense_data)
        sparse_data_ptr = irb.buffer_ptr(sparse_data)
        sparse_indices_ptr = irb.buffer_ptr(sparse_indices)
        sparse_indptr_ptr = irb.buffer_ptr(sparse_indptr)

        out_data_ptr = irb.buffer_ptr(out_data)

        with irb.for_range(0, oshape[0], kind="vectorize", name="row") as row:
            with irb.for_range(0, oshape[1], kind="parallel", name="col") as col:
                out_data_ptr[row, col] = dense_data_ptr[row, col]

        with irb.for_range(0, oshape[0], kind="parallel", name="row") as row:
            offset = sparse_indptr_ptr[row]
            diff = sparse_indptr_ptr[row + 1] - sparse_indptr_ptr[row]
            with irb.for_range(0, diff, kind="serial", name="idx") as idx:
                real_idx = offset + idx
                col = sparse_indices_ptr[real_idx]
                out_data_ptr[row, col] = sparse_data_ptr[real_idx] + out_data_ptr[row, col]

        return irb.get()

    return te.extern(
        shape=oshape,
        inputs=[dense_data_inp, sparse_data_inp, sparse_indices_inp, sparse_indptr_inp],
        fcompute=lambda ins, outs: _csr_add_ir(ins[0], ins[1], ins[2], ins[3], outs[0]),
        tag="sparse_add_csr",
        dtype=[
            dense_data_inp.dtype,
            sparse_data_inp.dtype,
            sparse_indices_inp.dtype,
            sparse_indptr_inp.dtype,
        ],
        name="sparse_add_csr_output",
    )
