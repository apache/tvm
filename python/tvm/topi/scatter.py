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
# pylint: disable=invalid-name, too-many-arguments, too-many-nested-blocks
"""Scatter operator"""
from ..tir import decl_buffer, ir_builder, AssertStmt, StringImm, Evaluate, expr
from ..te import extern, hybrid


@hybrid.script
def _scatter_1d(data, indices, updates):
    out = output_tensor(data.shape, data.dtype)
    for i in range(data.shape[0]):
        out[i] = data[i]
    for i in range(indices.shape[0]):
        out[indices[i] if indices[i] >= 0 else indices[i] + data.shape[0]] = updates[i]
    return out


@hybrid.script
def _scatter_2d(data, indices, updates, axis):
    out = output_tensor(data.shape, data.dtype)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            out[i, j] = data[i, j]
    if axis == 0:
        for i in range(indices.shape[0]):
            for j in range(indices.shape[1]):
                out[
                    indices[i, j] if indices[i, j] >= 0 else indices[i, j] + data.shape[axis], j
                ] = updates[i, j]
    else:
        for i in range(indices.shape[0]):
            for j in range(indices.shape[1]):
                out[
                    i, indices[i, j] if indices[i, j] >= 0 else indices[i, j] + data.shape[axis]
                ] = updates[i, j]

    return out


@hybrid.script
def _scatter_3d(data, indices, updates, axis):
    out = output_tensor(data.shape, data.dtype)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                out[i, j, k] = data[i, j, k]
    if axis == 0:
        for i in range(indices.shape[0]):
            for j in range(indices.shape[1]):
                for k in range(indices.shape[2]):
                    out[
                        indices[i, j, k]
                        if indices[i, j, k] >= 0
                        else indices[i, j, k] + data.shape[axis],
                        j,
                        k,
                    ] = updates[i, j, k]
    elif axis == 1:
        for i in range(indices.shape[0]):
            for j in range(indices.shape[1]):
                for k in range(indices.shape[2]):
                    out[
                        i,
                        indices[i, j, k]
                        if indices[i, j, k] >= 0
                        else indices[i, j, k] + data.shape[axis],
                        k,
                    ] = updates[i, j, k]
    else:
        for i in range(indices.shape[0]):
            for j in range(indices.shape[1]):
                for k in range(indices.shape[2]):
                    out[
                        i,
                        j,
                        indices[i, j, k]
                        if indices[i, j, k] >= 0
                        else indices[i, j, k] + data.shape[axis],
                    ] = updates[i, j, k]

    return out


@hybrid.script
def _scatter_4d(data, indices, updates, axis):
    out = output_tensor(data.shape, data.dtype)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                for l in range(data.shape[3]):
                    out[i, j, k, l] = data[i, j, k, l]

    if axis == 0:
        for i in range(indices.shape[0]):
            for j in range(indices.shape[1]):
                for k in range(indices.shape[2]):
                    for l in range(indices.shape[3]):
                        out[
                            indices[i, j, k, l]
                            if indices[i, j, k, l] >= 0
                            else indices[i, j, k, l] + data.shape[axis],
                            j,
                            k,
                            l,
                        ] = updates[i, j, k, l]
    elif axis == 1:
        for i in range(indices.shape[0]):
            for j in range(indices.shape[1]):
                for k in range(indices.shape[2]):
                    for l in range(indices.shape[3]):
                        out[
                            i,
                            indices[i, j, k, l]
                            if indices[i, j, k, l] >= 0
                            else indices[i, j, k, l] + data.shape[axis],
                            k,
                            l,
                        ] = updates[i, j, k, l]
    elif axis == 2:
        for i in range(indices.shape[0]):
            for j in range(indices.shape[1]):
                for k in range(indices.shape[2]):
                    for l in range(indices.shape[3]):
                        out[
                            i,
                            j,
                            indices[i, j, k, l]
                            if indices[i, j, k, l] >= 0
                            else indices[i, j, k, l] + data.shape[axis],
                            l,
                        ] = updates[i, j, k, l]
    else:
        for i in range(indices.shape[0]):
            for j in range(indices.shape[1]):
                for k in range(indices.shape[2]):
                    for l in range(indices.shape[3]):
                        out[
                            i,
                            j,
                            k,
                            indices[i, j, k, l]
                            if indices[i, j, k, l] >= 0
                            else indices[i, j, k, l] + data.shape[axis],
                        ] = updates[i, j, k, l]

    return out


def scatter(data, indices, updates, axis=0):
    """Update data at positions defined by indices with values in updates

    Parameters
    ----------
    data : relay.Expr
        The input data to the operator.

    indices : relay.Expr
        The index locations to update.

    updates : relay.Expr
        The values to update.

    axis : int
        The axis to scatter on

    Returns
    -------
    ret : relay.Expr
        The computed result.
    """
    if axis < 0:
        axis += len(data.shape)
    assert axis >= 0
    assert axis < len(data.shape)

    if len(data.shape) == 1:
        return _scatter_1d(data, indices, updates)
    if len(data.shape) == 2:
        return _scatter_2d(data, indices, updates, axis)
    if len(data.shape) == 3:
        return _scatter_3d(data, indices, updates, axis)
    if len(data.shape) == 4:
        return _scatter_4d(data, indices, updates, axis)
    raise ValueError("scatter only support for 1-4 dimensions")


def _verify_scatter_nd_inputs(data, indices, updates):
    mdim = int(indices.shape[0])
    assert mdim <= len(data.shape), (
        f"The first dimension of the indices ({mdim}) must be less than or equal to "
        f"the length of the shape of the output ({len(shape)})."
    )
    for i in range(len(indices.shape) - 1):
        if isinstance(indices.shape[i + 1], expr.Var) or isinstance(updates.shape[i], expr.Var):
            continue
        assert indices.shape[i + 1] == updates.shape[i], (
            f"Dimension of indices[{i+1}] ({indices.shape[i+1]}) must equal dimension of "
            f"updates[{i}] ({updates.shape[i]})."
        )
    for i in range(mdim, len(data.shape)):
        data_ind = i - mdim + len(indices.shape) - 1
        if isinstance(updates.shape[data_ind], expr.Var) or isinstance(data.shape[i], expr.Var):
            continue
        assert updates.shape[data_ind] == data.shape[i], (
            f"Dimension of updates[{data_ind}] ({updates.shape[data_ind]}) must equal dimension "
            f"of out_shape[{i}] ({data.shape[i]})."
        )

    assert (
        "int" in indices.dtype
    ), f"Indices must be a tensor of integers, but its elements are {indices.dtype}."


def scatter_nd(data, indices, updates, mode):
    """Scatter elements from a n-dimension array.

    Given updates with shape (Y_0, ..., Y_{K-1}, X_M, ..., X_{N-1}), indices with shape
    (M, Y_0, ..., Y_{K-1}), and output copied from data with shape (X_0, X_1, ..., X_{N-1}),
    scatter_nd computes

    .. code-block::

        output[indices[0, y_0, ..., y_{K-1}],
               ...,
               indices[M-1, y_0, ..., y_{K-1}],
               x_M,
               ...,
               x_{N-1}
              ] = f(output[...], updates[y_0, ..., y_{K-1}, x_M, ..., x_{N-1}])

    where the update function f is determinted by the mode.

    Parameters
    ----------
    data : tvm.te.Tensor
        The source array.

    indices : tvm.te.Tensor
        The indices of the values to extract.

    updates : tvm.te.Tensor
        The updates to apply at the Indices

    mode : string
        The update mode for the algorithm, either "update" or "add"
        If update, the update values will replace the input data
        If add, the update values will be added to the input data

    Returns
    -------
    ret : tvm.te.Tensor
    """
    _verify_scatter_nd_inputs(data, indices, updates)

    def gen_ir(data_ptr, indices_ptr, updates_ptr, out_ptr):
        ib = ir_builder.create()

        data = ib.buffer_ptr(data_ptr)
        indices = ib.buffer_ptr(indices_ptr)
        updates = ib.buffer_ptr(updates_ptr)
        out = ib.buffer_ptr(out_ptr)

        fused_shape = 1
        for i in data.shape:
            fused_shape *= i
        with ib.for_range(0, fused_shape) as i:
            out[i] = data[i]

        # We combine all the indices dimensions but the first one into a single
        # dimension so we can iterate it in single loop instead of an arbitrary
        # number of loops. We do the same thing for all the data dimensions.
        fused_indices_dimension = 1
        for i in indices_ptr.shape[1:]:
            fused_indices_dimension *= i

        fused_data_dimension = 1
        for i in data_ptr.shape[len(indices_ptr.shape) - 1 :]:
            fused_data_dimension *= i

        with ib.for_range(0, fused_indices_dimension, name="i") as i:
            with ib.for_range(0, fused_data_dimension, name="j") as j:
                offset = fused_data_dimension
                index = j  # This is x_M, .. x_{N-1} part of the index into out.
                # Build up the indices[0, y_0, .. y_{K-1}], .. indices[M-1, y_0, .. y_{K-1}] part
                # of the index into out.
                for l in reversed(range(indices_ptr.shape[0].value)):
                    # indices[i * l * fused_indices_dimension] = indices[l, y_0, ... y_{k-1}]
                    index += offset * indices[i + l * fused_indices_dimension]
                    ib.emit(
                        AssertStmt(
                            indices[i + l * fused_indices_dimension] < shape[l],
                            StringImm("index out of bounds"),
                            Evaluate(0),
                        )
                    )
                    offset *= shape[l]
                if mode == "add":
                    out[index] += updates[i * fused_data_dimension + j]
                elif mode == "update":
                    out[index] = updates[i * fused_data_dimension + j]
                else:
                    raise NotImplementedError("scatter_nd mode not in [update, add]:", mode)

        return ib.get()

    out_buf = decl_buffer(shape, data.dtype, "out_buf")
    return extern(
        [shape],
        [data, indices, updates],
        lambda ins, outs: gen_ir(ins[0], ins[1], ins[2], outs[0]),
        dtype=data.dtype,
        out_buffers=[out_buf],
        name="scatter_nd_generic",
        tag="scatter_nd_generic",
    )
