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
# pylint: disable=invalid-name
"""ScatterND operator"""
from tvm import te, tir  # hide redefinition of min and max
from tvm.tir import expr


def _verify_scatter_nd_inputs(data, indices, updates):
    mdim = int(indices.shape[0])
    assert mdim <= len(data.shape), (
        f"The first dimension of the indices ({mdim}) must be less than or equal to "
        f"the length of the shape of the output ({len(data.shape)})."
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
        # pylint: disable=invalid-name
        ib = tir.ir_builder.create()

        data = ib.buffer_ptr(data_ptr)
        indices = ib.buffer_ptr(indices_ptr)
        updates = ib.buffer_ptr(updates_ptr)
        out = ib.buffer_ptr(out_ptr)

        # We combine all the indices dimensions but the first one into a single
        # dimension so we can iterate it in single loop instead of an arbitrary
        # number of loops. We do the same thing for all the update dimensions.
        fused_indices_dimension = 1
        for i in indices_ptr.shape[1:]:
            fused_indices_dimension *= i

        fused_updates_dimension = 1
        for i in updates_ptr.shape[len(indices_ptr.shape) - 1 :]:
            fused_updates_dimension *= i

        fused_shape = 1
        for i in data_ptr.shape:
            fused_shape *= i

        with ib.for_range(0, fused_shape) as i:
            out[i] = data[i]

        with ib.for_range(0, fused_indices_dimension) as i:
            with ib.for_range(0, fused_updates_dimension, kind="parallel") as j:
                offset = fused_updates_dimension
                index = j  # This is x_M, .. x_{N-1} part of the index into out.
                # Build up the indices[0, y_0, .. y_{K-1}], .. indices[M-1, y_0, .. y_{K-1}] part
                # of the index into out.
                for l in reversed(range(indices_ptr.shape[0].value)):
                    # indices[i * l * fused_indices_dimension] = indices[l, y_0, ... y_{k-1}]
                    index += offset * indices[i + l * fused_indices_dimension]
                    offset *= data_ptr.shape[l]
                if mode == "update":
                    out[index] = updates[i * fused_updates_dimension + j]
                elif mode == "add":
                    out[index] += updates[i * fused_updates_dimension + j]
                elif mode == "mul":
                    out[index] *= updates[i * fused_updates_dimension + j]
                elif mode == "min":
                    out[index] = tir.min(out[index], updates[i * fused_updates_dimension + j])
                elif mode == "max":
                    out[index] = tir.max(out[index], updates[i * fused_updates_dimension + j])
                else:
                    raise NotImplementedError(
                        "scatter_nd mode not in [update, add, mul, min, max]:", mode
                    )

        return ib.get()

    out_buf = tir.decl_buffer(data.shape, data.dtype, "out_buf")
    return te.extern(
        [data.shape],
        [data, indices, updates],
        lambda ins, outs: gen_ir(ins[0], ins[1], ins[2], outs[0]),
        dtype=data.dtype,
        out_buffers=[out_buf],
        name="scatter_nd.generic",
        tag="scatter_nd.generic",
    )
