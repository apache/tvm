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
"""Scatter operators for x86"""
import tvm
from tvm import te
from ..scatter import _verify_scatter_nd_inputs


def scatter_nd(data, indices, shape):
    """Scatter elements from a n-dimension array.

    Given data with shape (Y_0, ..., Y_{K-1}, X_M, ..., X_{N-1}), indices with shape
    (M, Y_0, ..., Y_{K-1}), and output with shape (X_0, X_1, ..., X_{N-1}), scatter_nd computes

    .. code-block::

        output[indices[0, y_0, ..., y_{K-1}],
               ...,
               indices[M-1, y_0, ..., y_{K-1}],
               x_M,
               ...,
               x_{N-1}
              ] = data[y_0, ..., y_{K-1}, x_M, ..., x_{N-1}]

    all other entries in the output are 0. Repeated indices are summed.

    Parameters
    ----------
    data : tvm.te.Tensor
        The source array.

    indices : tvm.te.Tensor
        The indices of the values to extract.

    shape : Sequence[int]
        The output shape. This must be specified because it cannot be inferred.

    Returns
    -------
    ret : tvm.te.Tensor
    """
    _verify_scatter_nd_inputs(data, indices, shape)

    def gen_ir(data_ptr, indices_ptr, out_ptr):
        # pylint: disable=invalid-name
        ib = tvm.tir.ir_builder.create()

        data = ib.buffer_ptr(data_ptr)
        indices = ib.buffer_ptr(indices_ptr)
        out = ib.buffer_ptr(out_ptr)

        # We combine all the indices dimensions but the first one into a single
        # dimension so we can iterate it in single loop instead of an arbitrary
        # number of loops. We do the same thing for all the data dimensions.
        fused_indices_dimension = 1
        for i in indices_ptr.shape[1:]:
            fused_indices_dimension *= i

        fused_data_dimension = 1
        for i in data_ptr.shape[len(indices_ptr.shape) - 1 :]:
            fused_data_dimension *= i

        fused_shape = 1
        for i in shape:
            fused_shape *= i

        # zero data
        # TODO(tkonolige): could we use topi.full to zero it instead?
        with ib.for_range(0, fused_shape) as i:
            out[i] = tvm.tir.Cast(data_ptr.dtype, 0)

        with ib.for_range(0, fused_indices_dimension) as i:
            with ib.for_range(0, fused_data_dimension, kind="parallel") as j:
                offset = fused_data_dimension
                index = j  # This is x_M, .. x_{N-1} part of the index into out.
                # Build up the indices[0, y_0, .. y_{K-1}], .. indices[M-1, y_0, .. y_{K-1}] part
                # of the index into out.
                for l in reversed(range(indices_ptr.shape[0].value)):
                    # indices[i * l * fused_indices_dimension] = indices[l, y_0, ... y_{k-1}]
                    index += offset * indices[i + l * fused_indices_dimension]
                    offset *= shape[l]
                out[index] += data[i * fused_data_dimension + j]

        return ib.get()

    out_buf = tvm.tir.decl_buffer(shape, data.dtype, "out_buf")
    return te.extern(
        [shape],
        [data, indices],
        lambda ins, outs: gen_ir(ins[0], ins[1], outs[0]),
        dtype=data.dtype,
        out_buffers=[out_buf],
        name="scatter_nd_x86",
        tag="scatter_nd_x86",
    )
