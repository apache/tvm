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
# pylint: disable=redefined-builtin
"""Linear algebra operators."""
from tvm import _ffi

from ..expr import Call
from . import ty
from . import ty_guard as tg

# pylint: disable=invalid-name


def linear(
    data: ty.Tensor,
    weight: ty.Tensor,
    bias: ty.Optional[ty.Tensor] = None,
    out_dtype: ty.DType = None,
) -> Call:
    """Applies a linear transformation to the incoming data: y = xA^T + b

    Parameters
    ----------
    data : relax.Expr
        The input data.

    weight : relax.Expr
        The weight tensor.

    bias : Optional[Expr]
        The bias tensor.

    out_dtype: Optional[Union[str, DataType]]
        The data type of the matmul result.
        When it is not specified, the output dtype will be the the same as input dtype.

    Notes
    -----
    Relax does not regard the Linear Op as a primitive Op,
    while combine the transpose, matmul and add op to implement it.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    from tvm.relax.op import permute_dims  # pylint: disable=import-outside-toplevel

    # Since weight can be 1D or 2D, we use `axes=None` to support both cases.
    x = matmul(data, permute_dims(weight, axes=None), out_dtype=out_dtype)
    return x + bias if bias is not None else x


## (TVM-TOOL) py_op begin linear_algebra/*
def matmul(
    x1: ty.Tensor,
    x2: ty.Tensor,
    out_dtype: ty.DType = None,
) -> Call:
    """TBD

    Parameters
    ----------
    x1 : ty.Tensor
        TODO(tvm-unity-team): add doc
    x2 : ty.Tensor
        TODO(tvm-unity-team): add doc
    out_dtype : ty.DType
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    x1 = tg.check(0, "x1", tg.Tensor([]), x1)
    x2 = tg.check(1, "x2", tg.Tensor([]), x2)
    out_dtype = tg.check(2, "out_dtype", tg.DType(), out_dtype)
    _ffi_func = _ffi.get_global_func("relax.op.matmul")
    return _ffi_func(x1, x2, out_dtype)


## (TVM-TOOL) py_op end linear_algebra/*
