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

"""Operations that act on the DLTensor container """
from typing import Optional, Sequence, Union

from tvm.ir.expr import PrimExpr

from . import _ffi_api
from ..expr import Expr, PrimValue, ShapeExpr, DataTypeImm

PrimExprLike = Union[int, PrimExpr]


def view(
    data: Expr,
    shape: Optional[Union[Sequence[PrimExprLike], Expr]] = None,
    dtype: Optional[Expr] = None,
    relative_byte_offset: Optional[Expr] = None,
) -> Expr:
    """Broadcasts a tensor to a specified shape.

    Parameters
    ----------
    data : relax.Expr

        The input data to the operator.

    shape : Optional[Union[Sequence[PrimExprLike], Expr]]

        The target shape.  Should be a `relax.ShapeExpr`, or a
        collection that can be converted to a `relax.ShapeExpr`.

    dtype : Optional[Expr]

        The target datatype.  Should be a `relax.ShapeExpr`, or a
        collection that can be converted to a `relax.ShapeExpr`.

    relative_byte_offset: Optional[Expr]

        The offset of the output NDArray, relative to the byte offset
        of `data`.  If `None`, the offset of the view is the same as
        the offset of `data`.

    Returns
    -------
    result : relax.Expr
        The tensor view

    """

    def _normalize(expr, relax_cls):
        if expr is None or isinstance(expr, Expr):
            return expr
        else:
            return relax_cls(expr)

    shape = _normalize(shape, ShapeExpr)
    dtype = _normalize(dtype, DataTypeImm)
    relative_byte_offset = _normalize(relative_byte_offset, PrimValue)

    return _ffi_api.view(data, shape, dtype, relative_byte_offset)  # type: ignore
