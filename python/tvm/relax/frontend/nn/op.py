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
# pylint: disable=missing-docstring,too-many-lines,invalid-name,protected-access
"""nn.Tensor operators."""
from typing import List, Optional, Sequence, Union, Tuple

from tvm import tir as _tir

from ... import expr as rx
from ... import op as _op
from ...block_builder import BlockBuilder
from ...struct_info import TensorStructInfo, TupleStructInfo
from .core import Tensor

IntExpr = Union[int, _tir.PrimExpr]


def _wrap_nested(expr: rx.Expr, name: str) -> Union[Tensor, Tuple[Tensor]]:
    expr = BlockBuilder.current().emit(expr, name)
    if isinstance(expr.struct_info_, TensorStructInfo):
        return Tensor(_expr=expr)
    if isinstance(expr.struct_info_, TupleStructInfo):
        return tuple(
            _wrap_nested(
                rx.TupleGetItem(expr, i),
                name=f"{name}.{i}",
            )
            for i in range(expr.struct_info_.fields)
        )
    raise TypeError(f"Unsupported return type: {expr.struct_info_}")


def add(a: Tensor, b: Tensor, name: str = "add") -> Tensor:
    return _wrap_nested(_op.add(a._expr, b._expr), name)


def multiply(a: Tensor, b: Tensor, name: str = "mul") -> Tensor:
    return _wrap_nested(_op.multiply(a._expr, b._expr), name)


def divide(a: Tensor, b: Tensor, name: str = "divide") -> Tensor:
    return _wrap_nested(_op.divide(a._expr, b._expr), name)


def matmul(a: Tensor, b: Tensor, out_dtype: Optional[str] = None, name: str = "matmul") -> Tensor:
    return _wrap_nested(_op.matmul(a._expr, b._expr, out_dtype=out_dtype), name)


def maximum(x1: Tensor, x2: Tensor, name: str = "maximum"):
    return _wrap_nested(_op.maximum(x1._expr, x2._expr), name)


def minimum(x1: Tensor, x2: Tensor, name: str = "minimum"):
    return _wrap_nested(_op.minimum(x1._expr, x2._expr), name)


def broadcast_to(x: Tensor, shape: Sequence[IntExpr], name: str = "broadcast_to") -> Tensor:
    return _wrap_nested(_op.broadcast_to(x._expr, shape), name)


def permute_dims(x: Tensor, axes: Optional[List[int]] = None, name: str = "permute_dims") -> Tensor:
    return _wrap_nested(_op.permute_dims(x._expr, axes=axes), name)


def reshape(x: Tensor, shape: Sequence[IntExpr], name="reshape") -> Tensor:
    return _wrap_nested(_op.reshape(x._expr, shape), name)


def repeat(x: Tensor, repeats: int, axis: Optional[int] = None, name="repeat") -> Tensor:
    return _wrap_nested(_op.repeat(x._expr, repeats, axis), name)


def squeeze(x: Tensor, axis: int = -1, name: str = "squeeze") -> Tensor:
    return _wrap_nested(_op.squeeze(x._expr, axis), name)


def take(x: Tensor, indices: Tensor, axis: Optional[int] = None, name="take") -> Tensor:
    return _wrap_nested(_op.take(x._expr, indices._expr, axis), name)


def astype(x: Tensor, dtype: str, name: str = "astype") -> Tensor:
    return _wrap_nested(_op.astype(x._expr, dtype), name)


def silu(x: Tensor, name: str = "silu") -> Tensor:
    return _wrap_nested(_op.nn.silu(x._expr), name)


def softmax(x: Tensor, axis: int = -1, name: str = "softmax") -> Tensor:
    return _wrap_nested(_op.nn.softmax(x._expr, axis), name)


def rms_norm(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    axes: Union[int, List[int]],
    epsilon: float = 1e-5,
    name: str = "rms_norm",
) -> Tensor:
    if bias is None:
        bias = _op.zeros(weight.shape, dtype=weight.dtype)
    else:
        bias = bias._expr
    return _wrap_nested(_op.nn.rms_norm(x._expr, weight._expr, bias, axes, epsilon), name)


def triu(x: Tensor, diagonal: int = 0, name: str = "triu") -> Tensor:
    return _wrap_nested(_op.triu(x._expr, diagonal), name)


def full(
    shape: Sequence[IntExpr],
    fill_value: Tensor,
    dtype: str = "float32",
    name: str = "full",
) -> Tensor:
    from tvm import relax  # pylint: disable=import-outside-toplevel

    if isinstance(fill_value, (_tir.FloatImm, _tir.IntImm)):
        fill_value = relax.const(fill_value.value, dtype=dtype)
    elif isinstance(fill_value, (int, float)):
        fill_value = relax.const(fill_value, dtype=dtype)
    else:
        fill_value = fill_value._expr
    return _wrap_nested(_op.full(shape, fill_value, dtype), name)


def zeros(
    shape: Sequence[IntExpr],
    dtype: str = "float32",
    name: str = "zeros",
) -> Tensor:
    return _wrap_nested(_op.zeros(shape, dtype), name)
