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
# pylint: disable= redefined-builtin
"""Relax core operators."""

# Register operator gradient functions
from . import _op_gradient, builtin, ccl, distributed, grad, image, memory, nn, op_attrs

# Operators
from .base import (
    assert_op,
    call_builtin_with_ctx,
    call_dps_packed,
    call_inplace_packed,
    call_pure_packed,
    call_tir,
    call_tir_inplace,
    call_tir_with_grad,
    hint_on_device,
    invoke_closure,
    invoke_pure_closure,
    make_closure,
    null_value,
    print,
    register_gradient,
    shape_of,
    shape_to_tensor,
    tensor_to_shape,
    to_vdevice,
)
from .binary import (
    add,
    bitwise_and,
    bitwise_or,
    bitwise_xor,
    divide,
    equal,
    floor_divide,
    floor_mod,
    greater,
    greater_equal,
    left_shift,
    less,
    less_equal,
    logical_and,
    logical_or,
    logical_xor,
    maximum,
    minimum,
    mod,
    multiply,
    not_equal,
    power,
    right_shift,
    subtract,
)
from .create import (
    arange,
    full,
    full_like,
    ones,
    ones_like,
    eye,
    eye_like,
    tril,
    triu,
    zeros,
    zeros_like,
)
from .datatype import astype, wrap_param
from .index import dynamic_strided_slice, strided_slice, take
from .linear_algebra import einsum, linear, matmul
from .manipulate import (
    broadcast_to,
    collapse_sum_like,
    collapse_sum_to,
    concat,
    expand_dims,
    flatten,
    flip,
    gather_elements,
    gather_nd,
    layout_transform,
    one_hot,
    permute_dims,
    repeat,
    reshape,
    scatter_elements,
    scatter_nd,
    split,
    squeeze,
    tile,
)
from .mask import masked_fill
from .qdq import dequantize, quantize
from .sampling import multinomial_from_uniform
from .search import argmax, argmin, where
from .set import nonzero, unique
from .sorting import argsort, sort, topk
from .statistical import cumprod, cumsum, max, mean, min, prod, std, sum, variance
from .ternary import ewise_fma
from .unary import (
    abs,
    acos,
    acosh,
    asin,
    asinh,
    atan,
    atanh,
    bitwise_not,
    ceil,
    clip,
    cos,
    cosh,
    erf,
    exp,
    floor,
    isfinite,
    isinf,
    isnan,
    log,
    logical_not,
    negative,
    round,
    rsqrt,
    sigmoid,
    sign,
    sin,
    sinh,
    sqrt,
    square,
    tan,
    tanh,
)


def _register_op_make():
    # pylint: disable=import-outside-toplevel
    from .. import expr
    from . import _ffi_api

    expr._op_ffi_api = _ffi_api  # type: ignore


_register_op_make()
