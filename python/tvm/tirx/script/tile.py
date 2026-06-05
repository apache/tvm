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
"""Tile primitive shorthand namespace for TIRx TVMScript."""

import functools

from tvm.tirx import Buffer, BufferRegion

from .builder import tirx as _builder

_TILE_ARG_TYPES = (Buffer, BufferRegion)


def _get_arg(args, kwargs, index, name):
    if len(args) > index:
        return args[index]
    return kwargs.get(name)


def _require_buffer_arg(op_name, arg_name, value):
    if not isinstance(value, _TILE_ARG_TYPES):
        raise TypeError(
            f"Tx.{op_name} is tile-only and expects `{arg_name}` to be a Buffer "
            f"or BufferRegion; use T.{op_name} for expression/builtin calls"
        )


def _validate_tile_call(op_name, args, kwargs):
    dst = _get_arg(args, kwargs, 0, "dst")
    _require_buffer_arg(op_name, "dst", dst)

    if op_name in {"cast", "max", "min", "permute_layout", "silu"}:
        src = _get_arg(args, kwargs, 1, "src")
        _require_buffer_arg(op_name, "src", src)
    elif op_name in {"sqrt", "exp", "exp2", "reciprocal"}:
        src = _get_arg(args, kwargs, 1, "src")
        if src is not None:
            _require_buffer_arg(op_name, "src", src)


def _tile_scoped_op(op_name):
    scoped_op = getattr(_builder, op_name)

    @functools.wraps(scoped_op._fn)  # pylint: disable=protected-access
    def wrapper(*args, scope=None, **kwargs):
        _validate_tile_call(op_name, args, kwargs)
        return scoped_op._fn(*args, scope=scope, **kwargs)  # pylint: disable=protected-access

    return _builder.ScopedOp(wrapper)


_SCOPED_TILE_OP_NAMES = [
    "add",
    "binary_chain",
    "binary_reduce",
    "cast",
    "copy",
    "copy_async",
    "exp",
    "exp2",
    "fdiv",
    "fill",
    "fma",
    "gemm",
    "gemm_async",
    "max",
    "maximum",
    "memset",
    "min",
    "minimum",
    "mul",
    "permute_layout",
    "reciprocal",
    "reduce_negate",
    "select",
    "silu",
    "sqrt",
    "sub",
    "sum",
    "unary_reduce",
    "zero",
]

for _op_name in _SCOPED_TILE_OP_NAMES:
    globals()[_op_name] = _tile_scoped_op(_op_name)

cluster = _builder.ScopeNamespace("cluster", "cluster")
cta = _builder.ScopeNamespace("cta", "cta")
wg = _builder.ScopeNamespace("warpgroup", "wg")
warpgroup = _builder.ScopeNamespace("warpgroup", "warpgroup")
warp = _builder.ScopeNamespace("warp", "warp")
thread = _builder.ScopeNamespace("thread", "thread")

compose_op = _builder.compose_op
tvm_kernel_replace_point = _builder.tvm_kernel_replace_point

__all__ = [
    *_SCOPED_TILE_OP_NAMES,
    "cluster",
    "compose_op",
    "cta",
    "thread",
    "tvm_kernel_replace_point",
    "warp",
    "warpgroup",
    "wg",
]
