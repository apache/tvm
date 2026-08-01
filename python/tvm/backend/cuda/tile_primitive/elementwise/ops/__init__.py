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

"""Per-op data model + ALL_OPS registry.

``OpSpec`` describes one elementwise op. ``VecImpl`` describes one packed-PTX
or CUDA-intrinsic emit available for that op (e.g. ``add_f32x2``); a list of
these (widest-first) lets ``reg.py``/``smem.py`` pick the widest matching
both the layout and the op's available intrinsics, like copy picks
``copy_{128,64,32,16,8}b`` based on bit-width and tail contiguity.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from tvm.ir.expr import Expr
from tvm.tirx import BufferRegion, TilePrimitiveCall


@dataclass
class SrcSpec:
    """One operand of an elementwise op.

    Either a ``BufferRegion`` (per-element load) or a scalar ``Expr``.
    ``index_fn``, if given, derives per-element indices for broadcasting srcs:
        ``index_fn(dst_indices, dst_start, dst_extent, src_start, src_extent) -> list[Expr]``
    Default is the standard ``get_indices`` over the src's own region.
    """

    buf_region: BufferRegion | None = None
    scalar: Expr | None = None
    index_fn: Callable | None = None

    @property
    def is_scalar(self) -> bool:
        return self.scalar is not None

    @property
    def buffer(self):
        return self.buf_region.buffer if self.buf_region is not None else None


@dataclass
class Plan:
    """Parsed elementwise op ready for a schedule to consume."""

    dst: BufferRegion
    srcs: list[SrcSpec]
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class VecImpl:
    """One packed-vector implementation registered for an op.

    Mirrors the ``copy_{Nb}`` menu in copy: each entry says "I can process
    ``vec_len`` consecutive elements per call". The schedule picks the widest
    one whose ``vec_len`` divides the layout's contig tail AND whose
    ``applies()`` returns ``True``.
    """

    vec_len: int  # elements per packed call
    applies: Callable[[TilePrimitiveCall, Any, Plan], tuple[bool, str | None]]
    # emit(dst_ptr, src_ptrs, extras) -> Stmt
    #   dst_ptr: typed ptr to ``vec_len`` consecutive dst elements
    #   src_ptrs[i]: typed ptr to ``vec_len`` consecutive src[i] elements,
    #                OR a scalar Expr if src[i].is_scalar.
    # Runs in Python at @T.prim_func build time -- branching on src kind is a
    # normal Python ``if``, not a TVMScript shape limitation. This is what
    # collapses the old 4x2 shape-explosion in schema.py's factories.
    emit: Callable


@dataclass
class OpSpec:
    """Metadata for an elementwise op."""

    name: str
    # parse(op_call) -> (Plan, msg|None); msg explains why parse failed.
    parse: Callable[[TilePrimitiveCall], tuple[Plan | None, str | None]]
    # Scalar compute used by the fallback emit path (wrapped in Tx.vectorized).
    # compute_scalar(src_vals_at_one_idx, extras, dst_dtype) -> Expr
    compute_scalar: Callable[[list, dict, str], Any]
    # Optional dtype check on plan.extras (e.g. unary bias/scale dtype agreement).
    check_extras: Callable | None = None
    # Widest-first vec impls. Schedule picks first matching layout+applies.
    vec_impls: list[VecImpl] = field(default_factory=list)


def _build_all_ops() -> dict[str, OpSpec]:
    """Aggregate per-family op specs. Deferred imports avoid cycles
    (vec_emit/* imports VecImpl from this module)."""
    from .binary import BINARY_OPS
    from .cast import CAST_OPS
    from .fma import FMA_OPS
    from .unary import UNARY_OPS

    return {**UNARY_OPS, **BINARY_OPS, **CAST_OPS, **FMA_OPS}


ALL_OPS: dict[str, OpSpec] = _build_all_ops()


__all__ = ["ALL_OPS", "OpSpec", "Plan", "SrcSpec", "VecImpl"]
