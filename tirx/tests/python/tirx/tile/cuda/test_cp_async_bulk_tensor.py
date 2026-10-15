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
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Explicit TMA views, transfer regions, and descriptor encoding.

Caller views specify the source and destination physical addresses explicitly.
"""

import math
from dataclasses import dataclass
from functools import reduce
from operator import mul

import numpy as np
import pytest
import tvm_ffi

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.backend.cuda.layout import mma_atom_layout, mma_atom_shape, mma_shared_layout
from tvm_tirx.ir import PrimType, PtrType, Range, TensorRegion, TensorRegionType
from tvm_tirx.script import tirx as T
from tvm_tirx.sym import Analyzer
from tvm_tirx.tirx import IntImm, Var
from tvm_tirx.tirx.layout import ComposeLayout, S, TileLayout
from tvm_tirx.tirx.thread_scope import ThreadScope
from tvm_tirx.tirx.tile.dispatch import DispatchContext, run_dispatch

_TMA_OPS = {
    "tirx.ptx.cp_async_bulk_tensor_g2s_cta",
    "tirx.ptx.cp_async_bulk_tensor_g2s_cluster",
    "tirx.ptx.cp_async_bulk_tensor_s2g",
    "tirx.ptx.cp_reduce_async_bulk_tensor",
}


def _ptx_call_parts(call):
    """Decode one ptx Call: (modifier map, operand name -> args tuple).

    The dialect's Call layout is [operands..., pred?][slot tokens][pred marker];
    the operand positions follow the table's layout for the call's tokens, so
    tests name operands instead of hard-coding argument indices.
    """
    from tvm_tirx.backend.cuda.ptx.table import TABLE, mods, operand_layout

    entry = TABLE[call.op.name.removeprefix("tirx.ptx.")]
    n_slots = len(entry.slots)
    tokens = [str(a.value) for a in call.args[len(call.args) - n_slots - 1 : -1]]
    mod_map = mods(entry, tokens)
    operands = {
        slot.name: tuple(call.args[i : i + n]) for slot, i, n in operand_layout(entry, mod_map)
    }
    return mod_map, operands


def _unwrap_shared_addr(value):
    """Peel the dialect's generic-pointer -> shared-window conversion."""
    if isinstance(value, tvm_tirx.ir.Call) and value.op.name == "tirx.cuda.cvta_generic_to_shared":
        return value.args[0]
    return value


class _TMACounter:
    def __init__(self):
        self.loop_extents = []
        self.total = 0
        self.calls = []
        self.weighted_calls = []

    def _visit_for(self, op, visitor):
        self.loop_extents.append(op.extent)
        visitor.default_visit(op)
        self.loop_extents.pop()

    def _visit_evaluate(self, op, visitor):
        if isinstance(op.value, tvm_tirx.ir.Call) and op.value.op.name in _TMA_OPS:
            multiplier = reduce(mul, (int(extent) for extent in self.loop_extents), 1)
            self.total += multiplier
            self.calls.append(op.value)
            self.weighted_calls.append((op.value, multiplier))
        visitor.default_visit(op)

    def visit_stmt(self, stmt):
        tvm_ffi.structural_visit(
            stmt,
            [(tvm_tirx.ir.For, self._visit_for), (tvm_tirx.ir.Evaluate, self._visit_evaluate)],
        )


class _EncodeCollector:
    def __init__(self):
        self.calls = []

    def _visit_call(self, op):
        if isinstance(op.op, tvm_tirx.ir.Op) and op.op.name == "tirx.cuda.tensormap_encode_tiled":
            self.calls.append(op)

    def visit_stmt(self, stmt):
        tvm_ffi.structural_walk(stmt, (tvm_tirx.ir.Call, self._visit_call), order="post")


class _SelectCollector:
    def __init__(self):
        self.nodes = []

    def _visit_select(self, op):
        self.nodes.append(op)

    def visit_stmt(self, stmt):
        tvm_ffi.structural_walk(stmt, (tvm_tirx.tirx.Select, self._visit_select), order="post")


class _PrefetchCollector:
    def __init__(self):
        self.names = []

    def _visit_call(self, op):
        if isinstance(op.op, tvm_tirx.ir.Op) and op.op.name == "tirx.ptx.prefetch":
            addr = op.args[0]
            # A tensormap address arrives as a u64 handle, so the dialect
            # coerces it with an explicit reinterpret before the call.
            if isinstance(addr, tvm_tirx.ir.Call) and addr.op.name == "tirx.reinterpret":
                addr = addr.args[0]
            if (
                isinstance(addr, tvm_tirx.ir.Call)
                and addr.op.name == "tirx.address_of"
                and isinstance(addr.args[0], Var)
            ):
                self.names.append(addr.args[0].name)

    def visit_stmt(self, stmt):
        tvm_ffi.structural_walk(stmt, (tvm_tirx.ir.Call, self._visit_call), order="post")


def _plain_layout(shape, strides=None):
    shape = tuple(shape)
    if strides is None:
        return TileLayout(S[shape])
    return TileLayout(S[shape : tuple(strides)])


def _ranges(region):
    return [Range.from_min_extent(start, extent) for start, extent in region]


def _count_tma(stmt):
    counter = _TMACounter()
    counter.visit_stmt(stmt.body if isinstance(stmt, tvm_tirx.tirx.Function) else stmt)
    return counter


def _collect_encodes(stmts):
    collector = _EncodeCollector()
    for stmt in stmts:
        collector.visit_stmt(stmt)
    return collector.calls


def _encode_signature(call):
    rank = call.attrs.rank
    cursor = 2
    dims = tuple(call.args[cursor : cursor + rank])
    cursor += rank
    strides = tuple(call.args[cursor : cursor + rank - 1])
    cursor += rank - 1
    boxes = tuple(call.args[cursor : cursor + rank])
    cursor += rank
    element_strides = tuple(call.args[cursor : cursor + rank])
    cursor += rank
    enums = (
        call.attrs.interleave,
        call.attrs.swizzle,
        call.attrs.l2_promotion,
        call.attrs.oob_fill,
    )
    forced_dtype = call.attrs.force_cu_dtype if call.attrs.force_cu_dtype >= 0 else None
    return {
        "dtype": str(call.attrs.descriptor_dtype),
        "rank": rank,
        "base": call.args[1],
        "dims": dims,
        "strides": strides,
        "boxes": boxes,
        "element_strides": element_strides,
        "enums": enums,
        "forced_dtype": forced_dtype,
    }


def _ints(values):
    return tuple(int(value) for value in values)


def _tma_case(case_id, **kwargs):
    return pytest.param(case_id, kwargs, id=case_id)


_BASELINE_AUTO_CASES = [
    ("g2s-2d-8x256", "float16", 3),
    ("g2s-2d-8x256-swizzle2", "float16", 2),
    ("g2s-2d-8x256-swizzle1", "float16", 1),
    ("g2s-2d-8x256-swizzle0", "float16", 0),
    ("g2s-2d-8x256-int8", "int8", 3),
    ("g2s-2d-8x256-bf16", "bfloat16", 3),
    ("g2s-2d-8x256-fp32", "float32", 3),
    ("g2s-2d-8x256-uint8", "uint8", 3),
    ("g2s-2d-8x256-fp8e4m3", "float8_e4m3fn", 3),
    ("g2s-2d-8x256-fp8e5m2", "float8_e5m2", 3),
]

TMA_CASES = [
    *[
        _tma_case(
            case_id,
            g_shape=(8, 256),
            g_region=((0, 8), (0, 256)),
            s_shape=(8, 256),
            s_region=((0, 8), (0, 256)),
            g_layout=_plain_layout((8, 256)),
            s_layout=mma_shared_layout(dtype, swizzle, (8, 256)),
            dtype=dtype,
        )
        for case_id, dtype, swizzle in _BASELINE_AUTO_CASES
    ],
    _tma_case(
        "g2s-3d-shared-64x256",
        g_shape=(64, 256),
        g_region=((0, 64), (0, 256)),
        s_shape=(3, 64, 256),
        s_region=((1, 1), (0, 64), (0, 256)),
        g_layout=_plain_layout((64, 256)),
        s_layout=mma_shared_layout("float16", 3, (3, 64, 256)),
    ),
    _tma_case(
        "g2s-2d-32x512-atom",
        g_shape=(32, 512),
        g_region=((0, 32), (0, 512)),
        s_shape=(32, 512),
        s_region=((0, 32), (0, 512)),
        g_layout=_plain_layout((32, 512)),
        s_layout=(
            mma_atom_layout("float16", 3)
            .tile_to((16, 256), mma_atom_shape("float16", 3))
            .tile_to((32, 512), (16, 256))
        ),
    ),
    _tma_case(
        "g2s-2d-partial-8192",
        g_shape=(8192, 8192),
        g_region=((0, 128), (0, 64)),
        s_shape=(128, 64),
        s_region=((0, 128), (0, 64)),
        g_layout=_plain_layout((8192, 8192)),
        s_layout=mma_shared_layout("float16", 3, (128, 64)),
    ),
    _tma_case(
        "g2s-edge-4d-shared-128x64",
        g_shape=(128, 64),
        g_region=((0, 128), (0, 64)),
        s_shape=(2, 2, 128, 64),
        s_region=((0, 1), (0, 1), (0, 128), (0, 64)),
        g_layout=_plain_layout((128, 64)).canonicalize(),
        s_layout=mma_shared_layout("float16", 3, (2, 2, 128, 64)).canonicalize(),
    ),
    _tma_case(
        "g2s-edge-partial-offset",
        g_shape=(128, 64),
        g_region=((64, 24), (0, 64)),
        s_shape=(2, 2, 24, 64),
        s_region=((0, 1), (0, 1), (0, 24), (0, 64)),
        g_layout=_plain_layout((128, 64)).canonicalize(),
        s_layout=mma_shared_layout("float16", 3, (2, 2, 24, 64)).canonicalize(),
    ),
    _tma_case(
        "g2s-edge-large-region",
        g_shape=(256, 64),
        g_region=((128, 128), (0, 64)),
        s_shape=(256, 64),
        s_region=((0, 128), (0, 64)),
        g_layout=_plain_layout((256, 64)).canonicalize(),
        s_layout=mma_shared_layout("float16", 3, (256, 64)).canonicalize(),
    ),
    _tma_case(
        "g2s-partial-3d-shared-a",
        g_shape=(128, 256),
        g_region=((0, 32), (0, 64)),
        s_shape=(6, 128, 64),
        s_region=((0, 1), (0, 32), (0, 64)),
        g_layout=_plain_layout((128, 256)).canonicalize(),
        s_layout=mma_shared_layout("float16", 3, (6, 128, 64)).canonicalize(),
    ),
    _tma_case(
        "g2s-partial-3d-shared-b",
        g_shape=(256, 512),
        g_region=((0, 64), (0, 64)),
        s_shape=(4, 256, 64),
        s_region=((1, 1), (0, 64), (0, 64)),
        g_layout=_plain_layout((256, 512)).canonicalize(),
        s_layout=mma_shared_layout("float16", 3, (4, 256, 64)).canonicalize(),
    ),
    _tma_case(
        "g2s-3d-full-contiguous",
        g_shape=(4, 32, 64),
        g_region=((0, 4), (0, 32), (0, 64)),
        s_shape=(4, 32, 64),
        s_region=((0, 4), (0, 32), (0, 64)),
        g_layout=_plain_layout((4, 32, 64)),
        s_layout=_plain_layout((4, 32, 64)),
    ),
    _tma_case(
        "g2s-3d-partial-contiguous",
        g_shape=(8, 16, 128),
        g_region=((0, 4), (0, 16), (0, 128)),
        s_shape=(4, 16, 128),
        s_region=((0, 4), (0, 16), (0, 128)),
        g_layout=_plain_layout((8, 16, 128)),
        s_layout=_plain_layout((4, 16, 128)),
    ),
    _tma_case(
        "g2s-3d-stride-gap-outer",
        g_shape=(8, 32, 64),
        g_region=((0, 8), (0, 32), (0, 64)),
        s_shape=(8, 32, 64),
        s_region=((0, 8), (0, 32), (0, 64)),
        g_layout=_plain_layout((8, 32, 64)),
        s_layout=_plain_layout((8, 32, 64), (4096, 64, 1)),
    ),
    _tma_case(
        "g2s-4d-reorder-a",
        g_shape=(2, 128, 8, 64),
        g_region=((0, 1), (0, 128), (0, 1), (0, 64)),
        s_shape=(1, 1, 128, 64),
        s_region=((0, 1), (0, 1), (0, 128), (0, 64)),
        g_layout=_plain_layout((2, 128, 8, 64)).canonicalize(),
        s_layout=mma_shared_layout("float16", 3, (1, 1, 128, 64)).canonicalize(),
    ),
    _tma_case(
        "g2s-4d-reorder-b",
        g_shape=(4, 64, 4, 128),
        g_region=((0, 1), (0, 64), (0, 1), (0, 128)),
        s_shape=(1, 1, 64, 128),
        s_region=((0, 1), (0, 1), (0, 64), (0, 128)),
        g_layout=_plain_layout((4, 64, 4, 128)).canonicalize(),
        s_layout=mma_shared_layout("float16", 3, (1, 1, 64, 128)).canonicalize(),
    ),
    _tma_case(
        "g2s-multidim-4d-a",
        g_shape=(2, 2, 128, 64),
        g_region=((0, 1), (0, 1), (0, 128), (0, 64)),
        s_shape=(128, 64),
        s_region=((0, 128), (0, 64)),
        g_layout=_plain_layout((2, 2, 128, 64)).canonicalize(),
        s_layout=mma_shared_layout("float16", 3, (128, 64)),
    ),
    _tma_case(
        "g2s-multidim-4d-b",
        g_shape=(4, 64, 4, 128),
        g_region=((0, 1), (0, 64), (0, 1), (0, 128)),
        s_shape=(64, 128),
        s_region=((0, 64), (0, 128)),
        g_layout=_plain_layout((4, 64, 4, 128)).canonicalize(),
        s_layout=mma_shared_layout("float16", 3, (64, 128)),
    ),
    _tma_case(
        "g2s-multiphase-3x8x256",
        g_shape=(3, 8, 256),
        g_region=((0, 1), (0, 8), (0, 256)),
        s_shape=(8, 256),
        s_region=((0, 8), (0, 256)),
        g_layout=_plain_layout((3, 8, 256)),
        s_layout=mma_shared_layout("float16", 3, (8, 256)),
    ),
    _tma_case(
        "g2s-multiphase-5x64x256",
        g_shape=(5, 64, 256),
        g_region=((0, 1), (0, 64), (0, 256)),
        s_shape=(64, 256),
        s_region=((0, 64), (0, 256)),
        g_layout=_plain_layout((5, 64, 256)),
        s_layout=mma_shared_layout("float16", 3, (64, 256)),
    ),
    _tma_case(
        "g2s-multiphase-7x32x512-atom",
        g_shape=(7, 32, 512),
        g_region=((0, 1), (0, 32), (0, 512)),
        s_shape=(32, 512),
        s_region=((0, 32), (0, 512)),
        g_layout=_plain_layout((7, 32, 512)),
        s_layout=(
            mma_atom_layout("float16", 3)
            .tile_to((16, 256), mma_atom_shape("float16", 3))
            .tile_to((32, 512), (16, 256))
        ),
    ),
    _tma_case(
        "g2s-transpose-32x64",
        g_shape=(32, 64),
        g_region=((0, 32), (0, 64)),
        s_shape=(32, 64),
        s_region=((0, 32), (0, 64)),
        g_layout=_plain_layout((32, 64)),
        s_layout=_plain_layout((32, 64), (1, 32)),
    ),
    _tma_case(
        "g2s-transpose-64x32",
        g_shape=(64, 32),
        g_region=((0, 64), (0, 32)),
        s_shape=(64, 32),
        s_region=((0, 64), (0, 32)),
        g_layout=_plain_layout((64, 32)),
        s_layout=_plain_layout((64, 32), (1, 64)),
    ),
    _tma_case(
        "g2s-transpose-partial-region",
        g_shape=(128, 64),
        g_region=((0, 64), (0, 64)),
        s_shape=(64, 64),
        s_region=((0, 64), (0, 64)),
        g_layout=_plain_layout((128, 64)),
        s_layout=_plain_layout((64, 64), (1, 64)),
    ),
    _tma_case(
        "g2s-transpose-partial-offset",
        g_shape=(128, 64),
        g_region=((64, 64), (0, 32)),
        s_shape=(64, 32),
        s_region=((0, 64), (0, 32)),
        g_layout=_plain_layout((128, 64)),
        s_layout=_plain_layout((64, 32), (1, 64)),
    ),
    _tma_case(
        "g2s-non-prefix-compact-elides",
        g_shape=(16, 16, 128, 128),
        g_region=((3, 1), (4, 1), (0, 128), (0, 128)),
        s_shape=(128, 128),
        s_region=((0, 128), (0, 128)),
        g_layout=_plain_layout((16, 16, 128, 128), (1024 * 128, 128, 1024, 1)),
        s_layout=_plain_layout((128, 128)),
    ),
    _tma_case(
        "g2s-oob-zero",
        g_shape=(128, 64),
        g_region=((120, 16), (0, 64)),
        s_shape=(16, 64),
        s_region=((0, 16), (0, 64)),
        g_layout=_plain_layout((128, 64)),
        s_layout=mma_shared_layout("float16", 3, (16, 64)),
        config={"oob": "zero"},
    ),
    _tma_case(
        "g2s-oob-nan",
        g_shape=(128, 64),
        g_region=((120, 16), (0, 64)),
        s_shape=(16, 64),
        s_region=((0, 16), (0, 64)),
        g_layout=_plain_layout((128, 64)),
        s_layout=mma_shared_layout("float16", 3, (16, 64)),
        config={"oob": "nan"},
    ),
    _tma_case(
        "g2s-fa4-q",
        g_shape=(1, 2048, 32, 128),
        g_region=((0, 1), (0, 32), (0, 4), (0, 128)),
        s_shape=(2, 128, 128),
        s_region=((0, 1), (0, 128), (0, 128)),
        g_layout=TileLayout(S[(1, 2048, 32, 128)]),
        s_layout=mma_shared_layout("float16", 3, (2, 128, 128)),
    ),
    _tma_case(
        "g2s-fa4-k",
        g_shape=(1, 2048, 8, 128),
        g_region=((0, 1), (0, 128), (0, 1), (0, 128)),
        s_shape=(3, 128, 128),
        s_region=((0, 1), (0, 128), (0, 128)),
        g_layout=TileLayout(S[(1, 2048, 8, 128)]),
        s_layout=mma_shared_layout("float16", 3, (3, 128, 128)),
    ),
    _tma_case(
        "g2s-fa4-v",
        g_shape=(1, 2048, 8, 128),
        g_region=((0, 1), (0, 128), (0, 1), (0, 128)),
        s_shape=(3, 128, 128),
        s_region=((0, 1), (0, 128), (0, 128)),
        g_layout=TileLayout(S[(1, 2048, 8, 128)]),
        s_layout=mma_shared_layout("float16", 3, (3, 128, 128)),
    ),
    *[
        _tma_case(
            case_id,
            g_shape=(3, 8, 256),
            g_region=((0, 1), (0, 8), (0, 256)),
            s_shape=(8, 256),
            s_region=((0, 8), (0, 256)),
            g_layout=_plain_layout((3, 8, 256)),
            s_layout=mma_shared_layout(dtype, swizzle, (8, 256)),
            dtype=dtype,
            direction="s2g",
        )
        for case_id, dtype, swizzle in [
            ("s2g-multiphase-3x8x256", "float16", 3),
            ("s2g-multiphase-3x8x256-swizzle2", "float16", 2),
            ("s2g-multiphase-3x8x256-swizzle0", "float16", 0),
            ("s2g-multiphase-3x8x256-int8", "int8", 3),
            ("s2g-multiphase-3x8x256-fp32", "float32", 3),
        ]
    ],
    _tma_case(
        "s2g-multiphase-5x64x256",
        g_shape=(5, 64, 256),
        g_region=((0, 1), (0, 64), (0, 256)),
        s_shape=(64, 256),
        s_region=((0, 64), (0, 256)),
        g_layout=_plain_layout((5, 64, 256)),
        s_layout=mma_shared_layout("float16", 3, (64, 256)),
        direction="s2g",
    ),
    _tma_case(
        "s2g-multiphase-7x32x512-atom",
        g_shape=(7, 32, 512),
        g_region=((0, 1), (0, 32), (0, 512)),
        s_shape=(32, 512),
        s_region=((0, 32), (0, 512)),
        g_layout=_plain_layout((7, 32, 512)),
        s_layout=(
            mma_atom_layout("float16", 3)
            .tile_to((16, 256), mma_atom_shape("float16", 3))
            .tile_to((32, 512), (16, 256))
        ),
        direction="s2g",
    ),
    _tma_case(
        "s2g-keeps-multidim-coords",
        g_shape=(1024, 4, 1024),
        g_region=((128, 128), (1, 1), (32, 32)),
        s_shape=(128, 32),
        s_region=((0, 128), (0, 32)),
        g_layout=_plain_layout((1024, 4, 1024), (4 * 1024, 1024, 1)),
        s_layout=_plain_layout((128, 32), (32, 1)),
        dtype="bfloat16",
        direction="s2g",
    ),
    *[
        _tma_case(
            case_id,
            g_shape=(2, 128, 64),
            g_region=((0, 1), (0, 128), (0, 64)),
            s_shape=(128, 64),
            s_region=((0, 128), (0, 64)),
            g_layout=_plain_layout((2, 128, 64)),
            s_layout=mma_shared_layout("float16", 3, (128, 64)),
            direction="s2g",
            config=config,
        )
        for case_id, config in [
            ("s2g-oob-none", None),
            ("s2g-oob-zero", {"oob": "zero"}),
            ("s2g-oob-nan", {"oob": "nan"}),
        ]
    ],
    _tma_case(
        "reject-unknown-oob",
        g_shape=(3, 8, 256),
        g_region=((0, 1), (0, 8), (0, 256)),
        s_shape=(8, 256),
        s_region=((0, 8), (0, 256)),
        g_layout=_plain_layout((3, 8, 256)),
        s_layout=mma_shared_layout("float16", 3, (8, 256)),
        config={"oob": "bogus"},
    ),
    _tma_case(
        "reject-g2s-nan-on-non-float",
        g_shape=(128, 64),
        g_region=((120, 16), (0, 64)),
        s_shape=(16, 64),
        s_region=((0, 16), (0, 64)),
        g_layout=_plain_layout((128, 64)),
        s_layout=_plain_layout((16, 64)),
        dtype="int8",
        config={"oob": "nan"},
    ),
    _tma_case(
        "reject-s2g-nan-on-non-float",
        g_shape=(2, 128, 64),
        g_region=((0, 1), (0, 128), (0, 64)),
        s_shape=(128, 64),
        s_region=((0, 128), (0, 64)),
        g_layout=_plain_layout((2, 128, 64)),
        s_layout=_plain_layout((128, 64)),
        dtype="int8",
        direction="s2g",
        config={"oob": "nan"},
    ),
]


@dataclass(frozen=True)
class _TMAGolden:
    dtype: str
    dims: tuple
    strides: tuple
    boxes: tuple
    coordinates: tuple
    enums: tuple


def _tma_golden(
    dtype,
    dims,
    strides,
    boxes,
    coordinates=None,
    enums=(0, 3, 2, 0),
):
    return _TMAGolden(
        dtype=dtype,
        dims=dims,
        strides=strides,
        boxes=boxes,
        coordinates=(0,) * len(dims) if coordinates is None else coordinates,
        enums=enums,
    )


_TMA_CASE_GOLDENS = {
    "g2s-2d-8x256": _tma_golden("float16", (64, 8, 4), (512, 128), (64, 8, 4)),
    "g2s-2d-8x256-swizzle2": _tma_golden(
        "float16", (32, 8, 8), (512, 64), (32, 8, 8), enums=(0, 2, 2, 0)
    ),
    "g2s-2d-8x256-swizzle1": _tma_golden(
        "float16", (16, 8, 16), (512, 32), (16, 8, 16), enums=(0, 1, 2, 0)
    ),
    "g2s-2d-8x256-swizzle0": _tma_golden(
        "float16", (8, 8, 32), (512, 16), (8, 8, 32), enums=(0, 0, 2, 0)
    ),
    "g2s-2d-8x256-int8": _tma_golden("int8", (128, 8, 2), (256, 128), (128, 8, 2)),
    "g2s-2d-8x256-bf16": _tma_golden("bfloat16", (64, 8, 4), (512, 128), (64, 8, 4)),
    "g2s-2d-8x256-fp32": _tma_golden("float32", (32, 8, 8), (1024, 128), (32, 8, 8)),
    "g2s-2d-8x256-uint8": _tma_golden("uint8", (128, 8, 2), (256, 128), (128, 8, 2)),
    "g2s-2d-8x256-fp8e4m3": _tma_golden("float8_e4m3fn", (128, 8, 2), (256, 128), (128, 8, 2)),
    "g2s-2d-8x256-fp8e5m2": _tma_golden("float8_e5m2", (128, 8, 2), (256, 128), (128, 8, 2)),
    "g2s-3d-shared-64x256": _tma_golden("float16", (64, 64, 4), (512, 128), (64, 64, 4)),
    "g2s-2d-partial-8192": _tma_golden("float16", (8192, 8192), (16384,), (64, 128)),
    "g2s-edge-4d-shared-128x64": _tma_golden("float16", (64, 128), (128,), (64, 128)),
    "g2s-edge-partial-offset": _tma_golden("float16", (64, 128), (128,), (64, 24), (0, 64)),
    "g2s-edge-large-region": _tma_golden("float16", (64, 256), (128,), (64, 128), (0, 128)),
    "g2s-partial-3d-shared-a": _tma_golden("float16", (256, 128), (512,), (64, 32)),
    "g2s-partial-3d-shared-b": _tma_golden("float16", (512, 256), (1024,), (64, 64)),
    "g2s-3d-full-contiguous": _tma_golden(
        "float16", (64, 128), (128,), (64, 128), enums=(0, 0, 2, 0)
    ),
    "g2s-3d-partial-contiguous": _tma_golden(
        "float16", (128, 128), (256,), (128, 64), enums=(0, 0, 2, 0)
    ),
    "g2s-4d-reorder-a": _tma_golden("float16", (64, 256, 8), (1024, 128), (64, 128, 1)),
    "g2s-4d-reorder-b": _tma_golden(
        "float16",
        (64, 64, 2, 4, 4),
        (1024, 128, 65536, 256),
        (64, 64, 2, 1, 1),
    ),
    "g2s-multidim-4d-a": _tma_golden(
        "float16", (64, 128, 2, 2), (128, 32768, 16384), (64, 128, 1, 1)
    ),
    "g2s-multidim-4d-b": _tma_golden(
        "float16",
        (64, 64, 2, 4, 4),
        (1024, 128, 65536, 256),
        (64, 64, 2, 1, 1),
    ),
    "g2s-multiphase-3x8x256": _tma_golden(
        "float16", (64, 8, 4, 3), (512, 128, 4096), (64, 8, 4, 1)
    ),
    "g2s-multiphase-5x64x256": _tma_golden(
        "float16", (64, 64, 4, 5), (512, 128, 32768), (64, 64, 4, 1)
    ),
    "g2s-non-prefix-compact-elides": _tma_golden(
        "float16",
        (128, 2048, 16),
        (2048, 256),
        (128, 128, 1),
        (0, 384, 4),
        (0, 0, 2, 0),
    ),
    "g2s-oob-zero": _tma_golden("float16", (64, 128), (128,), (64, 16), (0, 120)),
    "g2s-oob-nan": _tma_golden("float16", (64, 128), (128,), (64, 16), (0, 120), (0, 3, 2, 1)),
    "g2s-fa4-q": _tma_golden("float16", (64, 32, 2048, 2), (256, 8192, 128), (64, 4, 32, 2)),
    "g2s-fa4-k": _tma_golden("float16", (64, 2048, 16), (2048, 128), (64, 128, 2)),
    "g2s-fa4-v": _tma_golden("float16", (64, 2048, 16), (2048, 128), (64, 128, 2)),
    "s2g-multiphase-3x8x256": _tma_golden(
        "float16", (64, 8, 4, 3), (512, 128, 4096), (64, 8, 4, 1)
    ),
    "s2g-multiphase-3x8x256-swizzle2": _tma_golden(
        "float16",
        (32, 8, 8, 3),
        (512, 64, 4096),
        (32, 8, 8, 1),
        enums=(0, 2, 2, 0),
    ),
    "s2g-multiphase-3x8x256-swizzle0": _tma_golden(
        "float16",
        (8, 8, 32, 3),
        (512, 16, 4096),
        (8, 8, 32, 1),
        enums=(0, 0, 2, 0),
    ),
    "s2g-multiphase-3x8x256-int8": _tma_golden(
        "int8", (128, 8, 2, 3), (256, 128, 2048), (128, 8, 2, 1)
    ),
    "s2g-multiphase-3x8x256-fp32": _tma_golden(
        "float32", (32, 8, 8, 3), (1024, 128, 8192), (32, 8, 8, 1)
    ),
    "s2g-multiphase-5x64x256": _tma_golden(
        "float16", (64, 64, 4, 5), (512, 128, 32768), (64, 64, 4, 1)
    ),
    "s2g-keeps-multidim-coords": _tma_golden(
        "bfloat16",
        (1024, 1024, 4),
        (8192, 2048),
        (32, 128, 1),
        (32, 128, 1),
        (0, 0, 2, 0),
    ),
    "s2g-oob-none": _tma_golden("float16", (64, 256), (128,), (64, 128)),
}


def _region(tensor, ranges=None):
    ranges = ranges or [(0, x) for x in tensor.ty.shape]
    return TensorRegion(tensor, _ranges(ranges), ty=TensorRegionType())


def _context(arch="sm_100a", ranges=None):
    return DispatchContext(
        tvm_tirx.target.Target({"kind": "cuda", "arch": arch}),
        ThreadScope("cuda.thread"),
        {},
        ranges or {},
    )


def _emit(opcode, global_view, shared_view, *, box, coords, sctx=None, **fields):
    sctx = sctx or _context()
    global_view = _region(global_view) if isinstance(global_view, Var) else global_view
    shared_view = _region(shared_view) if isinstance(shared_view, Var) else shared_view
    if ".global.shared::cta" in opcode:
        operands = (global_view, shared_view)
    else:
        operands = (shared_view, global_view)
        fields.setdefault("mbar", Var("mbar", "handle"))
    with sctx.target:
        impl = run_dispatch(
            T.cuda.tile.apply(T.ptx[opcode], *operands, box=box, coords=coords, **fields), sctx
        )
    return impl, list(sctx.callbacks.get("host_init_stmt", [])), sctx


def _load(rank, mode="tile"):
    return (
        f"cp.async.bulk.tensor.{rank}d.shared::cluster.global.{mode}"
        ".mbarrier::complete_tx::bytes.cta_group::1"
    )


def _store(rank):
    return f"cp.async.bulk.tensor.{rank}d.global.shared::cta.tile.bulk_group"


def _address_array(layout, shape, ranges, elem_offset=0):
    """Evaluate a fixed layout on all old region coordinates, independently of TMA."""
    extent = math.prod(int(e) for _, e in ranges)
    flat = np.arange(extent, dtype=np.int64)
    indices = []
    variables = []
    values = {}
    for i, (start, count) in enumerate(ranges):
        v = Var(f"i{i}", "int64")
        indices.append(v)
        variables.append(v)
        values[v] = start + flat // math.prod(int(e) for _, e in ranges[i + 1 :]) % int(count)
    expr = Analyzer().simplify(layout.apply(*indices, shape=shape)["m"] + elem_offset)

    def evaluate(expr):
        if isinstance(expr, IntImm):
            return int(expr)
        if isinstance(expr, Var):
            return values[expr]
        if isinstance(expr, tvm_tirx.tirx.Cast):
            return evaluate(expr.value)
        ops = {
            "Add": np.add,
            "Sub": np.subtract,
            "Mul": np.multiply,
            "FloorDiv": np.floor_divide,
            "FloorMod": np.remainder,
            "BitwiseXor": np.bitwise_xor,
            "BitwiseAnd": np.bitwise_and,
            "LShift": np.left_shift,
            "RShift": np.right_shift,
        }
        if type(expr).__name__ in ops:
            return ops[type(expr).__name__](evaluate(expr.a), evaluate(expr.b))
        if isinstance(expr, tvm_tirx.ir.Call):
            calls = {
                "tirx.bitwise_xor": np.bitwise_xor,
                "tirx.bitwise_and": np.bitwise_and,
                "tirx.shift_right": np.right_shift,
                "tirx.shift_left": np.left_shift,
            }
            return calls[expr.op.name](*(evaluate(a) for a in expr.args))
        raise TypeError(expr)

    return np.broadcast_to(evaluate(expr), (extent,))


def _original_addresses(kwargs):
    gshape = kwargs["g_shape"]
    sshape = kwargs.get("s_shape", gshape)
    gr = kwargs.get("g_region", [(0, x) for x in gshape])
    sr = kwargs.get("s_region", [(0, x) for x in sshape])
    ga = _address_array(
        kwargs.get("g_layout", _plain_layout(gshape)), gshape, gr, kwargs.get("g_elem_offset", 0)
    )
    sa = _address_array(
        kwargs.get("s_layout", _plain_layout(sshape)), sshape, sr, kwargs.get("s_elem_offset", 0)
    )
    return ga, sa


def _recipe_views(kwargs, recipe):
    """Declare the caller's explicit zero-copy aliases on the original storage."""
    dtype = kwargs.get("dtype", "float16")
    bits = tvm_tirx.DataType(recipe.dtype).bits
    original_bits = tvm_tirx.DataType(dtype).bits
    ga, sa = _original_addresses(kwargs)
    shape = tuple(reversed(recipe.dims))
    strides = tuple(reversed((1, *(s * 8 // bits for s in recipe.strides))))
    box = tuple(reversed(recipe.boxes))
    g = tvm_tirx.tirx.decl_tensor(
        kwargs["g_shape"], dtype, "global_storage", layout=kwargs.get("g_layout")
    )
    sshape = kwargs.get("s_shape", kwargs["g_shape"])
    s = tvm_tirx.tirx.decl_tensor(
        sshape,
        dtype,
        "shared_storage",
        scope="shared.dyn",
        layout=kwargs.get("s_layout"),
        data_alignment=1024,
    )
    gv = tvm_tirx.tirx.decl_tensor(
        shape, recipe.dtype, "global_view", data=g.data, layout=_plain_layout(shape, strides)
    )
    assert int(sa.min()) * original_bits % bits == 0
    base = int(sa.min()) * original_bits // bits
    layout = _plain_layout(box)
    if recipe.enums[1]:
        layout = ComposeLayout(7 - int(math.log2(bits)), recipe.enums[1], 3, layout)
    sv = tvm_tirx.tirx.decl_tensor(
        box,
        recipe.dtype,
        "shared_view",
        data=s.data,
        scope="shared.dyn",
        layout=layout,
        elem_offset=base,
        data_alignment=1024,
    )
    # Compare address PAIRS, not separate address sets: a transpose can preserve
    # both sets while copying every element to the wrong destination.
    coords = np.indices(box, dtype=np.int64).reshape(len(box), -1)
    coords += np.asarray(tuple(reversed(recipe.coordinates)))[:, None]
    new_ga = np.sum(coords * np.asarray(strides)[:, None], axis=0)
    new_sa = _address_array(layout, box, [(0, x) for x in box], base)

    def bytes_at(address, width):
        return (address[:, None] * width + np.arange(width)[None, :]).reshape(-1)

    ga, sa = (bytes_at(x, original_bits // 8) for x in (ga, sa))
    new_ga, new_sa = (bytes_at(x, bits // 8) for x in (new_ga, new_sa))
    original = np.stack((ga, sa), axis=1)
    explicit = np.stack((new_ga, new_sa), axis=1)
    np.testing.assert_array_equal(original[np.argsort(sa)], explicit[np.argsort(new_sa)])
    return gv, sv


@pytest.mark.parametrize(
    "case_id,kwargs", [p for p in TMA_CASES if p.values[0] in _TMA_CASE_GOLDENS]
)
def test_tma_codegen(case_id, kwargs):
    expected = _TMA_CASE_GOLDENS[case_id]
    gv, sv = _recipe_views(kwargs, expected)
    rank = len(expected.dims)
    opcode = _load(rank) if kwargs.get("direction", "g2s") == "g2s" else _store(rank)
    impl, host, _ = _emit(
        opcode,
        gv,
        sv,
        box=tuple(reversed(expected.boxes)),
        coords=tuple(reversed(expected.coordinates)),
        swizzle=expected.enums[1],
        l2_promotion=expected.enums[2],
        oob_fill=expected.enums[3],
    )
    counter = _count_tma(impl)
    assert counter.total == 1
    mod_map, operands = _ptx_call_parts(counter.calls[0])
    assert mod_map["dim"] == f"{rank}d"
    assert _ints(operands["coords"]) == expected.coordinates
    encodes = _collect_encodes(host)
    assert len(encodes) == 1
    signature = _encode_signature(encodes[0])
    assert signature["dtype"] == expected.dtype
    assert _ints(signature["dims"]) == expected.dims
    assert _ints(signature["strides"]) == expected.strides
    assert _ints(signature["boxes"]) == expected.boxes
    assert _ints(signature["enums"]) == expected.enums


def _direct(
    opcode,
    *,
    g_shape,
    s_shape=None,
    g_layout=None,
    s_layout=None,
    dtype="float16",
    box=None,
    coords=None,
    s_region=None,
    g_elem_offset=0,
    s_elem_offset=0,
    sctx=None,
    **fields,
):
    s_shape = s_shape or g_shape
    g = tvm_tirx.tirx.decl_tensor(
        g_shape, dtype, "A", layout=g_layout or _plain_layout(g_shape), elem_offset=g_elem_offset
    )
    s = tvm_tirx.tirx.decl_tensor(
        s_shape,
        dtype,
        "A_smem",
        scope="shared.dyn",
        layout=s_layout or _plain_layout(s_shape),
        elem_offset=s_elem_offset,
        data_alignment=1024,
    )
    box = box or tuple(e for _, e in s_region) if s_region else box or s_shape
    coords = coords or (0,) * len(box)
    return _emit(opcode, g, _region(s, s_region), box=box, coords=coords, sctx=sctx, **fields)


@pytest.mark.parametrize("rank", range(1, 6))
def test_explicit_direct_rank_1_through_5(rank):
    shape = (8,) * rank
    impl, host, _ = _direct(_load(rank), g_shape=shape)
    assert _count_tma(impl).total == 1
    signature = _encode_signature(_collect_encodes(host)[0])
    assert signature["rank"] == rank
    assert _ints(signature["dims"]) == shape
    assert _ints(signature["boxes"]) == shape


def test_explicit_retains_symbolic_outer_stride_for_runtime_encode():
    row_stride = Var("row_stride", "int32")
    shape = (64, 64)
    impl, host, _ = _direct(
        _load(2), g_shape=shape, dtype="bfloat16", g_layout=_plain_layout(shape, (row_stride, 1))
    )
    assert _count_tma(impl).total == 1
    signature = _encode_signature(_collect_encodes(host)[0])
    assert Analyzer().can_prove_equal(signature["strides"][0], row_stride * 2)


@pytest.mark.parametrize("stride", [2, Var("inner_stride", "int32")])
def test_explicit_rejects_unknown_or_nonunit_implicit_inner_stride(stride):
    with pytest.raises(RuntimeError, match="innermost global stride"):
        _direct(_load(2), g_shape=(64, 64), g_layout=_plain_layout((64, 64), (64, stride)))


def test_explicit_view_base_offset_is_encoded_without_coordinate_rewrite():
    _, host, _ = _direct(_load(2), g_shape=(64, 64), g_elem_offset=8)
    signature = _encode_signature(_collect_encodes(host)[0])
    assert isinstance(signature["base"], tvm_tirx.ir.Call)
    assert signature["base"].op.name == "tirx.address_of"
    base = signature["base"].args[0]
    assert int(base.source.ty.elem_offset) * 2 == 16
    assert _ints(base.indices) == (0, 0)
    assert _ints(signature["dims"]) == (64, 64)


def test_explicit_oob_reduce_and_tf32_configs():
    _, host, _ = _direct(_load(2), g_shape=(64, 64), oob_fill=1)
    assert _ints(_encode_signature(_collect_encodes(host)[0])["enums"])[-1] == 1
    opcode = "cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.bulk_group"
    impl, _, _ = _direct(opcode, g_shape=(64, 64))
    assert _count_tma(impl).total == 1
    assert _count_tma(impl).calls[0].op.name == "tirx.ptx.cp_reduce_async_bulk_tensor"
    _, host, _ = _direct(_load(2), g_shape=(64, 32), dtype="float32", format="tf32")
    assert int(_encode_signature(_collect_encodes(host)[0])["forced_dtype"]) == 11


def test_explicit_never_repairs_rank_or_descriptor_units():
    with pytest.raises(RuntimeError, match="rank"):
        _direct(_load(5), g_shape=(2, 2, 2, 2, 2, 8))
    with pytest.raises(RuntimeError, match="multiples of 16"):
        _direct(_load(2), g_shape=(32, 8), s_shape=(32, 16), dtype="uint8")


def test_explicit_requires_box_linear_shared_slice():
    with pytest.raises(RuntimeError, match="compact/swizzled box"):
        _direct(_load(2), g_shape=(4, 64), s_layout=_plain_layout((4, 64), (128, 1)))


def test_explicit_allows_different_operand_ranks_with_equal_payload_bytes():
    @T.function
    def kernel(A: T.Tensor((8, 8), "float16")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=1))
        dyn = T.alloc_tensor(ty_args=[T.Tensor((65,), "uint64", scope="shared.dyn", align=128)])
        A_smem = T.decl_tensor(
            dyn.data,
            ty_args=[
                T.Tensor((64,), "float16", layout=T.TileLayout(T.S[64]), data=dyn.data, align=128)
            ],
        )
        mbar = T.decl_tensor(
            dyn.data, ty_args=[T.Tensor((1,), "uint64", elem_offset=16, data=dyn.data)]
        )
        T.cuda.tile.apply(
            T.ptx[
                "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::1"
            ],
            A_smem.view(8, 8),
            A,
            box=(8, 8),
            coords=(0, 0),
            mbar=mbar.ptr_to([0]),
        )

    assert _count_tma(_lower_module(kernel)["main"]).total == 1


def _lower_module(func, arch="sm_100a"):
    target = tvm_tirx.target.Target({"kind": "cuda", "arch": arch})
    with target:
        return tvm_tirx.tirx.transform.LowerTIRx()(tvm_tirx.IRModule({"main": func}))


def _compile_module(func, arch="sm_100a"):
    target = tvm_tirx.target.Target({"kind": "cuda", "arch": arch})
    return tvm_tirx.tirx.compile(tvm_tirx.IRModule({"main": func}), target=target, pipeline="tirx")


# These recipes retain the old coordinate-only dimensions, split storage axes,
# and leading-unit-dimension inputs. No recipe is derived from an input layout.
_EXTRA_RECIPES = [
    pytest.param(
        dict(g_shape=(8, 64)),
        _tma_golden("float16", (64, 8), (128,), (64, 8), enums=(0, 0, 0, 0)),
        id="bounded-box",
    ),
    pytest.param(
        dict(
            g_shape=(64, 128, 2, 4, 4096),
            s_shape=(64, 64, 2, 4),
            g_region=((0, 64), (32, 64), (0, 2), (0, 4), (7, 1)),
            g_layout=_plain_layout((64, 128, 2, 4, 4096), (1, 512, 256, 64, 65536)),
            s_layout=_plain_layout((64, 64, 2, 4), (1, 64, 4096, 8192)),
        ),
        _tma_golden(
            "float16",
            (64, 128, 2, 4, 4096),
            (1024, 512, 128, 131072),
            (64, 64, 2, 4, 1),
            (0, 32, 0, 0, 7),
            (0, 0, 0, 0),
        ),
        id="coordinate-only-dimension",
    ),
    pytest.param(
        dict(
            g_shape=(4, 256, 8, 512),
            s_shape=(1, 128, 1, 128),
            g_region=((2, 1), (64, 128), (3, 1), (128, 128)),
        ),
        _tma_golden(
            "float16",
            (512, 256, 4, 8),
            (8192, 2097152, 1024),
            (128, 128, 1, 1),
            (128, 64, 2, 3),
            (0, 0, 0, 0),
        ),
        id="unit-copy-dimensions",
    ),
    pytest.param(
        dict(
            g_shape=(24576, 7168),
            s_shape=(128, 128),
            g_region=((128, 128), (0, 128)),
            g_layout=TileLayout(S[(4, 6144, 7168) : (44040192, 7168, 1)]),
        ),
        _tma_golden("float16", (7168, 24576), (14336,), (128, 128), (0, 128), (0, 0, 0, 0)),
        id="split-coordinate-only-dimension",
    ),
    pytest.param(
        dict(
            g_shape=(1, 8192),
            s_shape=(256,),
            g_region=((0, 1), (512, 256)),
            g_layout=_plain_layout((1, 8192), (8192, 1)),
        ),
        _tma_golden("float16", (8192,), (), (256,), (512,), (0, 0, 0, 0)),
        id="leading-unit-dimension",
    ),
    pytest.param(
        dict(g_shape=(2, 2, 2, 2, 2, 8)),
        _tma_golden("float16", (256,), (), (256,), enums=(0, 0, 0, 0)),
        id="explicit-rank-fold",
    ),
    pytest.param(
        dict(g_shape=(32, 8), dtype="uint8"),
        _tma_golden("uint8", (256,), (), (256,), enums=(0, 0, 0, 0)),
        id="explicit-short-stride-fold",
    ),
    pytest.param(
        dict(g_shape=(64, 8), dtype="uint8"),
        _tma_golden("uint16", (256,), (), (256,), enums=(0, 0, 0, 0)),
        id="explicit-dtype-promotion",
    ),
    pytest.param(
        dict(
            g_shape=(8, 16, 32, 4, 4),
            g_region=((0, 1), (0, 4), (0, 32), (0, 4), (0, 4)),
            s_region=((0, 1), (0, 4), (0, 32), (0, 4), (0, 4)),
            dtype="uint8",
        ),
        _tma_golden("uint16", (256, 16, 8), (512, 8192), (256, 4, 1), enums=(0, 0, 0, 0)),
        id="promoted-inner-chain",
    ),
    pytest.param(
        dict(
            g_shape=(2048, 128),
            s_shape=(5, 128, 16),
            g_region=((128, 128), (16, 16)),
            s_region=((0, 1), (0, 128), (0, 16)),
            g_layout=TileLayout(S[(16, 4, 32, 32, 4) : (16384, 4, 16, 512, 1)]),
            s_layout=TileLayout(S[(5, 4, 32, 4, 4) : (2048, 4, 16, 512, 1)]),
            dtype="uint8",
        ),
        _tma_golden("uint16", (256, 32, 16), (512, 16384), (256, 4, 1), (0, 4, 1), (0, 0, 0, 0)),
        id="interleaved-logical-dimensions",
    ),
]


@pytest.mark.parametrize("kwargs,recipe", _EXTRA_RECIPES)
def test_caller_descriptor_view_preserves_original_addresses(kwargs, recipe):
    gv, sv = _recipe_views(kwargs, recipe)
    impl, host, _ = _emit(
        _load(len(recipe.dims)),
        gv,
        sv,
        box=tuple(reversed(recipe.boxes)),
        coords=tuple(reversed(recipe.coordinates)),
    )
    assert _count_tma(impl).total == 1
    signature = _encode_signature(_collect_encodes(host)[0])
    assert _ints(signature["dims"]) == recipe.dims
    assert _ints(signature["strides"]) == recipe.strides
    assert _ints(signature["boxes"]) == recipe.boxes


def test_caller_explicit_issue_coordinates_and_pointer():
    # Original shape/layout retained, now explicitly ask for 512 one-row boxes.
    shape = (512, 64)
    g = tvm_tirx.tirx.decl_tensor(shape, "float16", "A", layout="default")
    s = tvm_tirx.tirx.decl_tensor(
        (512, 1, 64), "float16", "S", scope="shared", layout="default", data_alignment=128
    )
    impl, host, _ = _emit(_load(2), g, s, box=(1, 64), coords=(0, 0), coord_steps=((1, 0),))
    assert _count_tma(impl).total == 512
    sig = _encode_signature(_collect_encodes(host)[0])
    assert _ints(sig["boxes"]) == (64, 1)
    _, operands = _ptx_call_parts(_count_tma(impl).calls[0])
    repeat = _count_tma(impl).loop_extents
    assert repeat == []
    row = operands["coords"][1]
    assert isinstance(row, Var)
    pointer = _unwrap_shared_addr(operands["dst_mem"][0])
    access = pointer.args[0]
    address = access.source.ty.layout.apply(*access.indices, shape=access.source.ty.shape)["m"]
    assert Analyzer().can_prove_equal(address, row * 64)
    at_seven = tvm_ffi.structural_map(
        address, (Var, lambda var: T.int32(7) if var.same_as(row) else var)
    )
    assert int(Analyzer().simplify(at_seven)) == 448


def test_dynamic_global_dimension_and_coordinates_remain_runtime_values():
    seq_len, row = Var("seq_len", "uint32"), Var("row", "uint32")
    impl, host, _ = _direct(
        _load(2),
        g_shape=(seq_len * T.uint32(64), 64),
        s_shape=(128, 64),
        dtype="uint8",
        coords=(row, 0),
    )
    sig = _encode_signature(_collect_encodes(host)[0])
    assert Analyzer().can_prove_equal(sig["dims"][1], seq_len * T.uint32(64))
    _, operands = _ptx_call_parts(_count_tma(impl).calls[0])
    tvm_tirx.ir.assert_structural_equal(operands["coords"][1], T.cast(row, "int32"))


@pytest.mark.parametrize(
    "dtype,cols,swizzle", [("int64", 64, 0), ("bfloat16", 32, 2), ("bfloat16", 64, 3)]
)
def test_sparse_decode_nope_rope_descriptor_dtypes_and_swizzles(dtype, cols, swizzle):
    layout = (
        _plain_layout((8, cols)) if swizzle == 0 else mma_shared_layout(dtype, swizzle, (8, cols))
    )
    _, host, _ = _direct(
        _load(2, "tile::gather4"),
        g_shape=(256, cols),
        s_shape=(8, cols),
        s_region=((0, 4), (0, cols)),
        box=(1, cols),
        dtype=dtype,
        s_layout=layout,
        gather4=(0, 1, 2, 3),
        swizzle=swizzle,
    )
    sig = _encode_signature(_collect_encodes(host)[0])
    assert sig["dtype"] == dtype
    assert int(sig["enums"][1]) == swizzle


@pytest.mark.parametrize("cols,row_stride,expected_bytes", [(64, 82, 656), (56, 72, 576)])
def test_sparse_decode_kv_descriptor_uses_tma_row_stride(cols, row_stride, expected_bytes):
    _, host, _ = _direct(
        _load(2, "tile::gather4"),
        g_shape=(256, cols),
        s_shape=(4, cols),
        g_layout=_plain_layout((256, cols), (row_stride, 1)),
        dtype="int64",
        box=(1, cols),
        gather4=(0, 1, 2, 3),
    )
    assert _ints(_encode_signature(_collect_encodes(host)[0])["strides"]) == (expected_bytes,)


@pytest.mark.gpu
@pytest.mark.parametrize("stride", [64, 80, 96])
def test_symbolic_stride_gpu(stride):
    @T.function
    def kernel(
        A: T.Tensor((8, stride), "float16", layout="default"),
        Out: T.Tensor((8, 64), "float16"),
        row_stride: T.int32,
    ):
        Av = A.view(8, 64, layout=T.TileLayout(T.S[(8, 64) : (row_stride, 1)]))
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
        lane = T.cuda.lane_id()
        shared = T.alloc_shared((8, 64), "float16", layout="default", align=128)
        mbar = T.alloc_shared((1,), "uint64", layout="default")
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(mbar.ptr_to([0]), T.uint32(1))
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()
        if lane == 0:
            T.cuda.tile.apply(
                T.ptx[
                    "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                ],
                shared,
                Av,
                box=(8, 64),
                coords=(0, 0),
                mbar=mbar.ptr_to([0]),
            )
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(mbar.ptr_to([0]), T.uint32(1024))
        T.cuda.mbarrier_wait(mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()
        for r, j in T.grid(8, 2):
            Out[r, lane + 32 * j] = shared[r, lane + 32 * j]

    f = _compile_module(kernel)
    x = np.arange(8 * stride, dtype="float16").reshape(8, stride)
    out = tvm_tirx.runtime.tensor(np.zeros((8, 64), "float16"), tvm_tirx.cuda())
    f(tvm_tirx.runtime.tensor(x, tvm_tirx.cuda()), out, stride)
    np.testing.assert_array_equal(out.numpy(), x[:, :64])


@pytest.mark.gpu
@pytest.mark.parametrize("step", range(1, 9))
@pytest.mark.parametrize("swizzle", range(4))
def test_element_strides_gpu(step, swizzle):
    cols = 4 << swizzle
    rows = (8 + step - 1) // step
    layout = ComposeLayout(2, swizzle, 3, _plain_layout((rows, cols)))

    @T.function
    def kernel(A: T.Tensor((8, cols), "uint32"), Out: T.Tensor((rows, cols), "uint32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
        lane = T.cuda.lane_id()
        shared = T.alloc_shared((rows, cols), "uint32", layout=layout, align=1024)
        mbar = T.alloc_shared((1,), "uint64", layout="default")
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(mbar.ptr_to([0]), T.uint32(1))
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()
        if lane == 0:
            T.cuda.tile.apply(
                T.ptx[
                    "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                ],
                shared,
                A,
                box=(8, cols),
                coords=(0, 0),
                element_strides=(step, 1),
                swizzle=swizzle,
                mbar=mbar.ptr_to([0]),
            )
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(mbar.ptr_to([0]), T.uint32(rows * cols * 4))
        T.cuda.mbarrier_wait(mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()
        if lane == 0:
            for r, c in T.grid(rows, cols):
                Out[r, c] = shared[r, c]

    f = _compile_module(kernel)
    x = np.arange(8 * cols, dtype="uint32").reshape(8, cols)
    out = tvm_tirx.runtime.tensor(np.zeros((rows, cols), "uint32"), tvm_tirx.cuda())
    f(tvm_tirx.runtime.tensor(x, tvm_tirx.cuda()), out)
    np.testing.assert_array_equal(out.numpy(), x[::step])


_SELECTOR_SOURCE = """
@T.function
def selector_gather(
    A: T.Tensor((256,64), "bfloat16"),
    B: T.Tensor((512,80), "bfloat16", layout="default"),
    Out: T.Tensor((4,64), "bfloat16"),
    flag: T.int32,
):
    B_view = B.sub[16:512,8:72]
    A_map: T.let[T.handle("tensormap")] = T.stack_alloca("tensormap",1)
    T.cuda.tensormap_encode_tiled(A_map,A.data,64,256,128,64,1,1,1,descriptor_dtype="bfloat16",rank=2)
    B_map: T.let[T.handle("tensormap")] = T.stack_alloca("tensormap",1)
    T.cuda.tensormap_encode_tiled(B_map,B_view.ptr_to([0,0]),64,496,160,64,1,1,1,descriptor_dtype="bfloat16",rank=2)
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1,block=128))
    tid=T.cuda.thread_idx("x")
    shared=T.alloc_shared((4,64),"bfloat16",layout="default",align=128)
    mbar=T.alloc_shared((1,),"uint64",layout="default")
    if tid==0:
        T.ptx.mbarrier.init.shared.b64(mbar.ptr_to([0]),T.uint32(1))
    T.ptx.fence.proxy.async_.shared__cta()
    T.cuda.cta_sync()
    if tid==0:
        selected: T.let = T.Select(flag!=0,T.address_of(B_map),T.address_of(A_map))
        T.cuda.tile.apply(T.ptx["cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::1"],shared,selected,box=(1,64),coords=(0,0),gather4=(1,2,3,4),mbar=mbar.ptr_to([0]))
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(mbar.ptr_to([0]),T.uint32(512))
    T.cuda.mbarrier_wait(mbar.ptr_to([0]),0)
    T.cuda.cta_sync()
    for step in T.serial(2):
        flat=step*128+tid
        Out[flat//64,flat%64]=shared[flat//64,flat%64]
"""


def _from_source(source):
    return tvm_tirx.script.from_source(source, {"T": T})


def test_explicit_gather_selector_encodes_each_map_selects_address_and_issues_once():
    func = _from_source(_SELECTOR_SOURCE)
    lowered = _lower_module(func)
    encodes = _collect_encodes([lowered["main"].body])
    assert len(encodes) == 2
    signatures = [_encode_signature(c) for c in encodes]
    by_dims = {_ints(s["dims"]): s for s in signatures}
    assert set(by_dims) == {(64, 256), (64, 496)}
    assert _ints(by_dims[(64, 256)]["strides"]) == (128,)
    assert _ints(by_dims[(64, 496)]["strides"]) == (160,)
    assert [_ints(s["boxes"]) for s in signatures] == [(64, 1), (64, 1)]
    selects = _SelectCollector()
    selects.visit_stmt(lowered["main"].body)
    assert len(selects.nodes) == 1
    assert selects.nodes[0].ty == PrimType("uint64")
    assert _count_tma(lowered["main"]).total == 1
    executable = _compile_module(func)
    cuda_source = executable.mod.imports[0].inspect_source()
    assert "flag != 0) ?" in cuda_source
    assert "if (flag" not in cuda_source
    assert (
        cuda_source.count(
            "tvm_builtin_ptx_cp_async_bulk_tensor_g2s_cluster_async_bulk_tensor_2d_shared__cluster_global_tile__gather4_mbarrier__complete_tx__bytes_cta_group__1("
        )
        == 2
    )


def test_nested_selector_parser_printer_roundtrip_and_lowering_remap():
    func = _from_source(_SELECTOR_SOURCE)
    reparsed = _from_source(func.script(extra_config={"tirx.prefix": "T"}))
    tvm_tirx.ir.assert_structural_equal(func, reparsed, map_free_vars=True)
    lowered = _lower_module(reparsed)
    assert len(_collect_encodes([lowered["main"].body])) == 2
    assert _count_tma(lowered["main"]).total == 1


def _prefetch_source():
    return _SELECTOR_SOURCE.replace(
        "        selected: T.let",
        "        T.ptx.prefetch.tensormap(T.address_of(A_map))\n        selected: T.let",
    )


def test_prefetch_only_main_selector_descriptor():
    lowered = _lower_module(_from_source(_prefetch_source()))
    collector = _PrefetchCollector()
    collector.visit_stmt(lowered["main"].body)
    assert collector.names == ["A_map"]


def test_sparse_decode_absent_extra_map_has_no_dummy_descriptor_or_select():
    source = _prefetch_source()
    source = "\n".join(
        line
        for line in source.splitlines()
        if not line.strip().startswith(("B_map:", "T.cuda.tensormap_encode_tiled(B_map"))
    )
    source = source.replace(
        "T.Select(flag!=0,T.address_of(B_map),T.address_of(A_map))", "T.address_of(A_map)"
    )
    lowered = _lower_module(_from_source(source))
    assert len(_collect_encodes([lowered["main"].body])) == 1
    selects = _SelectCollector()
    selects.visit_stmt(lowered["main"].body)
    assert selects.nodes == []
    prefetches = _PrefetchCollector()
    prefetches.visit_stmt(lowered["main"].body)
    assert prefetches.names == ["A_map"]
    _compile_module(_from_source(source))


@pytest.mark.gpu
@pytest.mark.parametrize("subview", [False, True])
def test_explicit_gather_selector_gpu_roundtrip(subview):
    # False retains the original GPU regression's tensors and row indices;
    # True also executes the old codegen-only B.sub[16:512,8:72] case.
    source = _SELECTOR_SOURCE.replace('"bfloat16"', '"float16"')
    if not subview:
        source = source.replace("(512,80)", "(256,64)").replace("B.sub[16:512,8:72]", "B")
        source = source.replace("64,496,160", "64,256,128").replace(
            "gather4=(1,2,3,4)", "gather4=(7,3,19,5)"
        )
    f = _compile_module(_from_source(source))
    rng = np.random.default_rng(0)
    a_np = rng.standard_normal((256, 64)).astype("float16")
    b_np = rng.standard_normal((512, 80) if subview else (256, 64)).astype("float16")
    a, b = (tvm_tirx.runtime.tensor(x, tvm_tirx.cuda()) for x in (a_np, b_np))
    out = tvm_tirx.runtime.tensor(np.zeros((4, 64), "float16"), tvm_tirx.cuda())
    indices = [1, 2, 3, 4] if subview else [7, 3, 19, 5]
    for flag in (0, 1, -1):
        f(a, b, out, flag)
        expected = a_np if flag == 0 else b_np[16:512, 8:72] if subview else b_np
        np.testing.assert_array_equal(out.numpy(), expected[indices])


def test_caller_dynamic_stage_coordinates_and_pointer():
    kv_row, head, stage = (Var(name, "int32") for name in ("kv_row", "head", "stage"))
    sctx = _context(
        ranges={
            kv_row: Range.from_min_extent(0, 3969),
            head: Range.from_min_extent(0, 16),
            stage: Range.from_min_extent(0, 3),
        }
    )
    gv = tvm_tirx.tirx.decl_tensor(
        (32, 4096, 64), "float16", "G", layout=_plain_layout((32, 4096, 64), (64, 2048, 1))
    )
    sv = tvm_tirx.tirx.decl_tensor(
        (3, 2, 128, 64), "float16", "S", scope="shared", layout="default", data_alignment=128
    )
    impl, host, _ = _emit(
        _load(3),
        gv,
        _region(sv, ((stage, 1), (0, 2), (0, 128), (0, 64))),
        box=(2, 128, 64),
        coords=(head * 2, kv_row, 0),
        coord_steps=((0, 0, 0),),
        sctx=sctx,
    )
    assert _count_tma(impl).total == 1
    sig = _encode_signature(_collect_encodes(host)[0])
    assert _ints(sig["dims"]) == (64, 4096, 32)
    assert _ints(sig["strides"]) == (4096, 128)
    assert _ints(sig["boxes"]) == (64, 128, 2)
    _, args = _ptx_call_parts(_count_tma(impl).calls[0])
    analyzer = Analyzer()
    assert analyzer.can_prove_equal(args["coords"][1], kv_row)
    assert analyzer.can_prove_equal(args["coords"][2], head * 2)
    access = _unwrap_shared_addr(args["dst_mem"][0]).args[0]
    address = access.source.ty.layout.apply(*access.indices, shape=access.source.ty.shape)["m"]
    assert analyzer.can_prove_equal(address, stage * 16384)
    # Keep the original tensor and stage layout in the equivalence check.
    for row_value, head_value, stage_value in ((0, 0, 0), (7, 3, 1), (3968, 15, 2)):
        _recipe_views(
            dict(
                g_shape=(1, 4096, 16, 128),
                s_shape=(3, 128, 128),
                g_region=((0, 1), (row_value, 128), (head_value, 1), (0, 128)),
                s_region=((stage_value, 1), (0, 128), (0, 128)),
                s_layout=TileLayout(S[(3, 128, 2, 64) : (16384, 64, 8192, 1)]),
            ),
            _tma_golden(
                "float16",
                (64, 4096, 32),
                (4096, 128),
                (64, 128, 2),
                (0, row_value, head_value * 2),
                (0, 0, 0, 0),
            ),
        )


def test_dispatch_propagates_flat_bind_to_explicit_coordinates():
    @T.function
    def kernel(D: T.Tensor((33360, 6144), "bfloat16")):
        Dv = D.view(33360, 96, 64).permute(1, 0, 2)
        T.device_entry(launch=T.cuda.LaunchConfig(grid=192, block=1))
        block = T.cuda.block_idx("x")
        tile_index = T.alloc_local((1,), "int32")
        shared = T.alloc_shared(
            (2, 16, 128),
            "bfloat16",
            layout=T.TileLayout(T.S[(2, 16, 2, 64) : (2048, 64, 1024, 1)]),
            align=128,
        )
        sv = shared.view(2, 16, 2, 64).permute(0, 2, 1, 3).sub[0, :, :, :]
        tile_index[0] = block
        d_n: T.let = tile_index[0] * 128
        T.cuda.tile.apply(
            T.ptx["cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"],
            Dv,
            sv,
            box=(2, 16, 64),
            coords=(d_n // 64, 0, 0),
        )

    lowered = _lower_module(kernel)
    sig = _encode_signature(_collect_encodes([lowered["main"].body])[0])
    assert _ints(sig["dims"]) == (64, 33360, 96)
    assert _ints(sig["strides"]) == (12288, 128)
    assert _ints(sig["boxes"]) == (64, 16, 2)
    counter = _count_tma(lowered["main"])
    assert counter.total == 1
    _, args = _ptx_call_parts(counter.calls[0])
    assert _ints(args["coords"][:2]) == (0, 0)
    # The explicit coordinate keeps the ordinary let binding; it need not
    # inline a runtime load during dispatch.
    assert str(args["coords"][2]) == "d_n // 64"
    bindings = []
    tvm_ffi.structural_walk(lowered["main"].body, (tvm_tirx.ir.Bind, lambda bind: bindings.append(bind)))
    assert any(
        bind.var.name == "d_n" and str(bind.value) == "tile_index[0] * 128" for bind in bindings
    )


def _build_sparse_decode_qo_tma_regression():
    q_layout = mma_shared_layout("bfloat16", 3, (64, 512))
    tail_layout = mma_shared_layout("bfloat16", 2, (64, 64))
    q_elements, tail_elements = 64 * 512, 64 * 64
    shared_bytes = (2 * q_elements + tail_elements) * 2

    @T.function
    def kernel(
        Q_storage: T.Tensor((64 * 576,), "bfloat16"),
        O_storage: T.Tensor((64 * 512,), "bfloat16"),
        q_stride_b: T.int64,
        q_stride_s: T.int64,
        q_stride_h: T.int64,
        o_stride_b: T.int64,
        o_stride_s: T.int64,
        o_stride_h: T.int64,
    ):
        Q = Q_storage.view(
            1,
            1,
            64,
            576,
            layout=T.TileLayout(T.S[(1, 1, 64, 576) : (q_stride_b, q_stride_s, q_stride_h, 1)]),
        )
        O_view = O_storage.view(
            1,
            1,
            64,
            512,
            layout=T.TileLayout(T.S[(1, 1, 64, 512) : (o_stride_b, o_stride_s, o_stride_h, 1)]),
        )
        Qv = Q.sub[0, 0, :, :]
        Ov = O_view.sub[0, 0, :, :]
        Qt = Q.sub[0, 0, :, 512:576].view(64, 2, 32).permute(1, 0, 2)
        T.device_entry(
            launch=T.cuda.LaunchConfig(grid=1, block=128, dynamic_smem_bytes=shared_bytes + 8)
        )
        tid = T.cuda.thread_idx("x")
        dyn = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (shared_bytes + 8,), "uint8", scope="shared.dyn", layout="default", align=1024
                )
            ]
        )
        qs = T.decl_tensor(
            dyn.data,
            ty_args=[
                T.Tensor((64, 512), "bfloat16", scope="shared.dyn", layout=q_layout, align=1024)
            ],
        )
        ts = T.decl_tensor(
            dyn.data,
            ty_args=[
                T.Tensor(
                    (64, 64),
                    "bfloat16",
                    scope="shared.dyn",
                    layout=tail_layout,
                    elem_offset=q_elements,
                    align=1024,
                )
            ],
        )
        os = T.decl_tensor(
            dyn.data,
            ty_args=[
                T.Tensor(
                    (64, 512),
                    "bfloat16",
                    scope="shared.dyn",
                    layout=q_layout,
                    elem_offset=q_elements + tail_elements,
                    align=1024,
                )
            ],
        )
        mbar = T.decl_tensor(
            dyn.data,
            ty_args=[
                T.Tensor(
                    (1,),
                    "uint64",
                    scope="shared.dyn",
                    layout="default",
                    elem_offset=shared_bytes // 8,
                )
            ],
        )
        tv = ts.view(64, 2, 32).permute(1, 0, 2)
        policy = T.alloc_local((1,), "uint64", layout="default")
        if tid == 0:
            T.ptx.createpolicy.fractional.L2__evict_first.b64(policy[0])
            for tile in T.unroll(8):
                T.cuda.tile.apply(
                    T.ptx[
                        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint.cta_group::1"
                    ],
                    qs[:, tile * 64 : (tile + 1) * 64],
                    Qv,
                    box=(64, 64),
                    coords=(0, tile * 64),
                    swizzle=3,
                    l2_promotion=2,
                    mbar=mbar.ptr_to([0]),
                    cache_policy=policy[0],
                )
            T.cuda.tile.apply(
                T.ptx[
                    "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint.cta_group::1"
                ],
                tv,
                Qt,
                box=(2, 64, 32),
                coords=(0, 0, 0),
                swizzle=2,
                l2_promotion=2,
                mbar=mbar.ptr_to([0]),
                cache_policy=policy[0],
            )
            for tile in T.unroll(8):
                store_tile: T.let = tile
                T.cuda.tile.apply(
                    T.ptx[
                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                    ],
                    Ov,
                    os[:, store_tile * 64 : (store_tile + 1) * 64],
                    box=(64, 64),
                    coords=(0, store_tile * 64),
                    swizzle=3,
                    l2_promotion=2,
                    cache_policy=policy[0],
                )

    return kernel


def test_sparse_decode_runtime_stride_q_tail_o_counts_and_descriptor_dedup():
    kernel = _build_sparse_decode_qo_tma_regression()
    lowered = _lower_module(kernel)
    counter = _count_tma(lowered["main"])
    assert (
        sum(
            weight
            for call, weight in counter.weighted_calls
            if call.op.name == "tirx.ptx.cp_async_bulk_tensor_g2s_cluster"
        )
        == 9
    )
    assert (
        sum(
            weight
            for call, weight in counter.weighted_calls
            if call.op.name == "tirx.ptx.cp_async_bulk_tensor_s2g"
        )
        == 8
    )
    signatures = [_encode_signature(c) for c in _collect_encodes([lowered["main"].body])]
    assert len(signatures) == 3
    assert sorted(int(s["enums"][1]) for s in signatures) == [2, 3, 3]
    assert all(any(not isinstance(x, IntImm) for x in s["strides"]) for s in signatures)
    _compile_module(kernel)


def test_explicit_gather_uses_extracted_swizzled_slice_pointer():
    layout = mma_shared_layout("float16", 3, (8, 64))
    impl, _, _ = _direct(
        _load(2, "tile::gather4"),
        g_shape=(8, 64),
        s_shape=(8, 64),
        s_region=((4, 4), (0, 64)),
        box=(1, 64),
        s_layout=layout,
        swizzle=3,
        gather4=(0, 1, 2, 3),
    )
    _, args = _ptx_call_parts(_count_tma(impl).calls[0])
    pointer = args["dst_mem"][0]
    calls = []
    tvm_ffi.structural_walk(pointer, (tvm_tirx.ir.Call, lambda c: calls.append(c)))
    physical = next(c for c in calls if c.op.name == "tirx.cuda.cvta_generic_to_shared")
    # Row 4 has physical byte offset 576; the instruction must start at 512.
    concrete = tvm_ffi.structural_map(
        pointer, (tvm_tirx.ir.Call, lambda c: T.uint32(576) if c.same_as(physical) else c), order="post"
    )
    assert int(Analyzer().simplify(concrete)) == 512


@pytest.mark.gpu
@pytest.mark.parametrize("swizzle", [1, 2, 3])
@pytest.mark.parametrize("start", [0, 4])
def test_gather_swizzled_subregion_gpu(swizzle, start):
    cols = 8 << swizzle
    layout = mma_shared_layout("float16", swizzle, (8, cols))

    @T.function
    def kernel(A: T.Tensor((8, cols), "float16"), Out: T.Tensor((8, cols), "float16")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
        lane = T.cuda.lane_id()
        shared = T.alloc_shared((8, cols), "float16", layout=layout, align=1024)
        mbar = T.alloc_shared((1,), "uint64", layout="default")
        if lane == 0:
            for r, c in T.grid(8, cols):
                shared[r, c] = T.float16(-9)
            T.ptx.mbarrier.init.shared.b64(mbar.ptr_to([0]), T.uint32(1))
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()
        if lane == 0:
            T.cuda.tile.apply(
                T.ptx[
                    "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes"
                ],
                shared[start : start + 4, :],
                A,
                box=(1, cols),
                coords=(0, 0),
                gather4=(7, 3, 1, 5),
                swizzle=swizzle,
                mbar=mbar.ptr_to([0]),
            )
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(mbar.ptr_to([0]), T.uint32(4 * cols * 2))
        T.cuda.mbarrier_wait(mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()
        if lane == 0:
            for r, c in T.grid(8, cols):
                Out[r, c] = shared[r, c]

    f = _compile_module(kernel)
    x = np.arange(8 * cols, dtype="float16").reshape(8, cols)
    out = tvm_tirx.runtime.tensor(np.zeros_like(x), tvm_tirx.cuda())
    f(tvm_tirx.runtime.tensor(x, tvm_tirx.cuda()), out)
    expected = np.full_like(x, -9)
    expected[start : start + 4] = x[[7, 3, 1, 5]]
    np.testing.assert_array_equal(out.numpy(), expected)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "case_id",
    ["g2s-2d-32x512-atom", "g2s-multiphase-7x32x512-atom", "s2g-multiphase-7x32x512-atom"],
)
def test_caller_tiles_nested_shared_layout(case_id):
    kwargs = next(p.values[1] for p in TMA_CASES if p.values[0] == case_id)
    shape = kwargs["g_shape"]
    rank = len(shape)
    is_load = kwargs.get("direction", "g2s") == "g2s"
    opcode = _load(rank) if is_load else _store(rank)
    box = (8, 64) if rank == 2 else (1, 8, 64)
    steps = (
        ((16, 0), (0, 256), (8, 0), (0, 64))
        if rank == 2
        else ((0, 16, 0), (0, 0, 256), (0, 8, 0), (0, 0, 64))
    )
    view_shape = (2, 2, 2, 4, *box)
    layout = kwargs["s_layout"]

    @T.function
    def kernel(G: T.Tensor(shape, "float16"), A: T.Tensor((32, 512), "float16")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=128))
        tid = T.cuda.thread_idx("x")
        shared = T.alloc_shared((32, 512), "float16", layout=layout, align=1024)
        sv = shared.rearrange("(a b r) (c d v) -> a c b d r v", a=2, b=2, r=8, c=2, d=4, v=64).view(
            *view_shape
        )
        mbar = T.alloc_shared((1,), "uint64", layout="default")
        if T.constexpr(is_load):
            if tid == 0:
                T.ptx.mbarrier.init.shared.b64(mbar.ptr_to([0]), T.uint32(1))
        else:
            for step in T.serial(128):
                flat = step * 128 + tid
                shared[flat // 512, flat % 512] = A[flat // 512, flat % 512]
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()
        if tid == 0:
            if T.constexpr(is_load):
                T.cuda.tile.apply(
                    T.ptx[opcode],
                    sv,
                    G,
                    box=box,
                    coords=(0,) * rank,
                    coord_steps=steps,
                    swizzle=3,
                    mbar=mbar.ptr_to([0]),
                )
                T.ptx.mbarrier.arrive.expect_tx.shared.b64(mbar.ptr_to([0]), T.uint32(32768))
            else:
                T.cuda.tile.apply(
                    T.ptx[opcode], G, sv, box=box, coords=(0,) * rank, coord_steps=steps, swizzle=3
                )
                T.ptx.cp.async_.bulk.commit_group()
                T.ptx.cp.async_.bulk.wait_group(0)
        if T.constexpr(is_load):
            T.cuda.mbarrier_wait(mbar.ptr_to([0]), 0)
            T.cuda.cta_sync()
            for step in T.serial(128):
                flat = step * 128 + tid
                A[flat // 512, flat % 512] = shared[flat // 512, flat % 512]

    assert _count_tma(_lower_module(kernel)["main"]).total == 32
    f = _compile_module(kernel)
    rng = np.random.default_rng(17)
    values = rng.standard_normal(shape if is_load else (32, 512)).astype("float16")
    g = tvm_tirx.runtime.tensor(values if is_load else np.full(shape, -9, "float16"), tvm_tirx.cuda())
    a = tvm_tirx.runtime.tensor(np.zeros((32, 512), "float16") if is_load else values, tvm_tirx.cuda())
    f(g, a)
    if is_load:
        np.testing.assert_array_equal(a.numpy(), values if rank == 2 else values[0])
    else:
        expected = np.full(shape, -9, "float16")
        expected[0] = values
        np.testing.assert_array_equal(g.numpy(), expected)


def test_caller_explicit_strided_outer_issues():
    kwargs = next(p.values[1] for p in TMA_CASES if p.values[0] == "g2s-3d-stride-gap-outer")
    g = tvm_tirx.tirx.decl_tensor(kwargs["g_shape"], "float16", "A", layout=kwargs["g_layout"])
    s = tvm_tirx.tirx.decl_tensor(
        (8, 1, 32, 64),
        "float16",
        "S",
        scope="shared",
        layout=_plain_layout((8, 1, 32, 64), (4096, 0, 64, 1)),
        data_alignment=128,
    )
    impl, _, _ = _emit(_load(3), g, s, box=(1, 32, 64), coords=(0, 0, 0), coord_steps=((1, 0, 0),))
    assert _count_tma(impl).total == 8
    _, old_s = _original_addresses(kwargs)
    new_s = _address_array(s.ty.layout, s.ty.shape, ((0, 8), (0, 1), (0, 32), (0, 64)))
    np.testing.assert_array_equal(old_s, new_s)


@pytest.mark.parametrize(
    "case_id",
    [
        "g2s-transpose-32x64",
        "g2s-transpose-64x32",
        "g2s-transpose-partial-region",
        "g2s-transpose-partial-offset",
    ],
)
def test_transpose_is_not_a_tma_box(case_id):
    kwargs = next(p.values[1] for p in TMA_CASES if p.values[0] == case_id)
    with pytest.raises(RuntimeError, match="compact/swizzled box"):
        _direct(
            _load(2),
            g_shape=kwargs["g_shape"],
            s_shape=kwargs["s_shape"],
            g_layout=kwargs["g_layout"],
            s_layout=kwargs["s_layout"],
            coords=tuple(x for x, _ in kwargs["g_region"]),
        )


def test_descriptor_cache_distinguishes_explicit_dtype_views():
    data = Var("A", PtrType(PrimType("uint8"), "global"))
    sctx = _context()
    for dtype in ("uint8", "uint16"):
        g = tvm_tirx.tirx.decl_tensor((256,), dtype, "A_view", data=data, layout="default")
        s = tvm_tirx.tirx.decl_tensor(
            (256,), dtype, "S", scope="shared", layout="default", data_alignment=128
        )
        _emit(_load(1), g, s, box=(256,), coords=(0,), sctx=sctx)
        _emit(_load(1), g, s, box=(256,), coords=(0,), sctx=sctx)
    sigs = [_encode_signature(c) for c in _collect_encodes(sctx.callbacks["host_init_stmt"])]
    assert [s["dtype"] for s in sigs] == ["uint8", "uint16"]
    assert [_ints(s["dims"]) for s in sigs] == [(256,), (256,)]


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"g_shape": (0, 16)}, "dimensions"),
        ({"g_shape": (2**32 + 1, 16)}, "dimensions"),
        ({"g_layout": _plain_layout((4, 16), (-8, 1))}, "byte strides"),
        ({"g_layout": _plain_layout((4, 16), (4, 1))}, "byte strides"),
        ({"g_layout": _plain_layout((4, 16), (2**39, 1))}, "byte strides"),
        ({"g_elem_offset": 1}, "alignment"),
        ({"box": (4, 0)}, "box"),
        ({"box": (4, 257)}, "box"),
        ({"box": (4, 7)}, "box"),
        ({"element_strides": (1, 0)}, "element_strides"),
        ({"element_strides": (1, 9)}, "element_strides"),
        ({"element_strides": (1, 2)}, "element stride"),
        ({"g_layout": _plain_layout((4, 16), (16, 2))}, "innermost global stride"),
        ({"interleave": 3}, "interleave"),
        ({"swizzle": 9}, "swizzle"),
        ({"l2_promotion": 4}, "l2_promotion"),
        ({"oob_fill": 2}, "oob_fill"),
        ({"coords": (0,)}, "coords"),
        ({"format": "uint8"}, "storage element width"),
        ({"swizzle": 1, "box": (4, 24)}, "swizzle span"),
        ({"dtype": "int32", "oob_fill": 1}, "floating-point"),
    ],
)
def test_descriptor_static_boundary_matrix(kwargs, error):
    params = dict(g_shape=(4, 16), s_shape=(4, 16), box=(4, 16))
    params.update(kwargs)
    with pytest.raises((RuntimeError, ValueError), match=error):
        _direct(_load(2), **params)


def test_unknown_descriptor_constraints_are_checked_by_runtime_encode():
    dim, stride = Var("runtime_global_dim", "int64"), Var("runtime_stride", "int64")
    _, host, _ = _direct(
        _load(2), g_shape=(dim, 16), s_shape=(4, 16), g_layout=_plain_layout((dim, 16), (stride, 1))
    )
    sig = _encode_signature(_collect_encodes(host)[0])
    tvm_tirx.ir.assert_structural_equal(sig["dims"][1], dim)
    assert Analyzer().can_prove_equal(sig["strides"][0], stride * 2)


@pytest.mark.parametrize("fill", [0, 1])
def test_store_descriptor_oob_field_is_explicit(fill):
    # The old convenience API rejected any oob keyword on stores. CUDA permits
    # a floating descriptor with either fill field; stores never synthesize data.
    case_id = "s2g-oob-zero" if fill == 0 else "s2g-oob-nan"
    kwargs = next(p.values[1] for p in TMA_CASES if p.values[0] == case_id)
    recipe = _TMA_CASE_GOLDENS["s2g-oob-none"]
    g, s = _recipe_views(kwargs, recipe)
    impl, host, _ = _emit(_store(2), g, s, box=(128, 64), coords=(0, 0), swizzle=3, oob_fill=fill)
    assert _count_tma(impl).total == 1
    assert int(_encode_signature(_collect_encodes(host)[0])["enums"][-1]) == fill


@pytest.mark.parametrize(
    "case_id", ["reject-unknown-oob", "reject-g2s-nan-on-non-float", "reject-s2g-nan-on-non-float"]
)
def test_descriptor_fill_rejects_invalid_format(case_id):
    kwargs = next(p.values[1] for p in TMA_CASES if p.values[0] == case_id)
    rank = len(kwargs["g_shape"])
    box = (1, *kwargs["s_shape"]) if rank == 3 else kwargs["s_shape"]
    opcode = _store(rank) if kwargs.get("direction") == "s2g" else _load(rank)
    fill = "bogus" if case_id == "reject-unknown-oob" else 1
    with pytest.raises(RuntimeError, match="invalid literal|floating-point"):
        _direct(
            opcode,
            g_shape=kwargs["g_shape"],
            s_shape=kwargs["s_shape"],
            dtype=kwargs.get("dtype", "float16"),
            box=box,
            oob_fill=fill,
        )


def test_tma_case_matrix_is_complete():
    explicit_cases = {
        "g2s-2d-32x512-atom",
        "g2s-multiphase-7x32x512-atom",
        "s2g-multiphase-7x32x512-atom",
        "g2s-3d-stride-gap-outer",
        "g2s-transpose-32x64",
        "g2s-transpose-64x32",
        "g2s-transpose-partial-region",
        "g2s-transpose-partial-offset",
        "s2g-oob-zero",
        "s2g-oob-nan",
        "reject-unknown-oob",
        "reject-g2s-nan-on-non-float",
        "reject-s2g-nan-on-non-float",
    }
    ids = [p.values[0] for p in TMA_CASES]
    assert len(ids) == len(set(ids)) == 52
    assert set(ids) == set(_TMA_CASE_GOLDENS) | explicit_cases


@pytest.mark.gpu
@pytest.mark.parametrize("warp_offset", [0, 1, 3])
def test_caller_stages_unaligned_shared_region(warp_offset):
    # Original offset-count regressions wrote at 32 + warp_offset*64 + 2*64
    # halfwords, which is 64 bytes off the native TMA alignment. Preserve the
    # destination through explicit aligned staging, then caller scalar stores.
    layout = TileLayout(S[(4, 64)] + warp_offset * 64)
    with pytest.raises(RuntimeError, match="128-byte aligned"):
        _direct(
            _load(2),
            g_shape=(2, 64),
            s_shape=(4, 64),
            s_region=((2, 2), (0, 64)),
            s_layout=layout,
            s_elem_offset=32,
        )

    @T.function
    def kernel(A: T.Tensor((2, 64), "float16"), Out: T.Tensor((512,), "float16")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
        lane = T.cuda.lane_id()
        backing = T.alloc_shared((512,), "float16", layout="default", align=128)
        target = T.decl_tensor(
            backing.data,
            ty_args=[
                T.Tensor(
                    (4, 64), "float16", scope="shared", layout=layout, elem_offset=32, align=128
                )
            ],
        )
        stage = T.alloc_shared((2, 64), "float16", layout="default", align=128)
        mbar = T.alloc_shared((1,), "uint64", layout="default")
        if lane == 0:
            for i in T.serial(512):
                backing[i] = T.float16(-9)
            T.ptx.mbarrier.init.shared.b64(mbar.ptr_to([0]), T.uint32(1))
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()
        if lane == 0:
            T.cuda.tile.apply(
                T.ptx[
                    "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                ],
                stage,
                A,
                box=(2, 64),
                coords=(0, 0),
                mbar=mbar.ptr_to([0]),
            )
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(mbar.ptr_to([0]), T.uint32(256))
        T.cuda.mbarrier_wait(mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()
        if lane == 0:
            for r, c in T.grid(2, 64):
                target[r + 2, c] = stage[r, c]
            for i in T.serial(512):
                Out[i] = backing[i]

    f = _compile_module(kernel)
    x = np.arange(128, dtype="float16").reshape(2, 64)
    out = tvm_tirx.runtime.tensor(np.zeros(512, "float16"), tvm_tirx.cuda())
    f(tvm_tirx.runtime.tensor(x, tvm_tirx.cuda()), out)
    expected = np.full(512, -9, "float16")
    start = 32 + warp_offset * 64 + 128
    expected[start : start + 128] = x.reshape(-1)
    np.testing.assert_array_equal(out.numpy(), expected)


@pytest.mark.gpu
def test_caller_descriptor_selection_uses_first_true():
    source = _SELECTOR_SOURCE.replace('"bfloat16"', '"float16"')
    source = source.replace(
        "T.Select(flag!=0,T.address_of(B_map),T.address_of(A_map))",
        "T.Select(flag>0,T.address_of(A_map),T.Select(flag<2,T.address_of(B_map),T.address_of(A_map)))",
    )
    f = _compile_module(_from_source(source))
    rng = np.random.default_rng(23)
    a_np = rng.standard_normal((256, 64)).astype("float16")
    b_np = rng.standard_normal((512, 80)).astype("float16")
    a, b = (tvm_tirx.runtime.tensor(x, tvm_tirx.cuda()) for x in (a_np, b_np))
    out = tvm_tirx.runtime.tensor(np.zeros((4, 64), "float16"), tvm_tirx.cuda())
    for flag in (0, 1, 2, -1):
        f(a, b, out, flag)
        expected = a_np if flag > 0 else b_np[16:512, 8:72]
        np.testing.assert_array_equal(out.numpy(), expected[[1, 2, 3, 4]])


@pytest.mark.gpu
@pytest.mark.parametrize("swizzle", [0, 3])
@pytest.mark.parametrize("gather", [False, True])
def test_packed_fp4_payload_gpu(swizzle, gather):
    # 16U4_ALIGN16B: each 16 nibbles occupy the low eight bytes of a
    # 16-byte shared block. Undefined padding is deliberately not inspected.
    layout = ComposeLayout(5, swizzle, 3, TileLayout(S[(4, 8, 16) : (256, 32, 1)]))
    mode = "tile::gather4" if gather else "tile"
    box = (1, 128) if gather else (4, 128)
    fields = {"gather4": (7, 3, 1, 5)} if gather else {}
    opcode = f"cp.async.bulk.tensor.2d.shared::cta.global.{mode}.mbarrier::complete_tx::bytes"

    @T.function
    def kernel(A: T.Tensor((8, 64), "uint8", layout="default"), Out: T.Tensor((4, 64), "uint8")):
        Av = A.view("float4_e2m1fn")
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
        lane = T.cuda.lane_id()
        raw = T.alloc_shared((4, 128), "uint8", layout="default", align=1024)
        packed = T.decl_tensor(
            raw.data,
            ty_args=[
                T.Tensor((4, 128), "float4_e2m1fn", scope="shared", layout=layout, align=1024)
            ],
        )
        payload = packed.view("uint8")
        mbar = T.alloc_shared((1,), "uint64", layout="default")
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(mbar.ptr_to([0]), T.uint32(1))
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()
        if lane == 0:
            T.cuda.tile.apply(
                T.ptx[opcode],
                packed,
                Av,
                box=box,
                coords=(0, 0),
                swizzle=swizzle,
                mbar=mbar.ptr_to([0]),
                **fields,
            )
            # The transfer contains 256 payload bytes and occupies 512 bytes
            # in shared memory. Completion counts the former.
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(mbar.ptr_to([0]), T.uint32(256))
        T.cuda.mbarrier_wait(mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()
        for row, half in T.grid(4, 2):
            Out[row, lane + 32 * half] = payload[row, lane + 32 * half]

    f = _compile_module(kernel)
    x = np.random.default_rng(5).integers(0, 256, (8, 64), dtype="uint8")
    out = tvm_tirx.runtime.tensor(np.zeros((4, 64), "uint8"), tvm_tirx.cuda())
    f(tvm_tirx.runtime.tensor(x, tvm_tirx.cuda()), out)
    np.testing.assert_array_equal(out.numpy(), x[[7, 3, 1, 5]] if gather else x[:4])


@pytest.mark.gpu
@pytest.mark.parametrize("rank", [3, 4, 5])
@pytest.mark.parametrize("interleave", [1, 2])
@pytest.mark.parametrize("channel_step,width_step", [(1, 1), (2, 1), (1, 2), (2, 2)])
def test_interleaved_tma_roundtrip_gpu(rank, interleave, channel_step, width_step):
    # Global and shared axes: N,Cslices,...,W,Ci. The last axis is implicit
    # in the descriptor. PTX C coordinates count whole slices; one is copied.
    channels = 8 * interleave
    spatial = (4,) * (rank - 3) + (12,)
    physical = (3, 6, *spatial, channels)
    box = (2, 2 * channels + 1, *((2,) * (rank - 3)), 5)
    steps = (1, channel_step, *((1,) * (rank - 3)), width_step)
    coords = (1, 1, *((1,) * (rank - 2)))
    s_shape = (2, 1, *((2,) * (rank - 3)), (5 + width_step - 1) // width_step, channels)
    size = math.prod(physical)
    count = math.prod(s_shape)
    swizzle = 1 if interleave == 2 else 0
    s_layout = ComposeLayout(3, swizzle, 3, _plain_layout(s_shape))
    load = f"cp.async.bulk.tensor.{rank}d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
    store = f"cp.async.bulk.tensor.{rank}d.global.shared::cta.tile.bulk_group"

    @T.function
    def kernel(
        A: T.Tensor((size,), "uint16"),
        Out: T.Tensor((size,), "uint16"),
        Got: T.Tensor((count,), "uint16"),
    ):
        av = A.view(*physical)
        ov = Out.view(*physical)
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
        lane = T.cuda.lane_id()
        shared = T.alloc_shared(s_shape, "uint16", layout=s_layout, align=1024)
        flat = shared.view(count)
        mbar = T.alloc_shared((1,), "uint64", layout="default")
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(mbar.ptr_to([0]), T.uint32(1))
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()
        if lane == 0:
            T.cuda.tile.apply(
                T.ptx[load],
                shared,
                av,
                box=box,
                coords=coords,
                element_strides=steps,
                interleave=interleave,
                swizzle=swizzle,
                mbar=mbar.ptr_to([0]),
            )
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(mbar.ptr_to([0]), T.uint32(count * 2))
        T.cuda.mbarrier_wait(mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()
        if lane == 0:
            for i in T.serial(count):
                Got[i] = flat[i]
            T.cuda.tile.apply(
                T.ptx[store],
                ov,
                shared,
                box=box,
                coords=coords,
                element_strides=steps,
                interleave=interleave,
                swizzle=swizzle,
            )
            T.ptx.cp.async_.bulk.commit_group()
            T.ptx.cp.async_.bulk.wait_group(0)

    f = _compile_module(kernel)
    x = np.arange(size, dtype="uint16").reshape(physical)
    slices = (
        slice(1, 3),
        slice(1, 2),
        *((slice(1, 3),) * (rank - 3)),
        slice(1, 6, width_step),
        slice(None),
    )
    expected_tile = x[slices]
    assert expected_tile.shape == s_shape
    expected_out = np.full(physical, 65535, "uint16")
    expected_out[slices] = expected_tile
    out = tvm_tirx.runtime.tensor(np.full(size, 65535, "uint16"), tvm_tirx.cuda())
    got = tvm_tirx.runtime.tensor(np.zeros(count, "uint16"), tvm_tirx.cuda())
    f(tvm_tirx.runtime.tensor(x.reshape(-1), tvm_tirx.cuda()), out, got)
    np.testing.assert_array_equal(got.numpy(), expected_tile.reshape(-1))
    np.testing.assert_array_equal(out.numpy(), expected_out.reshape(-1))


def test_original_interleave_descriptor_contract():
    # Preserve the old validator's exact native fields, now with the Ci axis
    # and full shared payload that real interleaved hardware accesses require.
    _, host, _ = _direct(
        _load(3),
        g_shape=(4, 8, 16, 16),
        s_shape=(2, 1, 8, 16),
        g_layout=_plain_layout((4, 8, 16, 16), (128, 16, 16, 1)),
        s_layout=ComposeLayout(3, 1, 3, _plain_layout((2, 1, 8, 16))),
        box=(2, 4, 8),
        interleave=2,
        swizzle=1,
    )
    sig = _encode_signature(_collect_encodes(host)[0])
    assert _ints(sig["dims"]) == (16, 8, 4)
    assert _ints(sig["strides"]) == (32, 256)
    assert _ints(sig["boxes"]) == (8, 4, 2)
    with pytest.raises(RuntimeError, match="interleave=2 requires swizzle=1"):
        _direct(_load(3), g_shape=(4, 8, 16, 16), box=(2, 4, 8), interleave=2, swizzle=2)


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"g_shape": (64, 64)}, "inner dimension"),
        ({"box": (4, 64)}, "inner box 128"),
        ({"swizzle": 2}, "swizzle 0 or 3"),
        ({"opcode": _store(2)}, "loads only"),
    ],
)
def test_packed_fp4_rejects_invalid_native_contract(kwargs, error):
    fields = dict(
        opcode=_load(2),
        g_shape=(64, 128),
        s_shape=(4, 128),
        box=(4, 128),
        dtype="float4_e2m1fn",
        swizzle=3,
    )
    fields.update(kwargs)
    with pytest.raises(RuntimeError, match=error):
        _direct(**fields)


def test_odd_partial_uint8_box_requires_caller_staging():
    # Original region (..., :5) is not a legal 16-byte inner TMA box. No
    # automatic uint16 reinterpretation may change which bytes are copied.
    with pytest.raises(RuntimeError, match="multiple of 16 bytes"):
        _direct(_load(3), g_shape=(8, 64, 8), s_shape=(8, 64, 5), dtype="uint8")


@pytest.mark.gpu
def test_caller_splits_oversized_box():
    @T.function
    def kernel(A: T.Tensor((4096,), "uint8"), Out: T.Tensor((4096,), "uint8")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
        lane = T.cuda.lane_id()
        shared = T.alloc_shared((4096,), "uint8", layout="default", align=128)
        tiles = shared.view(16, 256)
        mbar = T.alloc_shared((1,), "uint64", layout="default")
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(mbar.ptr_to([0]), T.uint32(1))
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()
        if lane == 0:
            T.cuda.tile.apply(
                T.ptx[
                    "cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                ],
                tiles,
                A,
                box=(256,),
                coords=(0,),
                coord_steps=((256,),),
                mbar=mbar.ptr_to([0]),
            )
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(mbar.ptr_to([0]), T.uint32(4096))
        T.cuda.mbarrier_wait(mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()
        for i in T.serial(128):
            Out[i * 32 + lane] = shared[i * 32 + lane]

    assert _count_tma(_lower_module(kernel)["main"]).total == 16
    f = _compile_module(kernel)
    x = np.random.default_rng(31).integers(0, 256, 4096, dtype="uint8")
    out = tvm_tirx.runtime.tensor(np.zeros(4096, "uint8"), tvm_tirx.cuda())
    f(tvm_tirx.runtime.tensor(x, tvm_tirx.cuda()), out)
    np.testing.assert_array_equal(out.numpy(), x)


@pytest.mark.gpu
@pytest.mark.parametrize("rows,start", [(128, 4), (130, 0)])
def test_caller_stages_old_incomplete_chain(rows, start):
    shape = (3, 64, 8)
    layout = _plain_layout(shape, (512, 1, 64))
    # Reorganizing just the logical shape does not turn this transpose into
    # a TMA box. Keep the original storage and explicitly scatter a staging box.
    with pytest.raises(RuntimeError, match="compact/swizzled box"):
        _direct(
            _load(2),
            g_shape=(rows, 64),
            s_shape=(24, 64),
            s_layout=layout,
            box=(24, 64),
            coords=(start, 0),
        )

    @T.function
    def kernel(A: T.Tensor((rows, 64), "float16"), Out: T.Tensor((1536,), "float16")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
        lane = T.cuda.lane_id()
        shared = T.alloc_shared(shape, "float16", layout=layout, align=128)
        physical = T.decl_tensor(
            shared.data, ty_args=[T.Tensor((1536,), "float16", scope="shared")]
        )
        stage = T.alloc_shared((24, 64), "float16", layout="default", align=128)
        mbar = T.alloc_shared((1,), "uint64", layout="default")
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(mbar.ptr_to([0]), T.uint32(1))
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()
        if lane == 0:
            T.cuda.tile.apply(
                T.ptx[
                    "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                ],
                stage,
                A,
                box=(24, 64),
                coords=(start, 0),
                mbar=mbar.ptr_to([0]),
            )
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(mbar.ptr_to([0]), T.uint32(3072))
        T.cuda.mbarrier_wait(mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()
        for i in T.serial(48):
            flat = i * 32 + lane
            shared[flat // 512, flat // 8 % 64, flat % 8] = stage[flat // 64, flat % 64]
        T.cuda.cta_sync()
        for i in T.serial(48):
            Out[i * 32 + lane] = physical[i * 32 + lane]

    f = _compile_module(kernel)
    x = np.random.default_rng(37).standard_normal((rows, 64)).astype("float16")
    out = tvm_tirx.runtime.tensor(np.zeros(1536, "float16"), tvm_tirx.cuda())
    f(tvm_tirx.runtime.tensor(x, tvm_tirx.cuda()), out)
    expected = x[start : start + 24].reshape(shape).transpose(0, 2, 1).reshape(-1)
    np.testing.assert_array_equal(out.numpy(), expected)


def test_explicit_gather_source_and_destination_contract():
    fields = dict(g_shape=(256, 96), s_shape=(4, 64), box=(1, 64), gather4=(1, 2, 3, 4))
    _, host, _ = _direct(_load(2, "tile::gather4"), **fields)
    assert _ints(_encode_signature(_collect_encodes(host)[0])["strides"]) == (192,)
    with pytest.raises(RuntimeError, match="innermost global stride"):
        _direct(_load(2, "tile::gather4"), **fields, g_layout=_plain_layout((256, 96), (1, 256)))
    with pytest.raises(RuntimeError, match="compact/swizzled box"):
        _direct(_load(2, "tile::gather4"), **fields, s_layout=_plain_layout((4, 64), (128, 1)))


def test_descriptor_format_is_never_promoted():
    # The former promotion helper had eight guards against changing arithmetic,
    # coordinates, strides or packed storage. There is no promotion helper now:
    # the region dtype is the descriptor dtype, and hardware fields stay literal.
    _, host, _ = _direct(
        _load(2),
        g_shape=(64, 16),
        s_shape=(32, 16),
        box=(64, 16),
        coords=(1, 1),
        dtype="uint8",
        element_strides=(2, 1),
    )
    sig = _encode_signature(_collect_encodes(host)[0])
    assert sig["dtype"] == "uint8"
    assert _ints(sig["dims"]) == (16, 64)
    assert _ints(sig["boxes"]) == (16, 64)
    assert _ints(sig["element_strides"]) == (1, 2)


@pytest.mark.gpu
@pytest.mark.parametrize("interleave,swizzle", [(1, 0), (1, 1), (1, 2), (1, 3), (2, 1)])
def test_interleaved_channel_repeats_gpu(interleave, swizzle):
    channels = interleave * 8
    physical = (2, 5, 16, channels)
    shared_shape = (3, 1, 1, 8, channels)
    count = math.prod(shared_shape)
    layout = ComposeLayout(3, swizzle, 3, _plain_layout(shared_shape))

    @T.function
    def kernel(A: T.Tensor(physical, "uint16"), Out: T.Tensor((count,), "uint16")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
        lane = T.cuda.lane_id()
        shared = T.alloc_shared(shared_shape, "uint16", layout=layout, align=1024)
        flat = shared.view(count)
        mbar = T.alloc_shared((1,), "uint64", layout="default")
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(mbar.ptr_to([0]), T.uint32(1))
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()
        if lane == 0:
            T.cuda.tile.apply(
                T.ptx[
                    "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                ],
                shared,
                A,
                box=(1, 1, 8),
                coords=(1, 1, 2),
                coord_steps=((0, 1, 0),),
                interleave=interleave,
                swizzle=swizzle,
                mbar=mbar.ptr_to([0]),
            )
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(mbar.ptr_to([0]), T.uint32(count * 2))
        T.cuda.mbarrier_wait(mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()
        for i in T.serial(count // 32):
            Out[i * 32 + lane] = flat[i * 32 + lane]

    assert _count_tma(_lower_module(kernel)["main"]).total == 3
    f = _compile_module(kernel)
    if interleave == 1 and swizzle:
        pytest.skip(
            "B200 traps on 16B interleave with nonzero swizzle, including a raw-PTX reproducer; "
            "descriptor encoding and compilation are still checked"
        )
    x = np.arange(math.prod(physical), dtype="uint16").reshape(physical)
    out = tvm_tirx.runtime.tensor(np.zeros(count, "uint16"), tvm_tirx.cuda())
    f(tvm_tirx.runtime.tensor(x, tvm_tirx.cuda()), out)
    np.testing.assert_array_equal(out.numpy(), x[1, 1:4, 2:10, :].reshape(-1))


@pytest.mark.parametrize("arch,gather", [("sm_80", False), ("sm_90a", True)])
def test_native_compiler_rejects_unsupported_tma_target(arch, gather):
    mode = "tile::gather4" if gather else "tile"
    box = (1, 64) if gather else (4, 64)
    fields = {"gather4": (1, 2, 3, 4)} if gather else {}
    opcode = f"cp.async.bulk.tensor.2d.shared::cluster.global.{mode}.mbarrier::complete_tx::bytes"

    @T.function
    def kernel(A: T.Tensor((256, 64), "float16")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
        shared = T.alloc_shared((4, 64), "float16", layout="default", align=128)
        mbar = T.alloc_shared((1,), "uint64", layout="default")
        if T.cuda.lane_id() == 0:
            T.cuda.tile.apply(
                T.ptx[opcode], shared, A, box=box, coords=(0, 0), mbar=mbar.ptr_to([0]), **fields
            )

    # ISA target requirements belong to the PTX compiler, not another tile
    # registry's architecture policy. A wrong target must still fail to build.
    with pytest.raises(Exception, match="(?i)(requires|require|unsupported|not supported|feature)"):
        _compile_module(kernel, arch=arch)
