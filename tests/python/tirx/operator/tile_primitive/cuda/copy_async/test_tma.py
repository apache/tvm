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

"""Tests for the independent ``tma_auto`` and ``tma_explicit`` planners."""

from dataclasses import dataclass, replace
from functools import reduce
from operator import mul

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.arith import Analyzer
from tvm.ir import PointerType, PrimType, Range
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.testing import env
from tvm.tirx import IntImm, StringImm, Var
from tvm.tirx.cuda.tile_primitive.copy_async.tma import (
    AutoIssueAxis,
    IssueCoord,
    ProofStatus,
    TensorMapSpec,
    TMAPlan,
    _build_auto_plan,
    _build_explicit_plan,
    _promote_auto_once,
    _selector_compatibility,
    _validation_failures,
    copy_tma_auto_impl,
    copy_tma_explicit_impl,
    validate_tensor_map_spec,
)
from tvm.tirx.cuda.tile_primitive.tma_utils import (
    mma_atom_layout,
    mma_atom_shape,
    mma_shared_layout,
)
from tvm.tirx.exec_scope import ExecScope
from tvm.tirx.layout import S, TileLayout
from tvm.tirx.operator.tile_primitive.ops import CopyAsync
from tvm.tirx.stmt import BufferRegion
from tvm.tirx.stmt_functor import StmtExprVisitor
from tvm.tirx.tile_primitive import DispatchContext

_TMA_OPS = {
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
    from tvm.backend.cuda.ptx.table import TABLE, mods, operand_layout

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
    if isinstance(value, tvm.ir.Call) and value.op.name == "tirx.cuda.cvta_generic_to_shared":
        return value.args[0]
    return value


class _TMACounter(StmtExprVisitor):
    def __init__(self):
        super().__init__()
        self.loop_extents = []
        self.total = 0
        self.calls = []
        self.weighted_calls = []

    def visit_for_(self, op):
        self.loop_extents.append(op.extent)
        self.visit_stmt(op.body)
        self.loop_extents.pop()

    def visit_evaluate_(self, op):
        if isinstance(op.value, tvm.ir.Call) and op.value.op.name in _TMA_OPS:
            multiplier = reduce(mul, (int(extent) for extent in self.loop_extents), 1)
            self.total += multiplier
            self.calls.append(op.value)
            self.weighted_calls.append((op.value, multiplier))
        super().visit_evaluate_(op)


class _EncodeCollector(StmtExprVisitor):
    def __init__(self):
        super().__init__()
        self.calls = []

    def visit_call_(self, op):
        if (
            isinstance(op.op, tvm.ir.Op)
            and op.op.name == "tirx.tvm_call_packed"
            and isinstance(op.args[0], StringImm)
            and op.args[0].value == "runtime.cuTensorMapEncodeTiled"
        ):
            self.calls.append(op)
        super().visit_call_(op)


class _SelectCollector(StmtExprVisitor):
    def __init__(self):
        super().__init__()
        self.nodes = []

    def visit_select_(self, op):
        self.nodes.append(op)
        super().visit_select_(op)


class _PrefetchCollector(StmtExprVisitor):
    def __init__(self):
        super().__init__()
        self.names = []

    def visit_call_(self, op):
        if isinstance(op.op, tvm.ir.Op) and op.op.name == "tirx.ptx.prefetch":
            addr = op.args[0]
            # A tensormap address arrives as a u64 handle, so the dialect
            # coerces it with an explicit reinterpret before the call.
            if isinstance(addr, tvm.ir.Call) and addr.op.name == "tirx.reinterpret":
                addr = addr.args[0]
            if (
                isinstance(addr, tvm.ir.Call)
                and addr.op.name == "tirx.address_of"
                and isinstance(addr.args[0], Var)
            ):
                self.names.append(addr.args[0].name)
        super().visit_call_(op)


def _plain_layout(shape, strides=None):
    shape = tuple(shape)
    if strides is None:
        return TileLayout(S[shape])
    return TileLayout(S[shape : tuple(strides)])


def _ranges(region):
    return [Range.from_min_extent(start, extent) for start, extent in region]


def _make_op(
    *,
    g_shape,
    s_shape=None,
    g_region=None,
    s_region=None,
    g_layout=None,
    s_layout=None,
    dtype="float16",
    s_dtype=None,
    direction="g2s",
    config=None,
    g_data=None,
    g_elem_offset=0,
    s_elem_offset=0,
    target_arch="sm_90a",
    sctx=None,
):
    g_shape = tuple(g_shape)
    s_shape = tuple(s_shape or g_shape)
    g_region = tuple(g_region or ((0, extent) for extent in g_shape))
    s_region = tuple(s_region or ((0, extent) for extent in s_shape))
    g_layout = g_layout or _plain_layout(g_shape)
    s_layout = s_layout or _plain_layout(s_shape)
    s_dtype = s_dtype or dtype
    g_buf = tvm.tirx.decl_buffer(
        g_shape,
        dtype,
        "A",
        data=g_data,
        elem_offset=g_elem_offset,
        layout=g_layout,
    )
    s_buf = tvm.tirx.decl_buffer(
        s_shape,
        s_dtype,
        "A_smem",
        scope="shared.dyn",
        elem_offset=s_elem_offset,
        layout=s_layout,
    )
    config = dict(config or {})
    if direction == "g2s":
        config.setdefault("mbar", Var("mbar", "handle"))
        dst = BufferRegion(s_buf, _ranges(s_region))
        src = BufferRegion(g_buf, _ranges(g_region))
    else:
        dst = BufferRegion(g_buf, _ranges(g_region))
        src = BufferRegion(s_buf, _ranges(s_region))
    op = CopyAsync(dst, src, config=config)
    if sctx is None:
        target = tvm.target.Target({"kind": "cuda", "arch": target_arch})
        sctx = DispatchContext(target, ExecScope("thread"), {}, {})
    return op, sctx, g_buf, s_buf


def _lower_direct(variant, **kwargs):
    op, sctx, _, _ = _make_op(**kwargs)
    if variant == "tma_auto":
        impl = copy_tma_auto_impl(op, sctx)
    else:
        impl = copy_tma_explicit_impl(op, sctx)
    return impl, list(sctx.callbacks.get("host_init_stmt", [])), sctx


def _auto_plan(**kwargs):
    op, sctx, _, _ = _make_op(**kwargs)
    return _build_auto_plan(op, sctx)


def _direct_plan(variant, **kwargs):
    op, sctx, _, _ = _make_op(**kwargs)
    if variant == "tma_auto":
        return _build_auto_plan(op, sctx)
    plan, _ = _build_explicit_plan(op, sctx)
    return plan


def _count_tma(stmt):
    counter = _TMACounter()
    counter.visit_stmt(stmt.body if isinstance(stmt, tvm.tirx.PrimFunc) else stmt)
    return counter


def _collect_encodes(stmts):
    collector = _EncodeCollector()
    for stmt in stmts:
        collector.visit_stmt(stmt)
    return collector.calls


def _encode_signature(call):
    rank = int(call.args[3])
    cursor = 5
    dims = tuple(call.args[cursor : cursor + rank])
    cursor += rank
    strides = tuple(call.args[cursor : cursor + rank - 1])
    cursor += rank - 1
    boxes = tuple(call.args[cursor : cursor + rank])
    cursor += rank
    element_strides = tuple(call.args[cursor : cursor + rank])
    cursor += rank
    enums = tuple(call.args[cursor : cursor + 4])
    cursor += 4
    forced_dtype = call.args[cursor] if cursor < len(call.args) else None
    return {
        "dtype": call.args[2].value,
        "rank": rank,
        "base": call.args[4],
        "dims": dims,
        "strides": strides,
        "boxes": boxes,
        "element_strides": element_strides,
        "enums": enums,
        "forced_dtype": forced_dtype,
    }


def _ints(values):
    return tuple(int(value) for value in values)


def _make_spec(**overrides):
    g_data = Var("A", PointerType(PrimType("float16"), "global"))
    s_buf = tvm.tirx.decl_buffer(
        (4, 8),
        "float16",
        "A_smem",
        scope="shared.dyn",
        layout=_plain_layout((4, 8)),
    )
    values = dict(
        descriptor_dtype="float16",
        descriptor_bits=16,
        effective_bytes=2,
        packed_kind=None,
        force_cu_dtype=-1,
        base=g_data,
        base_key="A",
        descriptor_name="A",
        base_byte_offset=0,
        global_dims=(16, 64),
        global_strides=(32,),
        inner_stride=1,
        box_dims=(8, 4),
        element_strides=(1, 1),
        interleave=0,
        swizzle=0,
        l2_promotion=2,
        oob_fill=0,
        direction="g2s",
        load_mode="tile",
        target_arch="sm_90a",
        coordinates=(0, 0),
        gather4=(),
        smem_buffer=s_buf,
        smem_start=(0, 0),
        smem_base_offset=0,
        mbar=Var("mbar", "handle"),
        mbar_is_shared_addr=False,
        cta_group=1,
        cta_mask=0,
        cache_hint="",
        cache_policy=None,
        use_tma_reduce=None,
        payload_bits=1024,
        transaction_bits=1024,
    )
    values.update(overrides)
    return TensorMapSpec(**values)


def _finding(spec, rule):
    return [item for item in validate_tensor_map_spec(spec) if item.rule == rule]


def _from_source(source):
    return tvm.script.from_source(source, {"T": T, "Tx": Tx})


def _lower_module(func, arch="sm_100a"):
    target = tvm.target.Target({"kind": "cuda", "arch": arch})
    with target:
        return tvm.tirx.transform.LowerTIRx()(tvm.IRModule({"main": func}))


def _compile_module(func, arch="sm_100a"):
    target = tvm.target.Target({"kind": "cuda", "arch": arch})
    with target:
        return tvm.compile(
            tvm.IRModule({"main": func}),
            target=target,
            tir_pipeline="tirx",
        )


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


_TMA_EXPLICIT_CASES = {
    "g2s-oob-zero",
    "g2s-oob-nan",
    "s2g-oob-zero",
    "s2g-oob-nan",
    "reject-unknown-oob",
    "reject-g2s-nan-on-non-float",
    "reject-s2g-nan-on-non-float",
}


_TMA_CASE_ERRORS = {
    "g2s-2d-32x512-atom": r"stage=prefix-search: rank: .*got 6",
    "g2s-3d-stride-gap-outer": r"stage=shared-chain:",
    "g2s-multiphase-7x32x512-atom": r"stage=prefix-search: rank: .*got 6",
    "g2s-transpose-32x64": r"stage=prefix-search: global_stride_alignment:",
    "g2s-transpose-64x32": r"stage=prefix-search: global_stride_alignment:",
    "g2s-transpose-partial-region": r"stage=prefix-search: global_stride_alignment:",
    "g2s-transpose-partial-offset": r"stage=prefix-search: global_stride_alignment:",
    "s2g-multiphase-7x32x512-atom": r"stage=prefix-search: rank: .*got 6",
    "s2g-oob-zero": r"oob is only valid for explicit global-to-shared",
    "s2g-oob-nan": r"oob is only valid for explicit global-to-shared",
    "reject-unknown-oob": r"unsupported TensorMap oob='bogus'",
    "reject-g2s-nan-on-non-float": r"nan_oob_dtype:",
    "reject-s2g-nan-on-non-float": r"oob is only valid for explicit global-to-shared",
}


def test_tma_case_matrix_is_complete():
    case_ids = [case.values[0] for case in TMA_CASES]
    expected_ids = set(_TMA_CASE_GOLDENS) | set(_TMA_CASE_ERRORS)
    assert len(case_ids) == 52
    assert len(set(case_ids)) == len(case_ids)
    assert set(case_ids) == expected_ids
    assert not set(_TMA_CASE_GOLDENS) & set(_TMA_CASE_ERRORS)
    assert _TMA_EXPLICIT_CASES <= expected_ids


@pytest.mark.parametrize(("case_id", "kwargs"), TMA_CASES)
def test_tma_codegen(case_id, kwargs):
    variant = "tma_explicit" if case_id in _TMA_EXPLICIT_CASES else "tma_auto"
    if case_id in _TMA_CASE_ERRORS:
        with pytest.raises(Exception, match=_TMA_CASE_ERRORS[case_id]):
            _lower_direct(variant, **kwargs)
        return

    expected = _TMA_CASE_GOLDENS[case_id]
    plan = _direct_plan(variant, **kwargs)
    spec = plan.spec
    assert spec.descriptor_dtype == expected.dtype
    assert _ints(spec.global_dims) == expected.dims
    assert _ints(spec.global_strides) == expected.strides
    assert _ints(spec.box_dims) == expected.boxes
    assert _ints(spec.coordinates) == expected.coordinates
    assert _ints(spec.element_strides) == (1,) * len(expected.dims)
    assert (spec.interleave, spec.swizzle, spec.l2_promotion, spec.oob_fill) == expected.enums
    assert plan.issue_axes == ()

    impl, host, _ = _lower_direct(variant, **kwargs)
    counter = _count_tma(impl)
    assert counter.total == 1
    assert len(counter.calls) == 1
    call = counter.calls[0]
    mod_map, operands = _ptx_call_parts(call)
    assert mod_map["dim"] == f"{len(expected.dims)}d"
    assert _ints(operands["coords"]) == expected.coordinates

    encodes = _collect_encodes(host)
    assert len(encodes) == 1
    signature = _encode_signature(encodes[0])
    assert signature["dtype"] == expected.dtype
    assert _ints(signature["dims"]) == expected.dims
    assert _ints(signature["strides"]) == expected.strides
    assert _ints(signature["boxes"]) == expected.boxes
    assert _ints(signature["element_strides"]) == (1,) * len(expected.dims)
    assert _ints(signature["enums"]) == expected.enums


def test_auto_canonicalization_does_not_create_an_oversized_box():
    plan = _auto_plan(g_shape=(8, 64))
    assert plan.spec.descriptor_dtype == "float16"
    assert plan.spec.rank == 2
    assert _ints(plan.spec.global_dims) == (64, 8)
    assert _ints(plan.spec.box_dims) == (64, 8)
    assert plan.issue_axes == ()


def test_auto_sw128_gt_grouping_and_cuda_order():
    shape = (8, 256)
    impl, host, _ = _lower_direct(
        "tma_auto",
        g_shape=shape,
        s_layout=mma_shared_layout("float16", 3, shape),
    )
    assert _count_tma(impl).total == 1
    signature = _encode_signature(_collect_encodes(host)[0])
    assert signature["dtype"] == "float16"
    assert signature["rank"] == 3
    assert _ints(signature["dims"]) == (64, 8, 4)
    assert _ints(signature["strides"]) == (512, 128)
    assert _ints(signature["boxes"]) == (64, 8, 4)
    assert _ints(signature["enums"]) == (0, 3, 2, 0)


def test_auto_retains_coordinate_only_global_dimension():
    g_shape = (64, 128, 2, 4, 4096)
    s_shape = (64, 64, 2, 4)
    plan = _auto_plan(
        g_shape=g_shape,
        s_shape=s_shape,
        g_region=((0, 64), (32, 64), (0, 2), (0, 4), (7, 1)),
        g_layout=_plain_layout(g_shape, (1, 512, 256, 64, 65536)),
        s_layout=_plain_layout(s_shape, (1, 64, 4096, 8192)),
    )
    assert plan.spec.rank == 5
    assert _ints(plan.spec.global_dims) == g_shape
    assert _ints(plan.spec.global_strides) == (1024, 512, 128, 131072)
    assert _ints(plan.spec.box_dims) == (64, 64, 2, 4, 1)
    assert _ints(plan.spec.coordinates) == (0, 32, 0, 0, 7)
    assert plan.issue_axes == ()


def test_auto_retains_unit_copy_dimensions_with_coordinates_and_strides():
    plan = _auto_plan(
        g_shape=(4, 256, 8, 512),
        s_shape=(1, 128, 1, 128),
        g_region=((2, 1), (64, 128), (3, 1), (128, 128)),
    )
    assert plan.spec.rank == 4
    assert _ints(plan.spec.global_dims) == (512, 256, 4, 8)
    assert _ints(plan.spec.global_strides) == (8192, 2097152, 1024)
    assert _ints(plan.spec.box_dims) == (128, 128, 1, 1)
    assert _ints(plan.spec.coordinates) == (128, 64, 2, 3)
    assert int(plan.spec.base_byte_offset) == 0
    assert plan.issue_axes == ()


def test_auto_canonicalizes_dynamic_stage_slice_before_global_grouping():
    kv_row = Var("kv_row", "int32")
    head = Var("head", "int32")
    stage = Var("stage", "int32")
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    sctx = DispatchContext(
        target,
        ExecScope("thread"),
        {},
        {
            kv_row: Range.from_min_extent(0, 3969),
            head: Range.from_min_extent(0, 16),
            stage: Range.from_min_extent(0, 3),
        },
    )
    plan = _auto_plan(
        g_shape=(1, 4096, 16, 128),
        s_shape=(3, 128, 128),
        g_region=((0, 1), (kv_row, 128), (head, 1), (0, 128)),
        s_region=((stage, 1), (0, 128), (0, 128)),
        g_layout=TileLayout(S[(1, 4096, 16, 128)]),
        s_layout=TileLayout(S[(3, 128, 2, 64) : (16384, 64, 8192, 1)]),
        sctx=sctx,
    )
    assert plan.spec.rank == 3
    assert _ints(plan.spec.global_dims) == (64, 4096, 32)
    assert _ints(plan.spec.global_strides) == (4096, 128)
    assert _ints(plan.spec.box_dims) == (64, 128, 2)
    analyzer = Analyzer()
    assert analyzer.can_prove_equal(plan.spec.coordinates[0], 0)
    assert analyzer.can_prove_equal(plan.spec.coordinates[1], kv_row)
    assert analyzer.can_prove_equal(plan.spec.coordinates[2], head * 2)
    assert analyzer.can_prove_equal(plan.spec.smem_base_offset, stage * 16384)
    assert int(plan.spec.transaction_bits) // 8 == 32 * 1024
    assert plan.issue_axes == ()


def test_auto_shared_pointer_counts_each_offset_once():
    warp_offset = Var("warp_offset", "int32")
    plan = _auto_plan(
        g_shape=(2, 64),
        s_shape=(4, 64),
        s_region=((2, 2), (0, 64)),
        s_layout=TileLayout(S[(4, 64)] + warp_offset * 64),
        s_elem_offset=32,
    )
    assert Analyzer().can_prove_equal(
        plan.spec.smem_base_offset,
        32 + warp_offset * 64 + 128,
    )


def test_auto_defers_dynamic_global_dimension_bounds_to_runtime():
    seq_len = Var("seq_len", "uint32")
    row = Var("row", "uint32")
    plan = _auto_plan(
        g_shape=(seq_len * T.uint32(64), 64),
        s_shape=(128, 64),
        g_region=((row, 128), (0, 64)),
        g_layout=TileLayout(S[(seq_len * T.uint32(64), 64)]),
        s_layout=TileLayout(S[(128, 64)]),
        dtype="uint8",
    )
    assert plan.spec.rank == 2
    assert Analyzer().can_prove_equal(plan.spec.global_dims[0], 64)
    assert Analyzer().can_prove_equal(plan.spec.global_dims[1], seq_len * T.uint32(64))
    assert _ints(plan.spec.box_dims) == (64, 128)
    assert Analyzer().can_prove_equal(plan.spec.coordinates[0], 0)
    assert Analyzer().can_prove_equal(plan.spec.coordinates[1], row)
    assert plan.issue_axes == ()


def test_dispatch_propagates_flat_bind_to_auto_coordinate_proof():
    func = _from_source(
        """
@T.prim_func
def bind_coordinate(D_ptr: T.handle):
    D = T.match_buffer(D_ptr, (33360, 6144), "bfloat16")
    T.device_entry()
    block = T.cta_id([192])
    tid = T.thread_id([1])
    tile_index = T.alloc_local((1,), "int32")
    D_smem = T.alloc_buffer(
        (2, 16, 128),
        "bfloat16",
        scope="shared.dyn",
        layout=T.TileLayout(T.S[(2, 16, 2, 64) : (2048, 64, 1024, 1)]),
    )
    tile_index[0] = block
    d_n: T.let = tile_index[0] * 128
    if tid == 0:
        Tx.copy_async(
            D[0:16, d_n : d_n + 128],
            D_smem[0, :, :],
            dispatch="tma_auto",
        )
"""
    )
    lowered = _lower_module(func)
    encodes = _collect_encodes([lowered["main"].body])
    assert len(encodes) == 1
    signature = _encode_signature(encodes[0])
    assert _ints(signature["dims"]) == (64, 33360, 96)
    assert _ints(signature["strides"]) == (12288, 128)
    assert _ints(signature["boxes"]) == (64, 16, 2)

    counter = _count_tma(lowered["main"])
    assert counter.total == 1
    call = counter.calls[0]
    mod_map, operands = _ptx_call_parts(call)
    assert mod_map["dim"] == "3d"
    assert int(operands["coords"][0]) == 0
    assert int(operands["coords"][1]) == 0
    assert str(operands["coords"][2]) == "tile_index[0] * 2"


def test_auto_recovers_stride_from_split_coordinate_only_dimension():
    plan = _auto_plan(
        g_shape=(24576, 7168),
        s_shape=(128, 128),
        g_region=((128, 128), (0, 128)),
        g_layout=TileLayout(S[(4, 6144, 7168) : (44040192, 7168, 1)]),
    )
    assert plan.spec.rank == 2
    assert _ints(plan.spec.global_dims) == (7168, 24576)
    assert _ints(plan.spec.global_strides) == (14336,)
    assert _ints(plan.spec.box_dims) == (128, 128)
    assert _ints(plan.spec.coordinates) == (0, 128)
    assert plan.issue_axes == ()


def test_auto_preserves_coordinate_stride_for_interleaved_tensor_dimensions():
    plan = _auto_plan(
        g_shape=(2048, 128),
        s_shape=(5, 128, 16),
        g_region=((128, 128), (16, 16)),
        s_region=((0, 1), (0, 128), (0, 16)),
        g_layout=TileLayout(S[(16, 4, 32, 32, 4) : (16384, 4, 16, 512, 1)]),
        s_layout=TileLayout(S[(5, 4, 32, 4, 4) : (2048, 4, 16, 512, 1)]),
        dtype="uint8",
    )
    assert plan.spec.descriptor_dtype == "uint16"
    assert plan.spec.rank == 3
    assert _ints(plan.spec.global_dims) == (256, 32, 16)
    assert _ints(plan.spec.global_strides) == (512, 16384)
    assert _ints(plan.spec.box_dims) == (256, 4, 1)
    assert _ints(plan.spec.coordinates) == (0, 4, 1)
    assert plan.issue_axes == ()


def test_auto_recovers_leading_unit_dimension_stride():
    plan = _auto_plan(
        g_shape=(1, 8192),
        s_shape=(256,),
        g_region=((0, 1), (512, 256)),
        g_layout=_plain_layout((1, 8192), (8192, 1)),
    )
    assert plan.spec.rank == 1
    assert _ints(plan.spec.global_dims) == (8192,)
    assert _ints(plan.spec.global_strides) == ()
    assert _ints(plan.spec.box_dims) == (256,)
    assert _ints(plan.spec.coordinates) == (512,)
    assert plan.issue_axes == ()


def test_auto_maximum_prefix_and_mixed_radix_issue_pointer():
    plan = _auto_plan(g_shape=(512, 64))
    assert _ints(plan.spec.box_dims) == (64, 1)
    assert len(plan.issue_axes) == 1
    assert int(plan.issue_axes[0].extent) == 512
    assert int(plan.issue_axes[0].smem_stride) == 64
    offset, coords = plan.offsets_and_coords(IntImm("int32", 7))
    analyzer = Analyzer()
    assert int(analyzer.simplify(offset)) == 448
    assert _ints(analyzer.simplify(coord) for coord in coords) == (0, 7)

    impl, _, _ = _lower_direct("tma_auto", g_shape=(512, 64))
    assert _count_tma(impl).total == 512


def test_copy_tma_host_init_dtype_is_string():
    """The host-init encode call must carry the dtype as a StringImm, not a
    packed enum -- ``_encode_signature`` reads ``args[2].value`` as a str."""
    _, host_init_stmts, _ = _lower_direct(
        "tma_auto",
        g_shape=(8, 256),
        g_region=((0, 8), (0, 256)),
        s_shape=(8, 256),
        s_region=((0, 8), (0, 256)),
        dtype="float16",
    )
    encode_call = _collect_encodes(host_init_stmts)[0]
    assert isinstance(encode_call.args[2], StringImm)
    assert encode_call.args[2].value == "float16"


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        (
            {
                "g_shape": (8, 32, 64),
                "s_layout": _plain_layout((8, 32, 64), (4096, 64, 1)),
            },
            "stage=shared-chain",
        ),
        (
            {
                "g_shape": (128, 64),
                "g_region": ((4, 24), (0, 64)),
                "s_shape": (3, 64, 8),
                "s_layout": _plain_layout((3, 64, 8), (512, 1, 64)),
            },
            "stage=coordinate",
        ),
        (
            {
                "g_shape": (130, 64),
                "g_region": ((0, 24), (0, 64)),
                "s_shape": (3, 64, 8),
                "s_layout": _plain_layout((3, 64, 8), (512, 1, 64)),
            },
            "stage=global-shape",
        ),
    ],
)
def test_auto_requires_complete_chain_and_exact_division(kwargs, error):
    with pytest.raises(Exception, match=error):
        _auto_plan(**kwargs)


def test_auto_canonicalization_repairs_rank_and_unaligned_stride():
    rank_plan = _auto_plan(g_shape=(2, 2, 2, 2, 2, 8))
    assert rank_plan.spec.descriptor_dtype == "float16"
    assert rank_plan.spec.rank == 1
    assert _ints(rank_plan.spec.global_dims) == (256,)

    stride_plan = _auto_plan(g_shape=(32, 8), dtype="uint8")
    assert stride_plan.spec.descriptor_dtype == "uint8"
    assert stride_plan.spec.rank == 1
    assert _ints(stride_plan.spec.global_dims) == (256,)


def test_auto_promotion_preserves_payload_and_shared_pointer():
    plan = _auto_plan(g_shape=(64, 8), dtype="uint8")
    assert plan.spec.descriptor_dtype == "uint16"
    assert plan.spec.rank == 1
    assert _ints(plan.spec.global_dims) == (256,)
    assert _ints(plan.spec.box_dims) == (256,)
    assert int(plan.spec.payload_bits) == 4096
    assert plan.spec.smem_start == (0, 0)
    assert str(plan.spec.smem_buffer.dtype) == "uint8"


def test_auto_promotes_before_crossing_box_blocked_inner_chain_boundary():
    region = ((0, 1), (0, 4), (0, 32), (0, 4), (0, 4))
    plan = _auto_plan(
        g_shape=(8, 16, 32, 4, 4),
        g_region=region,
        s_region=region,
        dtype="uint8",
    )
    assert plan.spec.descriptor_dtype == "uint16"
    assert _ints(plan.spec.global_dims) == (256, 16, 8)
    assert _ints(plan.spec.global_strides) == (512, 8192)
    assert _ints(plan.spec.box_dims) == (256, 4, 1)
    assert _ints(plan.spec.coordinates) == (0, 0, 0)
    assert plan.issue_axes == ()
    assert int(plan.spec.payload_bits) == 16384


def test_auto_promotion_rejects_unsafe_modes():
    spec = _make_spec(
        descriptor_dtype="uint8",
        descriptor_bits=8,
        effective_bytes=1,
        global_dims=(8, 64),
        global_strides=(8,),
        box_dims=(8, 64),
        coordinates=(0, 0),
        payload_bits=4096,
        transaction_bits=4096,
    )
    plan = TMAPlan(spec)
    assert _promote_auto_once(plan) is not None
    assert _promote_auto_once(replace(plan, spec=replace(spec, box_dims=(7, 64)))) is None
    assert _promote_auto_once(replace(plan, spec=replace(spec, coordinates=(1, 0)))) is None
    assert (
        _promote_auto_once(
            replace(
                plan,
                issue_axes=(AutoIssueAxis(2, 8, (IssueCoord(0, 1, 2),)),),
            )
        )
        is None
    )
    assert _promote_auto_once(replace(plan, spec=replace(spec, use_tma_reduce="add"))) is None
    assert _promote_auto_once(replace(plan, spec=replace(spec, force_cu_dtype=11))) is None
    assert _promote_auto_once(replace(plan, spec=replace(spec, element_strides=(1, 2)))) is None
    assert _promote_auto_once(replace(plan, spec=replace(spec, inner_stride=2))) is None
    assert (
        _promote_auto_once(
            replace(
                plan,
                spec=replace(
                    spec,
                    descriptor_dtype="float4_e2m1fn",
                    descriptor_bits=4,
                    packed_kind="16u4_align16",
                ),
            )
        )
        is None
    )


def test_auto_odd_partial_and_unrepairable_box_fail_loudly():
    with pytest.raises(Exception, match="inner_box_bytes|global_stride_alignment"):
        _auto_plan(
            g_shape=(8, 64, 8),
            g_region=((0, 8), (0, 64), (0, 5)),
            s_shape=(8, 64, 5),
            dtype="uint8",
        )
    with pytest.raises(Exception, match="boxDim"):
        _auto_plan(g_shape=(4096,), dtype="uint8")


def test_auto_descriptor_cache_key_includes_promoted_dtype():
    data = Var("A", PointerType(PrimType("uint8"), "global"))
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_90a"})
    sctx = DispatchContext(target, ExecScope("thread"), {}, {})
    for rows in (32, 64):
        op, _, _, _ = _make_op(
            g_shape=(rows, 8),
            dtype="uint8",
            g_data=data,
            sctx=sctx,
        )
        copy_tma_auto_impl(op, sctx)
    signatures = [
        _encode_signature(call) for call in _collect_encodes(sctx.callbacks["host_init_stmt"])
    ]
    assert [item["dtype"] for item in signatures] == ["uint8", "uint16"]
    assert [_ints(item["dims"]) for item in signatures] == [(256,), (256,)]


@pytest.mark.parametrize("rank", range(1, 6))
def test_explicit_direct_rank_1_through_5(rank):
    shape = (8,) * rank
    impl, host, _ = _lower_direct("tma_explicit", g_shape=shape)
    assert _count_tma(impl).total == 1
    signature = _encode_signature(_collect_encodes(host)[0])
    assert signature["rank"] == rank
    assert _ints(signature["dims"]) == shape
    assert _ints(signature["boxes"]) == shape


def test_explicit_retains_symbolic_outer_stride_for_runtime_encode():
    row_stride = Var("row_stride", "int32")
    shape = (64, 64)
    impl, host, _ = _lower_direct(
        "tma_explicit",
        g_shape=shape,
        dtype="bfloat16",
        g_layout=_plain_layout(shape, (row_stride, 1)),
    )
    assert _count_tma(impl).total == 1
    signature = _encode_signature(_collect_encodes(host)[0])
    tvm.ir.assert_structural_equal(signature["strides"][0], row_stride * 2)


def test_explicit_rejects_unknown_or_nonunit_implicit_inner_stride():
    inner_stride = Var("inner_stride", "int32")
    with pytest.raises(Exception, match="innermost memory stride must be provably one"):
        _lower_direct(
            "tma_explicit",
            g_shape=(64, 64),
            g_layout=_plain_layout((64, 64), (64, inner_stride)),
        )
    with pytest.raises(Exception, match="innermost memory stride must be provably one"):
        _lower_direct(
            "tma_explicit",
            g_shape=(64, 64),
            g_layout=_plain_layout((64, 64), (64, 2)),
        )


def test_explicit_view_base_offset_is_encoded_without_coordinate_rewrite():
    _, host, _ = _lower_direct(
        "tma_explicit",
        g_shape=(64, 64),
        g_elem_offset=8,
    )
    signature = _encode_signature(_collect_encodes(host)[0])
    assert isinstance(signature["base"], tvm.ir.Call)
    assert signature["base"].op.name == "tirx.handle_add_byte_offset"
    assert int(signature["base"].args[1]) == 16
    assert _ints(signature["dims"]) == (64, 64)


def test_explicit_oob_reduce_and_tf32_configs():
    _, load_host, _ = _lower_direct(
        "tma_explicit",
        g_shape=(64, 64),
        config={"oob": "nan"},
    )
    assert _ints(_encode_signature(_collect_encodes(load_host)[0])["enums"])[-1] == 1

    store_impl, _, _ = _lower_direct(
        "tma_explicit",
        g_shape=(64, 64),
        direction="s2g",
        config={"use_tma_reduce": "add"},
    )
    counter = _count_tma(store_impl)
    assert counter.total == 1
    assert counter.calls[0].op.name == "tirx.ptx.cp_reduce_async_bulk_tensor"

    _, tf32_host, _ = _lower_direct(
        "tma_explicit",
        g_shape=(64, 32),
        dtype="float32",
        config={"tma_dtype": "tf32"},
    )
    signature = _encode_signature(_collect_encodes(tf32_host)[0])
    assert int(signature["forced_dtype"]) == 11


def test_explicit_never_repairs_rank_or_descriptor_units():
    with pytest.raises(Exception, match="rank"):
        _lower_direct("tma_explicit", g_shape=(2, 2, 2, 2, 2, 8))
    with pytest.raises(Exception, match="global_stride_alignment"):
        _lower_direct("tma_explicit", g_shape=(32, 8), dtype="uint8")


def test_explicit_requires_box_linear_shared_slice():
    with pytest.raises(Exception, match="shared-layout"):
        _lower_direct(
            "tma_explicit",
            g_shape=(4, 64),
            s_layout=_plain_layout((4, 64), (128, 1)),
        )


def test_explicit_gather_uses_extracted_swizzled_slice_pointer():
    impl, _, _ = _lower_direct(
        "tma_explicit",
        g_shape=(8, 64),
        s_shape=(8, 64),
        g_region=((0, 1), (0, 64)),
        s_region=((4, 4), (0, 64)),
        s_layout=mma_shared_layout("float16", 3, (8, 64)),
        config={"gather4": [0, 1, 2, 3]},
        target_arch="sm_100a",
    )
    _, operands = _ptx_call_parts(_count_tma(impl).calls[0])
    shared_ptr = _unwrap_shared_addr(operands["dst_mem"][0])
    assert shared_ptr.op.name == "tirx.address_of"
    assert int(shared_ptr.args[0].source.elem_offset) == 0
    assert int(shared_ptr.args[0].indices[0]) == 256


def test_explicit_shared_pointer_counts_each_offset_once():
    warp_offset = Var("warp_offset", "int32")
    layout = TileLayout(S[(4, 64)] + warp_offset * 64)
    impl, _, _ = _lower_direct(
        "tma_explicit",
        g_shape=(2, 64),
        s_shape=(4, 64),
        s_region=((2, 2), (0, 64)),
        s_layout=layout,
        s_elem_offset=32,
    )
    _, operands = _ptx_call_parts(_count_tma(impl).calls[0])
    shared_ptr = _unwrap_shared_addr(operands["dst_mem"][0])
    assert shared_ptr.op.name == "tirx.address_of"
    pointer_index = shared_ptr.args[0].indices[0]
    assert Analyzer().can_prove_equal(pointer_index, 32 + warp_offset * 64 + 128)


def test_explicit_allows_different_operand_ranks_with_equal_payload_bytes():
    source = _from_source(
        """
@T.prim_func
def rank_change(A_ptr: T.handle):
    A = T.match_buffer(A_ptr, (8, 8), "float16")
    T.device_entry()
    T.cta_id([1])
    tid = T.thread_id([1])
    dyn = T.alloc_buffer((65,), "uint64", scope="shared.dyn")
    T.attr({"tirx.dyn_smem_bytes": 65 * 8})
    A_smem = T.decl_buffer((64,), "float16", dyn.data, layout=T.TileLayout(T.S[64]))
    mbar = T.decl_buffer((1,), "uint64", dyn.data, elem_offset=16)
    if tid == 0:
        Tx.copy_async(
            A_smem[:], A[:, :], dispatch="tma_explicit", mbar=mbar.ptr_to([0])
        )
"""
    )
    lowered = _lower_module(source, "sm_90a")
    assert _count_tma(lowered["main"]).total == 1


_SELECTOR_SOURCE = """
@T.prim_func
def selector_gather(
    A_ptr: T.handle,
    B_ptr: T.handle,
    flag: T.int32,
):
    A = T.match_buffer(A_ptr, (256, 64), "bfloat16")
    B = T.match_buffer(B_ptr, (512, 80), "bfloat16")
    B_view = B.sub[16:512, 8:72]
    T.device_entry()
    T.cta_id([1])
    tid = T.thread_id([128])
    dyn = T.alloc_buffer((520,), "uint64", scope="shared.dyn")
    T.attr({"tirx.dyn_smem_bytes": 520 * 8})
    A_smem = T.decl_buffer(
        (4, 64), "bfloat16", dyn.data, layout=T.TileLayout(T.S[4, 64])
    )
    mbar = T.decl_buffer((1,), "uint64", dyn.data, elem_offset=64)
    if tid == 0:
        T.ptx.mbarrier.init.shared.b64(mbar.ptr_to([0]), T.uint32(1))
        Tx.copy_async(
            A_smem[:, :],
            A[0:1, :],
            dispatch="tma_explicit",
            mbar=mbar.ptr_to([0]),
            gather4=[1, 2, 3, 4],
            src_selector=[(flag != 0, B_view)],
        )
"""


def test_explicit_gather_selector_encodes_each_map_selects_address_and_issues_once():
    func = _from_source(_SELECTOR_SOURCE)
    lowered = _lower_module(func)
    encodes = _collect_encodes([lowered["main"].body])
    assert len(encodes) == 2
    signatures = [_encode_signature(call) for call in encodes]
    by_dims = {_ints(item["dims"]): item for item in signatures}
    assert set(by_dims) == {(64, 256), (64, 496)}
    assert _ints(by_dims[(64, 256)]["strides"]) == (128,)
    assert _ints(by_dims[(64, 496)]["strides"]) == (160,)
    assert [_ints(item["boxes"]) for item in signatures] == [(64, 1), (64, 1)]

    selects = _SelectCollector()
    selects.visit_stmt(lowered["main"].body)
    assert len(selects.nodes) == 1
    assert selects.nodes[0].ty == PrimType("uint64")
    assert _count_tma(lowered["main"]).total == 1

    executable = _compile_module(func)
    cuda_source = executable.mod.imports[0].inspect_source()
    assert "uint64_t selected_tensormap" in cuda_source
    assert "flag != 0) ?" in cuda_source
    assert "if (flag" not in cuda_source
    # One helper definition plus one call site: the selector collapses the two
    # descriptors into a single gather4 issue.
    assert (
        cuda_source.count(
            "tvm_builtin_ptx_cp_async_bulk_tensor_g2s_cluster_async_bulk_tensor_2d"
            "_shared__cluster_global_tile__gather4_mbarrier__complete_tx__bytes"
            "_cta_group__1("
        )
        == 2
    )


def test_explicit_selector_uses_first_true_and_requires_static_compatibility():
    main = _make_spec()
    compatible = replace(main, global_dims=(16, 128), global_strides=(64,), base_key="B")
    _selector_compatibility(main, compatible, 0)
    with pytest.raises(Exception, match="descriptor dtype"):
        _selector_compatibility(
            main,
            replace(
                compatible,
                descriptor_dtype="uint16",
            ),
            0,
        )
    symbolic_box = Var("candidate_box", "int32")
    with pytest.raises(Exception, match="incompatible box"):
        _selector_compatibility(main, replace(compatible, box_dims=(symbolic_box, 4)), 0)


def test_explicit_gather_validates_target_source_and_destination_layouts():
    common = dict(
        g_shape=(256, 96),
        g_region=((0, 1), (0, 64)),
        s_shape=(4, 64),
        g_layout=_plain_layout((256, 96), (96, 1)),
        config={"gather4": [1, 2, 3, 4]},
        target_arch="sm_100a",
    )
    impl, host, _ = _lower_direct("tma_explicit", **common)
    assert _count_tma(impl).total == 1
    assert _ints(_encode_signature(_collect_encodes(host)[0])["strides"]) == (192,)

    with pytest.raises(Exception, match="SM100"):
        _lower_direct("tma_explicit", **{**common, "target_arch": "sm_90a"})
    with pytest.raises(Exception, match="innermost memory stride"):
        _lower_direct(
            "tma_explicit",
            **{
                **common,
                "g_layout": _plain_layout((256, 96), (1, 256)),
            },
        )
    with pytest.raises(Exception, match="shared-layout|box-linear"):
        _lower_direct(
            "tma_explicit",
            **{
                **common,
                "s_layout": _plain_layout((4, 64), (128, 1)),
            },
        )


def test_nested_selector_parser_printer_roundtrip_and_lowering_remap():
    func = _from_source(_SELECTOR_SOURCE)
    script = func.script(extra_config={"tirx.prefix": "T"})
    reparsed = _from_source(script)
    tvm.ir.assert_structural_equal(func, reparsed, map_free_vars=True)
    lowered = _lower_module(reparsed)
    assert len(_collect_encodes([lowered["main"].body])) == 2
    assert _count_tma(lowered["main"]).total == 1


def test_prefetch_only_main_selector_descriptor():
    func = _from_source(
        _SELECTOR_SOURCE.replace(
            "gather4=[1, 2, 3, 4],",
            "gather4=[1, 2, 3, 4],\n            prefetch_tensormap=True,",
        )
    )
    lowered = _lower_module(func)
    collector = _PrefetchCollector()
    collector.visit_stmt(lowered["main"].body)
    assert collector.names == ["A_tensormap"]


def _build_sparse_decode_qo_tma_regression():
    q_layout = mma_shared_layout("bfloat16", 3, (64, 512))
    q_tail_layout = mma_shared_layout("bfloat16", 2, (64, 64))
    o_layout = mma_shared_layout("bfloat16", 3, (64, 512))
    q_elements = 64 * 512
    q_tail_elements = 64 * 64
    o_elements = 64 * 512
    shared_bytes = (q_elements + q_tail_elements + o_elements) * 2

    # fmt: off
    @T.prim_func
    def kernel(
        Q_ptr: T.handle,
        O_ptr: T.handle,
        q_stride_b: T.int64,
        q_stride_s: T.int64,
        q_stride_h: T.int64,
        o_stride_b: T.int64,
        o_stride_s: T.int64,
        o_stride_h: T.int64,
    ):
        Q_storage = T.match_buffer(Q_ptr, (64 * 576,), "bfloat16")
        O_storage = T.match_buffer(O_ptr, (64 * 512,), "bfloat16")
        Q = Q_storage.view(
            1,
            1,
            64,
            576,
            layout=T.TileLayout(
                T.S[(1, 1, 64, 576) : (q_stride_b, q_stride_s, q_stride_h, 1)]
            ),
        )
        O_view = O_storage.view(
            1,
            1,
            64,
            512,
            layout=T.TileLayout(
                T.S[(1, 1, 64, 512) : (o_stride_b, o_stride_s, o_stride_h, 1)]
            ),
        )
        T.device_entry()
        T.cta_id([1])
        tid = T.thread_id([128])
        dyn = T.alloc_buffer((shared_bytes + 8,), "uint8", scope="shared.dyn")
        T.attr({"tirx.dyn_smem_bytes": shared_bytes + 8})
        q_smem = T.decl_buffer(
            (64, 512), "bfloat16", dyn.data, scope="shared.dyn", layout=q_layout
        )
        q_tail_smem = T.decl_buffer(
            (64, 64),
            "bfloat16",
            dyn.data,
            elem_offset=q_elements,
            scope="shared.dyn",
            layout=q_tail_layout,
        )
        o_smem = T.decl_buffer(
            (64, 512),
            "bfloat16",
            dyn.data,
            elem_offset=q_elements + q_tail_elements,
            scope="shared.dyn",
            layout=o_layout,
        )
        mbar = T.decl_buffer(
            (1,), "uint64", dyn.data, elem_offset=shared_bytes // 8, scope="shared.dyn"
        )
        q_tail_smem_tma = q_tail_smem.view(64, 2, 32).permute(1, 0, 2)
        q_tail_gmem_tma = Q.sub[0, 0, :, 512:576].view(64, 2, 32).permute(1, 0, 2)
        if tid == 0:
            for q_tile in T.unroll(8):
                Tx.copy_async(
                    q_smem[:, q_tile * 64 : (q_tile + 1) * 64],
                    Q[0, 0, :, q_tile * 64 : (q_tile + 1) * 64],
                    dispatch="tma_explicit",
                    mbar=mbar.ptr_to([0]),
                    cache_hint="evict_first",
                    tensormap_l2_promotion="L2::128B",
                )
            Tx.copy_async(
                q_tail_smem_tma[:, :, :],
                q_tail_gmem_tma[:, :, :],
                dispatch="tma_explicit",
                mbar=mbar.ptr_to([0]),
                cache_hint="evict_first",
                tensormap_l2_promotion="L2::128B",
            )
            for o_tile in T.unroll(8):
                store_tile: T.let = o_tile
                Tx.copy_async(
                    O_view[0, 0, :, store_tile * 64 : (store_tile + 1) * 64],
                    o_smem[:, store_tile * 64 : (store_tile + 1) * 64],
                    dispatch="tma_explicit",
                    cache_hint="evict_first",
                    tensormap_l2_promotion="L2::128B",
                )
        # fmt: on

    return kernel


def test_sparse_decode_runtime_stride_q_tail_o_counts_and_descriptor_dedup():
    kernel = _build_sparse_decode_qo_tma_regression()
    lowered = _lower_module(kernel)
    counter = _count_tma(lowered["main"])
    load_op = "tirx.ptx.cp_async_bulk_tensor_g2s_cluster"
    store_op = "tirx.ptx.cp_async_bulk_tensor_s2g"
    assert sum(weight for call, weight in counter.weighted_calls if call.op.name == load_op) == 9
    assert sum(weight for call, weight in counter.weighted_calls if call.op.name == store_op) == 8

    encodes = _collect_encodes([lowered["main"].body])
    signatures = [_encode_signature(call) for call in encodes]
    assert len(signatures) == 3
    swizzles = [int(signature["enums"][1]) for signature in signatures]
    assert swizzles.count(3) == 2
    assert swizzles.count(2) == 1
    assert all(
        any(not isinstance(stride, IntImm) for stride in signature["strides"])
        for signature in signatures
    )
    _compile_module(kernel)


@pytest.mark.parametrize(
    ("dtype", "cols", "swizzle"),
    [
        ("int64", 64, 0),
        ("bfloat16", 32, 2),
        ("bfloat16", 64, 3),
    ],
)
def test_sparse_decode_nope_rope_descriptor_dtypes_and_swizzles(dtype, cols, swizzle):
    s_layout = (
        _plain_layout((8, cols)) if swizzle == 0 else mma_shared_layout(dtype, swizzle, (8, cols))
    )
    _, host, _ = _lower_direct(
        "tma_explicit",
        g_shape=(256, cols),
        s_shape=(8, cols),
        g_region=((0, 1), (0, cols)),
        s_region=((0, 4), (0, cols)),
        config={"gather4": [0, 1, 2, 3]},
        s_layout=s_layout,
        dtype=dtype,
        target_arch="sm_100a",
    )
    signature = _encode_signature(_collect_encodes(host)[0])
    assert signature["dtype"] == dtype
    assert int(signature["enums"][1]) == swizzle


@pytest.mark.parametrize(
    ("cols", "row_stride", "expected_bytes"),
    [
        (64, 82, 656),
        (56, 72, 576),
    ],
)
def test_sparse_decode_kv_descriptor_uses_tma_row_stride(cols, row_stride, expected_bytes):
    _, host, _ = _lower_direct(
        "tma_explicit",
        g_shape=(256, cols),
        s_shape=(4, cols),
        g_region=((0, 1), (0, cols)),
        g_layout=_plain_layout((256, cols), (row_stride, 1)),
        dtype="int64",
        config={"gather4": [0, 1, 2, 3]},
        target_arch="sm_100a",
    )
    signature = _encode_signature(_collect_encodes(host)[0])
    assert _ints(signature["strides"]) == (expected_bytes,)
    assert 584 not in _ints(signature["strides"])


def test_sparse_decode_absent_extra_map_has_no_dummy_descriptor_or_select():
    source = _SELECTOR_SOURCE.replace(
        "            src_selector=[(flag != 0, B_view)],\n",
        "            prefetch_tensormap=True,\n",
    )
    lowered = _lower_module(_from_source(source))
    assert len(_collect_encodes([lowered["main"].body])) == 1
    selects = _SelectCollector()
    selects.visit_stmt(lowered["main"].body)
    assert selects.nodes == []
    prefetches = _PrefetchCollector()
    prefetches.visit_stmt(lowered["main"].body)
    assert prefetches.names == ["A_tensormap"]


@pytest.mark.parametrize(
    ("mutate", "rule"),
    [
        (lambda spec: replace(spec, global_dims=(0, 64)), "global_dim"),
        (lambda spec: replace(spec, global_dims=((1 << 32) + 1, 64)), "global_dim"),
        (lambda spec: replace(spec, global_strides=(-16,)), "global_stride_range"),
        (lambda spec: replace(spec, global_strides=(8,)), "global_stride_alignment"),
        (lambda spec: replace(spec, global_strides=(1 << 40,)), "global_stride_range"),
        (lambda spec: replace(spec, base_byte_offset=2), "global_base_alignment"),
        (lambda spec: replace(spec, box_dims=(0, 4)), "box_dim"),
        (lambda spec: replace(spec, box_dims=(257, 4)), "box_dim"),
        (lambda spec: replace(spec, box_dims=(7, 4)), "inner_box_bytes"),
        (lambda spec: replace(spec, element_strides=(0, 1)), "element_stride"),
        (lambda spec: replace(spec, element_strides=(9, 1)), "element_stride"),
        (lambda spec: replace(spec, element_strides=(2, 1)), "inner_stride"),
        (lambda spec: replace(spec, inner_stride=2), "inner_stride"),
        (lambda spec: replace(spec, interleave=3), "interleave"),
        (lambda spec: replace(spec, swizzle=9), "swizzle"),
        (lambda spec: replace(spec, l2_promotion=4), "l2_promotion"),
        (lambda spec: replace(spec, oob_fill=2), "oob"),
        (lambda spec: replace(spec, target_arch="sm_80"), "target"),
        (lambda spec: replace(spec, cta_group=3), "cta_group"),
        (lambda spec: replace(spec, coordinates=(0,)), "array_lengths"),
        (lambda spec: replace(spec, force_cu_dtype=7), "forced_cuda_dtype"),
        (lambda spec: replace(spec, descriptor_bits=8), "dtype_bits"),
        (
            lambda spec: replace(spec, swizzle=1, box_dims=(24, 4)),
            "swizzle_inner_box",
        ),
        (
            lambda spec: replace(
                spec,
                descriptor_dtype="int32",
                descriptor_bits=32,
                effective_bytes=4,
                global_strides=(64,),
                box_dims=(4, 4),
                oob_fill=1,
            ),
            "nan_oob_dtype",
        ),
    ],
)
def test_shared_validator_static_boundary_matrix(mutate, rule):
    findings = _finding(mutate(_make_spec()), rule)
    assert any(item.status == ProofStatus.DISPROVEN for item in findings)


def test_shared_validator_unknown_policy_differs_for_auto_and_explicit():
    stride = Var("runtime_stride", "int64")
    spec = _make_spec(global_strides=(stride,))
    assert _validation_failures(spec, auto=False) == []
    auto_failures = _validation_failures(spec, auto=True)
    assert {item.rule for item in auto_failures} >= {
        "global_stride_range",
        "global_stride_alignment",
    }


def test_auto_defers_only_unknown_global_dimension_bounds_to_runtime():
    dynamic_dim = Var("runtime_global_dim", "int64")
    dynamic = _make_spec(global_dims=(dynamic_dim, 64))
    assert _finding(dynamic, "global_dim")[0].status == ProofStatus.UNKNOWN
    assert not any(
        finding.rule == "global_dim" for finding in _validation_failures(dynamic, auto=True)
    )

    invalid = replace(dynamic, global_dims=(0, 64))
    assert any(finding.rule == "global_dim" for finding in _validation_failures(invalid, auto=True))


def test_shared_validator_packed_and_interleave_rules():
    packed = _make_spec(
        descriptor_dtype="float4_e2m1fn",
        descriptor_bits=4,
        effective_bytes=1,
        packed_kind="16u4_align16",
        global_dims=(128, 64),
        global_strides=(64,),
        box_dims=(128, 4),
        swizzle=3,
    )
    assert _validation_failures(packed, auto=False) == []
    assert _finding(replace(packed, global_dims=(64, 64)), "packed_shape")[0].status == (
        ProofStatus.DISPROVEN
    )
    assert _finding(replace(packed, box_dims=(64, 4)), "packed_box")[0].status == (
        ProofStatus.DISPROVEN
    )
    assert _finding(replace(packed, swizzle=2), "packed_swizzle")[0].status == (
        ProofStatus.DISPROVEN
    )
    assert _finding(replace(packed, direction="s2g", mbar=None), "packed_direction")[0].status == (
        ProofStatus.DISPROVEN
    )

    interleaved = _make_spec(
        global_dims=(16, 8, 4),
        global_strides=(32, 256),
        box_dims=(8, 4, 2),
        element_strides=(1, 1, 1),
        coordinates=(0, 0, 0),
        interleave=2,
        swizzle=1,
    )
    assert _validation_failures(interleaved, auto=False) == []
    assert _finding(replace(interleaved, swizzle=2), "interleave_swizzle")[0].status == (
        ProofStatus.DISPROVEN
    )


def _build_selector_gather_gpu_kernel(dtype="float16"):
    rows = 256
    cols = 64
    shared_bytes = 4 * cols * tvm.DataType(dtype).bits // 8

    # fmt: off
    @T.prim_func
    def kernel(
        A_ptr: T.handle,
        B_ptr: T.handle,
        flag: T.int32,
        Out_ptr: T.handle,
    ):
        A = T.match_buffer(A_ptr, (rows, cols), dtype)
        B = T.match_buffer(B_ptr, (rows, cols), dtype)
        Out = T.match_buffer(Out_ptr, (4, cols), dtype)
        T.device_entry()
        T.cta_id([1])
        tid = T.thread_id([128])
        dyn = T.alloc_buffer((shared_bytes + 64,), "uint8", scope="shared.dyn")
        T.attr({"tirx.dyn_smem_bytes": shared_bytes + 64})
        A_smem = T.decl_buffer(
            (4, cols), dtype, dyn.data, layout=T.TileLayout(T.S[4, cols])
        )
        mbar = T.decl_buffer((1,), "uint64", dyn.data, elem_offset=shared_bytes // 8)
        mbar_ptr = T.meta_var(mbar.ptr_to([0]))
        if tid == 0:
            T.ptx.mbarrier.init.shared.b64(mbar_ptr, T.uint32(1))
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()
        if tid == 0:
            Tx.copy_async(
                A_smem[:, :],
                A[0:1, :],
                dispatch="tma_explicit",
                mbar=mbar_ptr,
                gather4=[7, 3, 19, 5],
                src_selector=[(flag != 0, B)],
            )
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(mbar_ptr, T.uint32(shared_bytes))
        T.cuda.mbarrier_wait(mbar_ptr, 0)
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()
        Tx.cta.copy(Out[:, :], A_smem[:, :])
        # fmt: on

    return kernel


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
def test_explicit_gather_selector_gpu_roundtrip():
    dtype = "float16"
    kernel = _build_selector_gather_gpu_kernel(dtype)
    executable = _compile_module(kernel, arch=env.cuda_arch())
    dev = tvm.cuda(0)
    rng = np.random.default_rng(0)
    a_np = rng.standard_normal((256, 64)).astype(dtype)
    b_np = rng.standard_normal((256, 64)).astype(dtype)
    indices = np.array([7, 3, 19, 5])

    def run():
        a = tvm.runtime.tensor(a_np, dev)
        b = tvm.runtime.tensor(b_np, dev)
        out = tvm.runtime.tensor(np.zeros((4, 64), dtype=dtype), dev)
        executable(a, b, 0, out)
        np.testing.assert_allclose(out.numpy(), a_np[indices])
        executable(a, b, 1, out)
        np.testing.assert_allclose(out.numpy(), b_np[indices])

    tvm.testing.run_with_gpu_lock(run)


if __name__ == "__main__":
    tvm.testing.main()
