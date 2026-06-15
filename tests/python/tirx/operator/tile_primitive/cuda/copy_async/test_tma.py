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
# pylint: disable=invalid-name, missing-function-docstring
import functools

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.ir import PointerType, PrimType
from tvm.ir.type import TensorMapType
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.testing import env
from tvm.tirx import IntImm, StringImm, Var
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import (
    mma_atom_layout,
    mma_atom_shape,
    mma_shared_layout,
)
from tvm.tirx.exec_scope import ExecScope
from tvm.tirx.layout import S, TileLayout
from tvm.tirx.operator.tile_primitive.dispatch_context import DispatchContext
from tvm.tirx.operator.tile_primitive.ops import CopyAsync
from tvm.tirx.stmt import DeclBuffer
from tvm.tirx.stmt_functor import StmtExprVisitor

# ===========================================================================
# Helpers
# ===========================================================================


class TMACounter(StmtExprVisitor):
    """Visitor to count total TMA operations including loop iterations.

    This verifies that TMA copy operations are optimized correctly,
    resulting in minimal TMA instructions instead of multiple iterations.
    """

    def __init__(self):
        super().__init__()
        self.loop_extents = []  # Stack of loop extents
        self.total_tma_ops = 0

    def visit_for_(self, op):
        extent = op.extent
        self.loop_extents.append(extent)
        self.visit_stmt(op.body)
        self.loop_extents.pop()

    def visit_evaluate_(self, op):
        if isinstance(op.value, tvm.tirx.Call):
            if op.value.op.name in (
                "tirx.ptx.cp_async_bulk_tensor_global_to_cluster",
                "tirx.ptx.cp_async_bulk_tensor_shared_to_global",
                "tirx.ptx.cp_async_bulk_tensor_shared_to_global_reduce",
            ):
                # Multiply all enclosing loop extents
                iters = 1
                for ext in self.loop_extents:
                    iters *= ext
                self.total_tma_ops += iters


def _make_tma_call(
    g_shape,
    g_region,
    s_shape,
    s_region,
    gmem_layout,
    smem_layout,
    dtype="float16",
    direction="g2s",
    config=None,
):
    """Construct TilePrimitiveCall + DispatchContext and call copy_tma_impl.

    Returns (impl, host_init_stmts) on success, raises DispatchFail on failure.
    impl is the device-side PrimFunc, host_init_stmts is a list of Stmt
    for host-side tensor map creation.
    """
    from tvm.ir import Range
    from tvm.tirx import Var
    from tvm.tirx.cuda.operator.tile_primitive.copy_async.tma import copy_tma_impl
    from tvm.tirx.stmt import BufferRegion

    g_buf = tvm.tirx.decl_buffer(g_shape, dtype, "A", layout=gmem_layout)
    s_buf = tvm.tirx.decl_buffer(s_shape, dtype, "A_smem", scope="shared.dyn", layout=smem_layout)

    g_ranges = [Range.from_min_extent(r[0], r[1] - r[0]) for r in g_region]
    s_ranges = [Range.from_min_extent(r[0], r[1] - r[0]) for r in s_region]

    config = dict(config or {})
    if direction == "g2s":
        mbar_ptr = Var("mbar_ptr", "handle")
        config.setdefault("mbar", mbar_ptr)
        config.setdefault("cta_group", 1)
        dst_br = BufferRegion(s_buf, s_ranges)
        src_br = BufferRegion(g_buf, g_ranges)
    else:  # s2g
        config.setdefault("cta_group", 1)
        dst_br = BufferRegion(g_buf, g_ranges)
        src_br = BufferRegion(s_buf, s_ranges)

    op_call = CopyAsync(dst_br, src_br, config=config)

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_90a"})
    sctx = DispatchContext(target, ExecScope("thread"), {}, {})

    impl = copy_tma_impl(op_call, sctx)
    host_init_stmts = list(sctx.callbacks.get("host_init_stmt", []))
    return impl, host_init_stmts


def _count_tma_ops(impl):
    """Count total TMA ops in a PrimFunc (including loop multiplier)."""
    counter = TMACounter()
    counter.visit_stmt(impl.body)
    return counter.total_tma_ops


def _build_expected_host_init(dtype, encode_args):
    """Build expected host_init Bind+SeqStmt for cuTensorMapEncodeTiled.

    encode_args is a list of ints: the numeric arguments to cuTensorMapEncodeTiled
    after (tensormap, dtype_str, ndim, A_ptr). The full call is:
        runtime.cuTensorMapEncodeTiled(tensormap, dtype_str, ndim, A_ptr, *encode_args)
    where ndim = encode_args[0] and the rest are the tensor map parameters.
    """
    A_tensormap = Var("A_tensormap", PointerType(TensorMapType(), "global"))
    stack_alloca = tvm.tirx.Call(
        "handle",
        tvm.ir.Op.get("tirx.tvm_stack_alloca"),
        [StringImm("tensormap"), IntImm("int32", 1)],
    )
    A_var = Var("A", PointerType(PrimType(dtype), "global"))
    call_args = (
        [
            StringImm("runtime.cuTensorMapEncodeTiled"),
            A_tensormap,
            StringImm(dtype),
            IntImm("int32", encode_args[0]),  # ndim
            A_var,
        ]
        + [IntImm("int32", v) for v in encode_args[1:]]
    )
    encode_call = tvm.tirx.Call("int32", tvm.ir.Op.get("tirx.tvm_call_packed"), call_args)
    replace_point = tvm.tirx.Evaluate(tvm.tirx.op.tvm_kernel_replace_point())
    return tvm.tirx.SeqStmt(
        [tvm.tirx.Bind(A_tensormap, stack_alloca), tvm.tirx.Evaluate(encode_call), replace_point]
    )


def _build_expected_impl(direction, dtype, s_shape, s_layout, impl_spec):
    """Build expected impl PrimFunc.

    impl_spec is a dict with:
        loop_extents: list[int]  — e.g. [1], [2, 2], [8]
        dim: int  — TMA rank (number of coordinates, also the dim arg to PTX call)
        elem_offset_fn: callable(loop_vars) -> PrimExpr  (or None for 0)
        coord_fn: callable(loop_vars) -> list[PrimExpr]  (dim coordinate args)
        s_start: optional list[int]  — starting index for address_of (default all zeros)
    """
    from tvm.tirx.layout import ComposeLayout, SwizzleLayout

    loop_extents = impl_spec["loop_extents"]
    dim = impl_spec["dim"]
    elem_offset_fn = impl_spec.get("elem_offset_fn")
    coord_fn = impl_spec["coord_fn"]

    # Mirror _to_tile_layout() in copy_async/tma.py:
    #   ComposeLayout → tile_layout
    #   SwizzleLayout → identity TileLayout(S[shape])
    #   TileLayout    → as-is
    if isinstance(s_layout, ComposeLayout):
        buf_layout = s_layout.tile_layout
    elif isinstance(s_layout, SwizzleLayout):
        buf_layout = TileLayout(S[tuple(s_shape)])
    else:
        buf_layout = s_layout

    # Create loop vars
    n_loops = len(loop_extents)
    if n_loops == 1:
        loop_vars = [Var("loop_vars", "int32")]
    else:
        loop_vars = [Var(f"loop_vars_{i}", "int32") for i in range(n_loops)]

    # Buffer
    s_buf_ptr = Var("s_buf_w_offset_ptr", PointerType(PrimType(dtype), "shared.dyn"))
    elem_offset = elem_offset_fn(loop_vars) if elem_offset_fn else IntImm("int32", 0)
    s_buf = tvm.tirx.decl_buffer(
        s_shape,
        dtype,
        "s_buf_w_offset",
        data=s_buf_ptr,
        elem_offset=elem_offset,
        scope="shared.dyn",
        layout=buf_layout,
    )

    # Free variables
    mbar_ptr = Var("mbar_ptr", "handle")
    A_tensormap = Var("A_tensormap", PointerType(TensorMapType(), "global"))

    # address_of(s_buf[s_start...])
    s_start = impl_spec.get("s_start")
    if s_start:
        buf_indices = [IntImm("int32", v) for v in s_start]
    else:
        buf_indices = [IntImm("int32", 0)] * len(s_shape)
    addr_of = tvm.tirx.Call(
        "handle", tvm.ir.Op.get("tirx.address_of"), [tvm.tirx.BufferLoad(s_buf, buf_indices)]
    )

    # Coordinate args (must have exactly `dim` entries)
    coords = coord_fn(loop_vars)
    tensormap_addr = tvm.tirx.Call("uint64", tvm.ir.Op.get("tirx.address_of"), [A_tensormap])

    # Build PTX call based on direction
    if direction == "g2s":
        # g2c(dim, addr, mbar, tensormap, cta_mask, cta_group,
        #     cache_policy, has_cache_policy, *coords)
        ptx_op = tvm.ir.Op.get("tirx.ptx.cp_async_bulk_tensor_global_to_cluster")
        ptx_args = [
            IntImm("int32", dim),
            addr_of,
            mbar_ptr,
            tensormap_addr,
            IntImm("int32", 0),
            IntImm("int32", 1),
            IntImm("uint64", 0),
            IntImm("int32", 0),
            *coords,
        ]
    else:  # s2g
        # s2g(dim, addr, tensormap, cache_policy, has_cache_policy, *coords)
        ptx_op = tvm.ir.Op.get("tirx.ptx.cp_async_bulk_tensor_shared_to_global")
        ptx_args = [
            IntImm("int32", dim),
            addr_of,
            tensormap_addr,
            IntImm("uint64", 0),
            IntImm("int32", 0),
            *coords,
        ]

    eval_stmt = tvm.tirx.Evaluate(tvm.tirx.Call("", ptx_op, ptx_args))

    # Wrap: DeclBuffer -> nested For loops (skipped when total extent is 1,
    # matching the implementation's always-unroll single-loop emission).
    body = DeclBuffer(s_buf, eval_stmt)
    for i in range(n_loops - 1, -1, -1):
        body = tvm.tirx.For(
            loop_vars[i],
            IntImm("int32", 0),
            IntImm("int32", loop_extents[i]),
            tvm.tirx.ForKind.UNROLLED,
            body,
        )

    func = tvm.tirx.PrimFunc([], body, ret_type=None, buffer_map={})
    func = func.with_attr("global_symbol", "impl")
    # default s_tir=False is implicit; nothing to set here
    return func


def _zeros(n):
    """Return n zero IntImm coords."""
    return [IntImm("int32", 0)] * n


def _atom_rank5_elem_offset(lvs):
    """elem_offset for the structural 5D atom plan: lv * 8192."""
    return lvs[0] * 8192


def _atom_rank5_coords(lvs):
    """coord_fn for the structural 5D atom plan: [0, 0, 0, lv*2, 0]."""
    return [
        IntImm("int32", 0),
        IntImm("int32", 0),
        IntImm("int32", 0),
        lvs[0] * 2,
        IntImm("int32", 0),
    ]


def _stride_gap_elem_offset(lvs):
    """elem_offset for stride-gap-outer: lv * 4096."""
    return lvs[0] * 4096


def _stride_gap_3d_coords(lvs):
    """coord_fn for stride-gap-outer (rank=3): [0, 0, lv]."""
    return [IntImm("int32", 0), IntImm("int32", 0), lvs[0]]


def _atom_multiphase_rank5_elem_offset(lvs):
    """elem_offset for the multiphase 5D atom plan: lv * 4096."""
    return lvs[0] * 4096


def _atom_multiphase_rank5_coords(lvs):
    """coord_fn for multiphase rank-5 atom: [0, 0, lv%2*4, lv//2*2, 0]."""
    return [
        IntImm("int32", 0),
        IntImm("int32", 0),
        (lvs[0] % 2) * 4,
        (lvs[0] // 2) * 2,
        IntImm("int32", 0),
    ]


# fmt: off
# Expected parameters for each TMA test case.
# Each entry maps case_id -> (impl_spec_dict, encode_args_list).
#
# impl_spec keys:
#   loop_extents: list[int] — iteration counts for nested loops
#   dim: int — TMA rank = number of coordinates = dim arg to PTX call
#   coord_fn: callable(loop_vars) -> list[PrimExpr] — coordinate arguments (len == dim)
#   elem_offset_fn: optional callable(loop_vars) -> PrimExpr — buffer offset
#
# encode_args: list[int] — all numeric args to cuTensorMapEncodeTiled
#   [ndim, global_strides..., global_dims..., box_dims..., elem_strides...,
#    interleave, swizzle_mode, l2_promotion, oob_fill]


# ===========================================================================
# Section 2: TMA unit tests — single parametrized structural-golden driver
# ===========================================================================


def _tma_case(
    *,
    id,
    g_shape,
    g_region,
    s_shape,
    s_region,
    gmem_layout,
    smem_layout,
    dtype="float16",
    direction="g2s",
    config=None,
    impl_spec=None,
    encode_args=None,
    raises=None,
):
    """Build a pytest.param carrying a dict-form case for ``test_copy_tma_codegen``.

    Required: ``g_shape``, ``g_region``, ``s_shape``, ``s_region``, ``gmem_layout``,
    ``smem_layout``, ``id``.

    Optional:
        ``dtype``: element dtype (default ``"float16"``).
        ``direction``: ``"g2s"`` or ``"s2g"`` (default ``"g2s"``).
        ``config``: op config dict forwarded to ``copy_tma_impl`` (e.g.
            ``{"oob": "nan"}``).
        ``impl_spec``: kwargs for ``_build_expected_impl``. ``None`` skips the
            device-impl structural check.
        ``encode_args``: list for ``_build_expected_host_init``. ``None`` skips
            the host-init structural check.
        ``raises``: ``(ExceptionClass, regex_str)`` to expect instead of a
            successful dispatch.
    """
    return pytest.param(
        dict(
            g_shape=g_shape, g_region=g_region,
            s_shape=s_shape, s_region=s_region,
            gmem_layout=gmem_layout, smem_layout=smem_layout,
            dtype=dtype, direction=direction, config=config,
            impl_spec=impl_spec, encode_args=encode_args, raises=raises,
        ),
        id=id,
    )


# fmt: off
TMA_CASES = [
    # ======================================================================
    # G2S — 2D baseline (swizzle + dtype variants sharing (8, 256) shape)
    # ======================================================================
    _tma_case(
        id="g2s-2d-8x256",
        g_shape=(8, 256), g_region=((0, 8), (0, 256)),
        s_shape=(8, 256), s_region=((0, 8), (0, 256)),
        gmem_layout=TileLayout(S[8, 256]),
        smem_layout=mma_shared_layout("float16", 3, (8, 256)),
        impl_spec=dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3)),
        encode_args=[3, 64, 8, 4, 512, 128, 64, 8, 4, 1, 1, 1, 0, 3, 2, 0],
    ),
    _tma_case(
        id="g2s-2d-8x256-swizzle2",
        g_shape=(8, 256), g_region=((0, 8), (0, 256)),
        s_shape=(8, 256), s_region=((0, 8), (0, 256)),
        gmem_layout=TileLayout(S[8, 256]),
        smem_layout=mma_shared_layout("float16", 2, (8, 256)),
        impl_spec=dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3)),
        encode_args=[3, 32, 8, 8, 512, 64, 32, 8, 8, 1, 1, 1, 0, 2, 2, 0],
    ),
    _tma_case(
        id="g2s-2d-8x256-swizzle1",
        g_shape=(8, 256), g_region=((0, 8), (0, 256)),
        s_shape=(8, 256), s_region=((0, 8), (0, 256)),
        gmem_layout=TileLayout(S[8, 256]),
        smem_layout=mma_shared_layout("float16", 1, (8, 256)),
        impl_spec=dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3)),
        encode_args=[3, 16, 8, 16, 512, 32, 16, 8, 16, 1, 1, 1, 0, 1, 2, 0],
    ),
    _tma_case(
        id="g2s-2d-8x256-swizzle0",
        g_shape=(8, 256), g_region=((0, 8), (0, 256)),
        s_shape=(8, 256), s_region=((0, 8), (0, 256)),
        gmem_layout=TileLayout(S[8, 256]),
        smem_layout=mma_shared_layout("float16", 0, (8, 256)),
        impl_spec=dict(loop_extents=[1], dim=2, coord_fn=lambda lv: _zeros(2)),
        encode_args=[2, 256, 8, 512, 256, 8, 1, 1, 0, 0, 2, 0],
    ),
    _tma_case(
        id="g2s-2d-8x256-int8",
        g_shape=(8, 256), g_region=((0, 8), (0, 256)),
        s_shape=(8, 256), s_region=((0, 8), (0, 256)),
        gmem_layout=TileLayout(S[8, 256]),
        smem_layout=mma_shared_layout("int8", 3, (8, 256)),
        dtype="int8",
        impl_spec=dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3)),
        encode_args=[3, 128, 8, 2, 256, 128, 128, 8, 2, 1, 1, 1, 0, 3, 2, 0],
    ),
    _tma_case(
        id="g2s-2d-8x256-bf16",
        g_shape=(8, 256), g_region=((0, 8), (0, 256)),
        s_shape=(8, 256), s_region=((0, 8), (0, 256)),
        gmem_layout=TileLayout(S[8, 256]),
        smem_layout=mma_shared_layout("bfloat16", 3, (8, 256)),
        dtype="bfloat16",
        impl_spec=dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3)),
        encode_args=[3, 64, 8, 4, 512, 128, 64, 8, 4, 1, 1, 1, 0, 3, 2, 0],
    ),
    _tma_case(
        id="g2s-2d-8x256-fp32",
        g_shape=(8, 256), g_region=((0, 8), (0, 256)),
        s_shape=(8, 256), s_region=((0, 8), (0, 256)),
        gmem_layout=TileLayout(S[8, 256]),
        smem_layout=mma_shared_layout("float32", 3, (8, 256)),
        dtype="float32",
        impl_spec=dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3)),
        encode_args=[3, 32, 8, 8, 1024, 128, 32, 8, 8, 1, 1, 1, 0, 3, 2, 0],
    ),
    _tma_case(
        id="g2s-2d-8x256-uint8",
        g_shape=(8, 256), g_region=((0, 8), (0, 256)),
        s_shape=(8, 256), s_region=((0, 8), (0, 256)),
        gmem_layout=TileLayout(S[8, 256]),
        smem_layout=mma_shared_layout("uint8", 3, (8, 256)),
        dtype="uint8",
        impl_spec=dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3)),
        encode_args=[3, 128, 8, 2, 256, 128, 128, 8, 2, 1, 1, 1, 0, 3, 2, 0],
    ),
    _tma_case(
        id="g2s-2d-8x256-fp8e4m3",
        g_shape=(8, 256), g_region=((0, 8), (0, 256)),
        s_shape=(8, 256), s_region=((0, 8), (0, 256)),
        gmem_layout=TileLayout(S[8, 256]),
        smem_layout=mma_shared_layout("float8_e4m3fn", 3, (8, 256)),
        dtype="float8_e4m3fn",
        impl_spec=dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3)),
        encode_args=[3, 128, 8, 2, 256, 128, 128, 8, 2, 1, 1, 1, 0, 3, 2, 0],
    ),
    _tma_case(
        id="g2s-2d-8x256-fp8e5m2",
        g_shape=(8, 256), g_region=((0, 8), (0, 256)),
        s_shape=(8, 256), s_region=((0, 8), (0, 256)),
        gmem_layout=TileLayout(S[8, 256]),
        smem_layout=mma_shared_layout("float8_e5m2", 3, (8, 256)),
        dtype="float8_e5m2",
        impl_spec=dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3)),
        encode_args=[3, 128, 8, 2, 256, 128, 128, 8, 2, 1, 1, 1, 0, 3, 2, 0],
    ),
    # ======================================================================
    # G2S — 3D / partial / edge / multidim layouts
    # ======================================================================
    _tma_case(
        id="g2s-3d-shared-64x256",
        g_shape=(64, 256), g_region=((0, 64), (0, 256)),
        s_shape=(3, 64, 256), s_region=((1, 2), (0, 64), (0, 256)),
        gmem_layout=TileLayout(S[64, 256]),
        smem_layout=mma_shared_layout("float16", 3, (3, 64, 256)),
        impl_spec=dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3), s_start=[1, 0, 0]),
        encode_args=[3, 64, 64, 4, 512, 128, 64, 64, 4, 1, 1, 1, 0, 3, 2, 0],
    ),
    _tma_case(
        id="g2s-2d-32x512-atom",
        g_shape=(32, 512), g_region=((0, 32), (0, 512)),
        s_shape=(32, 512), s_region=((0, 32), (0, 512)),
        gmem_layout=TileLayout(S[32, 512]),
        smem_layout=(
            mma_atom_layout("float16", 3)
            .tile_to((16, 256), mma_atom_shape("float16", 3))
            .tile_to((32, 512), (16, 256))
        ),
        impl_spec=dict(
            loop_extents=[2], dim=5,
            coord_fn=_atom_rank5_coords, elem_offset_fn=_atom_rank5_elem_offset,
        ),
        encode_args=[5, 64, 8, 4, 4, 2, 1024, 128, 8192, 512, 64, 8, 4, 2, 2, 1, 1, 1, 1, 1, 0, 3, 2, 0],  # noqa: E501
    ),
    _tma_case(
        id="g2s-2d-partial-8192",
        g_shape=(8192, 8192), g_region=((0, 128), (0, 64)),
        s_shape=(128, 64), s_region=((0, 128), (0, 64)),
        gmem_layout=TileLayout(S[8192, 8192]),
        smem_layout=mma_shared_layout("float16", 3, (128, 64)),
        impl_spec=dict(loop_extents=[1], dim=2, coord_fn=lambda lv: _zeros(2)),
        encode_args=[2, 8192, 8192, 16384, 64, 128, 1, 1, 0, 3, 2, 0],
    ),
    _tma_case(
        id="g2s-edge-4d-shared-128x64",
        g_shape=(128, 64), g_region=((0, 128), (0, 64)),
        s_shape=(2, 2, 128, 64), s_region=((0, 1), (0, 1), (0, 128), (0, 64)),
        gmem_layout=TileLayout(S[128, 64]).canonicalize(),
        smem_layout=mma_shared_layout("float16", 3, (2, 2, 128, 64)).canonicalize(),
        impl_spec=dict(loop_extents=[1], dim=2, coord_fn=lambda lv: _zeros(2)),
        encode_args=[2, 64, 128, 128, 64, 128, 1, 1, 0, 3, 2, 0],
    ),
    _tma_case(
        id="g2s-edge-partial-offset",
        g_shape=(128, 64), g_region=((64, 64 + 24), (0, 64)),
        s_shape=(2, 2, 24, 64), s_region=((0, 1), (0, 1), (0, 24), (0, 64)),
        gmem_layout=TileLayout(S[128, 64]).canonicalize(),
        smem_layout=mma_shared_layout("float16", 3, (2, 2, 24, 64)).canonicalize(),
        impl_spec=dict(
            loop_extents=[1], dim=2,
            coord_fn=lambda lv: [IntImm("int32", 0), IntImm("int32", 64)],
        ),
        encode_args=[2, 64, 128, 128, 64, 24, 1, 1, 0, 3, 2, 0],
    ),
    _tma_case(
        id="g2s-edge-large-region",
        g_shape=(256, 64), g_region=((128, 256), (0, 64)),
        s_shape=(256, 64), s_region=((0, 128), (0, 64)),
        gmem_layout=TileLayout(S[256, 64]).canonicalize(),
        smem_layout=mma_shared_layout("float16", 3, (256, 64)).canonicalize(),
        impl_spec=dict(
            loop_extents=[1], dim=2,
            coord_fn=lambda lv: [IntImm("int32", 0), IntImm("int32", 128)],
        ),
        encode_args=[2, 64, 256, 128, 64, 128, 1, 1, 0, 3, 2, 0],
    ),
    _tma_case(
        id="g2s-partial-3d-shared-a",
        g_shape=(128, 256), g_region=((0, 32), (0, 64)),
        s_shape=(6, 128, 64), s_region=((0, 1), (0, 32), (0, 64)),
        gmem_layout=TileLayout(S[128, 256]).canonicalize(),
        smem_layout=mma_shared_layout("float16", 3, (6, 128, 64)).canonicalize(),
        impl_spec=dict(loop_extents=[1], dim=2, coord_fn=lambda lv: _zeros(2)),
        encode_args=[2, 256, 128, 512, 64, 32, 1, 1, 0, 3, 2, 0],
    ),
    _tma_case(
        id="g2s-partial-3d-shared-b",
        g_shape=(256, 512), g_region=((0, 64), (0, 64)),
        s_shape=(4, 256, 64), s_region=((1, 2), (0, 64), (0, 64)),
        gmem_layout=TileLayout(S[256, 512]).canonicalize(),
        smem_layout=mma_shared_layout("float16", 3, (4, 256, 64)).canonicalize(),
        impl_spec=dict(loop_extents=[1], dim=2, coord_fn=lambda lv: _zeros(2), s_start=[1, 0, 0]),
        encode_args=[2, 512, 256, 1024, 64, 64, 1, 1, 0, 3, 2, 0],
    ),
    _tma_case(
        id="g2s-3d-full-contiguous",
        g_shape=(4, 32, 64), g_region=((0, 4), (0, 32), (0, 64)),
        s_shape=(4, 32, 64), s_region=((0, 4), (0, 32), (0, 64)),
        gmem_layout=TileLayout(S[4, 32, 64]),
        smem_layout=TileLayout(S[4, 32, 64]),
        impl_spec=dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3)),
        encode_args=[3, 64, 32, 4, 128, 4096, 64, 32, 4, 1, 1, 1, 0, 0, 2, 0],
    ),
    _tma_case(
        id="g2s-3d-partial-contiguous",
        g_shape=(8, 16, 128), g_region=((0, 4), (0, 16), (0, 128)),
        s_shape=(4, 16, 128), s_region=((0, 4), (0, 16), (0, 128)),
        gmem_layout=TileLayout(S[8, 16, 128]),
        smem_layout=TileLayout(S[4, 16, 128]),
        impl_spec=dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3)),
        encode_args=[3, 128, 16, 8, 256, 4096, 128, 16, 4, 1, 1, 1, 0, 0, 2, 0],
    ),
    _tma_case(
        id="g2s-3d-stride-gap-outer",
        g_shape=(8, 32, 64), g_region=((0, 8), (0, 32), (0, 64)),
        s_shape=(8, 32, 64), s_region=((0, 8), (0, 32), (0, 64)),
        gmem_layout=TileLayout(S[8, 32, 64]),
        smem_layout=TileLayout(S[(8, 32, 64):(4096, 64, 1)]),
        impl_spec=dict(
            loop_extents=[8], dim=3,
            coord_fn=_stride_gap_3d_coords, elem_offset_fn=_stride_gap_elem_offset,
            s_start=[0, 0, 0],
        ),
        encode_args=[3, 64, 32, 8, 128, 4096, 64, 32, 1, 1, 1, 1, 0, 0, 2, 0],
    ),
    _tma_case(
        id="g2s-4d-reorder-a",
        g_shape=(2, 128, 8, 64), g_region=((0, 1), (0, 128), (0, 1), (0, 64)),
        s_shape=(1, 1, 128, 64), s_region=((0, 1), (0, 1), (0, 128), (0, 64)),
        gmem_layout=TileLayout(S[2, 128, 8, 64]).canonicalize(),
        smem_layout=mma_shared_layout("float16", 3, (1, 1, 128, 64)).canonicalize(),
        impl_spec=dict(loop_extents=[1], dim=4, coord_fn=lambda lv: _zeros(4), s_start=[0, 0, 0, 0]),  # noqa: E501
        encode_args=[4, 64, 128, 8, 2, 1024, 128, 131072, 64, 128, 1, 1, 1, 1, 1, 1, 0, 3, 2, 0],
    ),
    _tma_case(
        id="g2s-4d-reorder-b",
        g_shape=(4, 64, 4, 128), g_region=((0, 1), (0, 64), (0, 1), (0, 128)),
        s_shape=(1, 1, 64, 128), s_region=((0, 1), (0, 1), (0, 64), (0, 128)),
        gmem_layout=TileLayout(S[4, 64, 4, 128]).canonicalize(),
        smem_layout=mma_shared_layout("float16", 3, (1, 1, 64, 128)).canonicalize(),
        impl_spec=dict(loop_extents=[1], dim=5, coord_fn=lambda lv: _zeros(5), s_start=[0, 0, 0, 0]),  # noqa: E501
        encode_args=[5, 64, 64, 2, 4, 4, 1024, 128, 256, 65536, 64, 64, 2, 1, 1, 1, 1, 1, 1, 1, 0, 3, 2, 0],  # noqa: E501
    ),
    _tma_case(
        id="g2s-multidim-4d-a",
        g_shape=(2, 2, 128, 64), g_region=((0, 1), (0, 1), (0, 128), (0, 64)),
        s_shape=(128, 64), s_region=((0, 128), (0, 64)),
        gmem_layout=TileLayout(S[2, 2, 128, 64]).canonicalize(),
        smem_layout=mma_shared_layout("float16", 3, (128, 64)),
        impl_spec=dict(loop_extents=[1], dim=4, coord_fn=lambda lv: _zeros(4)),
        encode_args=[4, 64, 128, 2, 2, 128, 16384, 32768, 64, 128, 1, 1, 1, 1, 1, 1, 0, 3, 2, 0],
    ),
    _tma_case(
        id="g2s-multidim-4d-b",
        g_shape=(4, 64, 4, 128), g_region=((0, 1), (0, 64), (0, 1), (0, 128)),
        s_shape=(64, 128), s_region=((0, 64), (0, 128)),
        gmem_layout=TileLayout(S[4, 64, 4, 128]).canonicalize(),
        smem_layout=mma_shared_layout("float16", 3, (64, 128)),
        impl_spec=dict(loop_extents=[1], dim=5, coord_fn=lambda lv: _zeros(5)),
        encode_args=[5, 64, 64, 2, 4, 4, 1024, 128, 256, 65536, 64, 64, 2, 1, 1, 1, 1, 1, 1, 1, 0, 3, 2, 0],  # noqa: E501
    ),
    # ======================================================================
    # G2S — per-phase slices (multiphase)
    # ======================================================================
    _tma_case(
        id="g2s-multiphase-3x8x256",
        g_shape=(3, 8, 256), g_region=((0, 1), (0, 8), (0, 256)),
        s_shape=(8, 256), s_region=((0, 8), (0, 256)),
        gmem_layout=TileLayout(S[3, 8, 256]),
        smem_layout=mma_shared_layout("float16", 3, (8, 256)),
        impl_spec=dict(loop_extents=[1], dim=4, coord_fn=lambda lv: _zeros(4)),
        encode_args=[4, 64, 8, 4, 3, 512, 128, 4096, 64, 8, 4, 1, 1, 1, 1, 1, 0, 3, 2, 0],
    ),
    _tma_case(
        id="g2s-multiphase-5x64x256",
        g_shape=(5, 64, 256), g_region=((0, 1), (0, 64), (0, 256)),
        s_shape=(64, 256), s_region=((0, 64), (0, 256)),
        gmem_layout=TileLayout(S[5, 64, 256]),
        smem_layout=mma_shared_layout("float16", 3, (64, 256)),
        impl_spec=dict(loop_extents=[1], dim=4, coord_fn=lambda lv: _zeros(4)),
        encode_args=[4, 64, 64, 4, 5, 512, 128, 32768, 64, 64, 4, 1, 1, 1, 1, 1, 0, 3, 2, 0],
    ),
    _tma_case(
        id="g2s-multiphase-7x32x512-atom",
        g_shape=(7, 32, 512), g_region=((0, 1), (0, 32), (0, 512)),
        s_shape=(32, 512), s_region=((0, 32), (0, 512)),
        gmem_layout=TileLayout(S[7, 32, 512]),
        smem_layout=(
            mma_atom_layout("float16", 3)
            .tile_to((16, 256), mma_atom_shape("float16", 3))
            .tile_to((32, 512), (16, 256))
        ),
        impl_spec=dict(
            loop_extents=[4], dim=5,
            coord_fn=_atom_multiphase_rank5_coords, elem_offset_fn=_atom_multiphase_rank5_elem_offset,  # noqa: E501
        ),
        encode_args=[5, 64, 8, 8, 4, 7, 1024, 128, 8192, 32768, 64, 8, 4, 2, 1, 1, 1, 1, 1, 1, 0, 3, 2, 0],  # noqa: E501
    ),
    # ======================================================================
    # G2S — transpose-like permuted layouts
    # ======================================================================
    _tma_case(
        id="g2s-transpose-32x64",
        g_shape=(32, 64), g_region=((0, 32), (0, 64)),
        s_shape=(32, 64), s_region=((0, 32), (0, 64)),
        gmem_layout=TileLayout(S[32, 64]),
        smem_layout=TileLayout(S[(32, 64):(1, 32)]),
        impl_spec=dict(
            loop_extents=[2048], dim=2,
            coord_fn=lambda lv: [lv[0] % 64, lv[0] // 64],
            elem_offset_fn=lambda lv: lv[0] % 64 * 32 + lv[0] // 64,
        ),
        encode_args=[2, 64, 32, 128, 1, 1, 1, 1, 0, 0, 2, 0],
    ),
    _tma_case(
        id="g2s-transpose-64x32",
        g_shape=(64, 32), g_region=((0, 64), (0, 32)),
        s_shape=(64, 32), s_region=((0, 64), (0, 32)),
        gmem_layout=TileLayout(S[64, 32]),
        smem_layout=TileLayout(S[(64, 32):(1, 64)]),
        impl_spec=dict(
            loop_extents=[2048], dim=2,
            coord_fn=lambda lv: [lv[0] % 32, lv[0] // 32],
            elem_offset_fn=lambda lv: lv[0] % 32 * 64 + lv[0] // 32,
        ),
        encode_args=[2, 32, 64, 64, 1, 1, 1, 1, 0, 0, 2, 0],
    ),
    _tma_case(
        id="g2s-transpose-partial-region",
        g_shape=(128, 64), g_region=((0, 64), (0, 64)),
        s_shape=(64, 64), s_region=((0, 64), (0, 64)),
        gmem_layout=TileLayout(S[128, 64]),
        smem_layout=TileLayout(S[(64, 64):(1, 64)]),
        impl_spec=dict(
            loop_extents=[4096], dim=2,
            coord_fn=lambda lv: [lv[0] % 64, lv[0] // 64],
            elem_offset_fn=lambda lv: lv[0] % 64 * 64 + lv[0] // 64,
        ),
        encode_args=[2, 64, 128, 128, 1, 1, 1, 1, 0, 0, 2, 0],
    ),
    _tma_case(
        id="g2s-transpose-partial-offset",
        g_shape=(128, 64), g_region=((64, 128), (0, 32)),
        s_shape=(64, 32), s_region=((0, 64), (0, 32)),
        gmem_layout=TileLayout(S[128, 64]),
        smem_layout=TileLayout(S[(64, 32):(1, 64)]),
        impl_spec=dict(
            loop_extents=[2048], dim=2,
            coord_fn=lambda lv: [lv[0] % 32, lv[0] // 32 + 64],
            elem_offset_fn=lambda lv: lv[0] % 32 * 64 + lv[0] // 32,
        ),
        encode_args=[2, 64, 128, 128, 1, 1, 1, 1, 0, 0, 2, 0],
    ),
    # ======================================================================
    # G2S — non-prefix compact (4D gmem collapses to one TMA tile)
    # ======================================================================
    _tma_case(
        id="g2s-non-prefix-compact-elides",
        g_shape=(16, 16, 128, 128), g_region=((3, 4), (4, 5), (0, 128), (0, 128)),
        s_shape=(128, 128), s_region=((0, 128), (0, 128)),
        gmem_layout=TileLayout(S[(16, 16, 128, 128):(1024 * 128, 128, 1024, 1)]),
        smem_layout=TileLayout(S[128, 128]),
        impl_spec=dict(
            loop_extents=[1], dim=4,
            coord_fn=lambda lv: [
                IntImm("int32", 0), IntImm("int32", 0),
                IntImm("int32", 4), IntImm("int32", 3),
            ],
        ),
        encode_args=[4, 128, 128, 16, 16, 2048, 256, 262144, 128, 128, 1, 1, 1, 1, 1, 1, 0, 0, 2, 0],  # noqa: E501
    ),
    # ======================================================================
    # G2S — oob contract (config={"oob": ...}); fill kind is encoded in
    # encode_args[-1]. ``None`` and ``"zero"`` both map to fill_kind=0.
    # ======================================================================
    _tma_case(
        id="g2s-oob-zero",
        g_shape=(128, 64), g_region=((120, 136), (0, 64)),
        s_shape=(16, 64), s_region=((0, 16), (0, 64)),
        gmem_layout=TileLayout(S[128, 64]),
        smem_layout=mma_shared_layout("float16", 3, (16, 64)),
        config={"oob": "zero"},
        impl_spec=dict(
            loop_extents=[1], dim=2,
            coord_fn=lambda lv: [IntImm("int32", 0), IntImm("int32", 120)],
        ),
        encode_args=[2, 64, 128, 128, 64, 16, 1, 1, 0, 3, 2, 0],
    ),
    _tma_case(
        id="g2s-oob-nan",
        g_shape=(128, 64), g_region=((120, 136), (0, 64)),
        s_shape=(16, 64), s_region=((0, 16), (0, 64)),
        gmem_layout=TileLayout(S[128, 64]),
        smem_layout=mma_shared_layout("float16", 3, (16, 64)),
        config={"oob": "nan"},
        impl_spec=dict(
            loop_extents=[1], dim=2,
            coord_fn=lambda lv: [IntImm("int32", 0), IntImm("int32", 120)],
        ),
        encode_args=[2, 64, 128, 128, 64, 16, 1, 1, 0, 3, 2, 1],
    ),
    # ======================================================================
    # G2S — flash_attention4 Q/K/V regression baselines
    # Representative config: batch=1, seq_len=2048, num_qo_heads=32,
    # num_kv_heads=8, head_dim=128 → GQA_RATIO=4, SEQ_Q_PER_TILE=32,
    # BLK_M=BLK_N=128, SMEM_PIPE_DEPTH_Q=2, SMEM_PIPE_DEPTH_KV=3. Each case
    # lowers to exactly one cp_async_bulk_tensor; structural golden locks
    # rank / shape / coord / box.
    # ======================================================================
    _tma_case(
        id="g2s-fa4-q",
        g_shape=(1, 2048, 32, 128), g_region=((0, 1), (0, 32), (0, 4), (0, 128)),
        s_shape=(2, 128, 128), s_region=((0, 1), (0, 128), (0, 128)),
        gmem_layout=TileLayout(S[1, 2048, 32, 128]),
        smem_layout=mma_shared_layout("float16", 3, (2, 128, 128)),
        impl_spec=dict(loop_extents=[1], dim=5, coord_fn=lambda lv: _zeros(5)),
        encode_args=[5, 64, 32, 2048, 2, 1, 256, 8192, 128, 0, 64, 4, 32, 2, 1, 1, 1, 1, 1, 1, 0, 3, 2, 0],  # noqa: E501
    ),
    _tma_case(
        id="g2s-fa4-k",
        g_shape=(1, 2048, 8, 128), g_region=((0, 1), (0, 128), (0, 1), (0, 128)),
        s_shape=(3, 128, 128), s_region=((0, 1), (0, 128), (0, 128)),
        gmem_layout=TileLayout(S[1, 2048, 8, 128]),
        smem_layout=mma_shared_layout("float16", 3, (3, 128, 128)),
        impl_spec=dict(loop_extents=[1], dim=5, coord_fn=lambda lv: _zeros(5)),
        encode_args=[5, 64, 2048, 2, 8, 1, 2048, 128, 256, 0, 64, 128, 2, 1, 1, 1, 1, 1, 1, 1, 0, 3, 2, 0],  # noqa: E501
    ),
    _tma_case(
        id="g2s-fa4-v",
        g_shape=(1, 2048, 8, 128), g_region=((0, 1), (0, 128), (0, 1), (0, 128)),
        s_shape=(3, 128, 128), s_region=((0, 1), (0, 128), (0, 128)),
        gmem_layout=TileLayout(S[1, 2048, 8, 128]),
        smem_layout=mma_shared_layout("float16", 3, (3, 128, 128)),
        impl_spec=dict(loop_extents=[1], dim=5, coord_fn=lambda lv: _zeros(5)),
        encode_args=[5, 64, 2048, 2, 8, 1, 2048, 128, 256, 0, 64, 128, 2, 1, 1, 1, 1, 1, 1, 1, 0, 3, 2, 0],  # noqa: E501
    ),
    # ======================================================================
    # S2G — per-phase slices (swizzle + dtype variants)
    # ======================================================================
    _tma_case(
        id="s2g-multiphase-3x8x256",
        direction="s2g",
        g_shape=(3, 8, 256), g_region=((0, 1), (0, 8), (0, 256)),
        s_shape=(8, 256), s_region=((0, 8), (0, 256)),
        gmem_layout=TileLayout(S[3, 8, 256]),
        smem_layout=mma_shared_layout("float16", 3, (8, 256)),
        impl_spec=dict(loop_extents=[1], dim=4, coord_fn=lambda lv: _zeros(4)),
        encode_args=[4, 64, 8, 4, 3, 512, 128, 4096, 64, 8, 4, 1, 1, 1, 1, 1, 0, 3, 2, 0],
    ),
    _tma_case(
        id="s2g-multiphase-5x64x256",
        direction="s2g",
        g_shape=(5, 64, 256), g_region=((0, 1), (0, 64), (0, 256)),
        s_shape=(64, 256), s_region=((0, 64), (0, 256)),
        gmem_layout=TileLayout(S[5, 64, 256]),
        smem_layout=mma_shared_layout("float16", 3, (64, 256)),
        impl_spec=dict(loop_extents=[1], dim=4, coord_fn=lambda lv: _zeros(4)),
        encode_args=[4, 64, 64, 4, 5, 512, 128, 32768, 64, 64, 4, 1, 1, 1, 1, 1, 0, 3, 2, 0],
    ),
    _tma_case(
        id="s2g-multiphase-7x32x512-atom",
        direction="s2g",
        g_shape=(7, 32, 512), g_region=((0, 1), (0, 32), (0, 512)),
        s_shape=(32, 512), s_region=((0, 32), (0, 512)),
        gmem_layout=TileLayout(S[7, 32, 512]),
        smem_layout=(
            mma_atom_layout("float16", 3)
            .tile_to((16, 256), mma_atom_shape("float16", 3))
            .tile_to((32, 512), (16, 256))
        ),
        impl_spec=dict(
            loop_extents=[4], dim=5,
            coord_fn=_atom_multiphase_rank5_coords, elem_offset_fn=_atom_multiphase_rank5_elem_offset,  # noqa: E501
        ),
        encode_args=[5, 64, 8, 8, 4, 7, 1024, 128, 8192, 32768, 64, 8, 4, 2, 1, 1, 1, 1, 1, 1, 0, 3, 2, 0],  # noqa: E501
    ),
    _tma_case(
        id="s2g-multiphase-3x8x256-swizzle2",
        direction="s2g",
        g_shape=(3, 8, 256), g_region=((0, 1), (0, 8), (0, 256)),
        s_shape=(8, 256), s_region=((0, 8), (0, 256)),
        gmem_layout=TileLayout(S[3, 8, 256]),
        smem_layout=mma_shared_layout("float16", 2, (8, 256)),
        impl_spec=dict(loop_extents=[1], dim=4, coord_fn=lambda lv: _zeros(4)),
        encode_args=[4, 32, 8, 8, 3, 512, 64, 4096, 32, 8, 8, 1, 1, 1, 1, 1, 0, 2, 2, 0],
    ),
    _tma_case(
        id="s2g-multiphase-3x8x256-swizzle0",
        direction="s2g",
        g_shape=(3, 8, 256), g_region=((0, 1), (0, 8), (0, 256)),
        s_shape=(8, 256), s_region=((0, 8), (0, 256)),
        gmem_layout=TileLayout(S[3, 8, 256]),
        smem_layout=mma_shared_layout("float16", 0, (8, 256)),
        impl_spec=dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3)),
        encode_args=[3, 256, 8, 3, 512, 4096, 256, 8, 1, 1, 1, 1, 0, 0, 2, 0],
    ),
    _tma_case(
        id="s2g-multiphase-3x8x256-int8",
        direction="s2g",
        g_shape=(3, 8, 256), g_region=((0, 1), (0, 8), (0, 256)),
        s_shape=(8, 256), s_region=((0, 8), (0, 256)),
        gmem_layout=TileLayout(S[3, 8, 256]),
        smem_layout=mma_shared_layout("int8", 3, (8, 256)),
        dtype="int8",
        impl_spec=dict(loop_extents=[1], dim=4, coord_fn=lambda lv: _zeros(4)),
        encode_args=[4, 128, 8, 2, 3, 256, 128, 2048, 128, 8, 2, 1, 1, 1, 1, 1, 0, 3, 2, 0],
    ),
    _tma_case(
        id="s2g-multiphase-3x8x256-fp32",
        direction="s2g",
        g_shape=(3, 8, 256), g_region=((0, 1), (0, 8), (0, 256)),
        s_shape=(8, 256), s_region=((0, 8), (0, 256)),
        gmem_layout=TileLayout(S[3, 8, 256]),
        smem_layout=mma_shared_layout("float32", 3, (8, 256)),
        dtype="float32",
        impl_spec=dict(loop_extents=[1], dim=4, coord_fn=lambda lv: _zeros(4)),
        encode_args=[4, 32, 8, 8, 3, 1024, 128, 8192, 32, 8, 8, 1, 1, 1, 1, 1, 0, 3, 2, 0],
    ),
    # ======================================================================
    # S2G — retain multi-dim coords without linear-carry (bf16, custom layout)
    # ======================================================================
    _tma_case(
        id="s2g-keeps-multidim-coords",
        direction="s2g",
        g_shape=(1024, 4, 1024), g_region=((128, 128 + 128), (1, 1 + 1), (32, 32 + 32)),
        s_shape=(128, 32), s_region=((0, 128), (0, 32)),
        gmem_layout=TileLayout(S[(1024, 4, 1024):(4 * 1024, 1024, 1)]),
        smem_layout=TileLayout(S[(128, 32):(32, 1)]),
        dtype="bfloat16",
        impl_spec=dict(
            loop_extents=[1], dim=3,
            coord_fn=lambda lv: [
                IntImm("int32", 32),
                IntImm("int32", 128),
                IntImm("int32", 1),
            ],
        ),
    ),
    # ======================================================================
    # S2G — oob contract variants over the same (2, 128, 64) shape. ``None``
    # and ``"zero"`` map to fill_kind=0; ``"nan"`` maps to fill_kind=1. The
    # descriptor geometry is identical across the three variants.
    # ======================================================================
    _tma_case(
        id="s2g-oob-none",
        direction="s2g",
        g_shape=(2, 128, 64), g_region=((0, 1), (0, 128), (0, 64)),
        s_shape=(128, 64), s_region=((0, 128), (0, 64)),
        gmem_layout=TileLayout(S[(2, 128, 64)]),
        smem_layout=mma_shared_layout("float16", 3, (128, 64)),
        config=None,
        impl_spec=dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3)),
        encode_args=[3, 64, 128, 2, 128, 16384, 64, 128, 1, 1, 1, 1, 0, 3, 2, 0],
    ),
    _tma_case(
        id="s2g-oob-zero",
        direction="s2g",
        g_shape=(2, 128, 64), g_region=((0, 1), (0, 128), (0, 64)),
        s_shape=(128, 64), s_region=((0, 128), (0, 64)),
        gmem_layout=TileLayout(S[(2, 128, 64)]),
        smem_layout=mma_shared_layout("float16", 3, (128, 64)),
        config={"oob": "zero"},
        impl_spec=dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3)),
        encode_args=[3, 64, 128, 2, 128, 16384, 64, 128, 1, 1, 1, 1, 0, 3, 2, 0],
    ),
    _tma_case(
        id="s2g-oob-nan",
        direction="s2g",
        g_shape=(2, 128, 64), g_region=((0, 1), (0, 128), (0, 64)),
        s_shape=(128, 64), s_region=((0, 128), (0, 64)),
        gmem_layout=TileLayout(S[(2, 128, 64)]),
        smem_layout=mma_shared_layout("float16", 3, (128, 64)),
        config={"oob": "nan"},
        impl_spec=dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3)),
        encode_args=[3, 64, 128, 2, 128, 16384, 64, 128, 1, 1, 1, 1, 0, 3, 2, 1],
    ),
    # ======================================================================
    # Rejection cases — oob contract validation
    # ======================================================================
    _tma_case(
        id="reject-unknown-oob",
        direction="s2g",
        g_shape=(3, 8, 256), g_region=((0, 1), (0, 8), (0, 256)),
        s_shape=(8, 256), s_region=((0, 8), (0, 256)),
        gmem_layout=TileLayout(S[3, 8, 256]),
        smem_layout=mma_shared_layout("float16", 3, (8, 256)),
        config={"oob": "bogus"},
        raises=(Exception, "Unsupported TMA oob mode"),
    ),
    _tma_case(
        id="reject-g2s-nan-on-non-float",
        g_shape=(128, 64), g_region=((120, 136), (0, 64)),
        s_shape=(16, 64), s_region=((0, 16), (0, 64)),
        gmem_layout=TileLayout(S[128, 64]),
        smem_layout=TileLayout(S[16, 64]),
        dtype="int8",
        config={"oob": "nan"},
        raises=(Exception, "requires a floating-point dtype"),
    ),
    _tma_case(
        id="reject-s2g-nan-on-non-float",
        direction="s2g",
        g_shape=(2, 128, 64), g_region=((0, 1), (0, 128), (0, 64)),
        s_shape=(128, 64), s_region=((0, 128), (0, 64)),
        gmem_layout=TileLayout(S[2, 128, 64]),
        smem_layout=TileLayout(S[128, 64]),
        dtype="int8",
        config={"oob": "nan"},
        raises=(Exception, "requires a floating-point dtype"),
    ),
]
# fmt: on


@pytest.mark.parametrize("case", TMA_CASES)
def test_copy_tma_codegen(case):
    """Unified structural-golden driver for every TMA unit test case.

    See ``_tma_case`` for the dict-form input. When ``raises`` is set, the
    test expects ``_make_tma_call`` to raise; otherwise it compares the
    emitted device impl and host tensormap-init against the inlined
    ``impl_spec`` / ``encode_args`` goldens.
    """
    call_kwargs = dict(
        g_shape=case["g_shape"],
        g_region=case["g_region"],
        s_shape=case["s_shape"],
        s_region=case["s_region"],
        gmem_layout=case["gmem_layout"],
        smem_layout=case["smem_layout"],
        dtype=case["dtype"],
        direction=case["direction"],
        config=case["config"],
    )
    if case["raises"] is not None:
        exc, match = case["raises"]
        with pytest.raises(exc, match=match):
            _make_tma_call(**call_kwargs)
        return

    impl, host_init_stmts = _make_tma_call(**call_kwargs)
    if case["impl_spec"] is not None:
        expected_impl = _build_expected_impl(
            case["direction"],
            case["dtype"],
            case["s_shape"],
            case["smem_layout"],
            case["impl_spec"],
        )
        tvm.ir.assert_structural_equal(impl, expected_impl, map_free_vars=True)
    if case["encode_args"] is not None:
        expected_host = _build_expected_host_init(case["dtype"], case["encode_args"])
        assert len(host_init_stmts) == 1
        tvm.ir.assert_structural_equal(host_init_stmts[0], expected_host, map_free_vars=True)


# Section 3: TMA special cases (symbolic dimension, buffer view)
# ===========================================================================


@pytest.mark.cuda
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
@pytest.mark.parametrize("swizzle_len", [3])
@pytest.mark.parametrize("dtype", ["float16"])
def test_copy_tma_symbolic_dimension(dtype, swizzle_len):
    """Test TMA copy with symbolic dimension in global buffer (like hgemm pattern).

    This tests the pattern:
        Tx.copy_async(A_smem[ks, :, :], A[m_st : m_st + BLK_M, k_start : k_start + BLK_K], **tma_copy)  # noqa: E501

    Where M is a symbolic dimension in the global buffer.
    """  # noqa: E501
    # Fixed dimensions
    K = 256
    BLK_M = 64
    BLK_K = 64
    SMEM_PIPE_DEPTH = 2
    M_CONCRETE = 128  # Concrete value for testing
    thread_cnt = 128

    dev = tvm.cuda(0)

    # Shared memory layout with swizzle
    shared_layout = T.ComposeLayout(
        T.SwizzleLayout(3, swizzle_len, 3, swizzle_inner=True),
        T.TileLayout(T.S[(SMEM_PIPE_DEPTH, BLK_M, BLK_K) : (BLK_M * BLK_K, BLK_K, 1)]),
    )

    # Compute bytes for mbarrier
    smem_bytes = SMEM_PIPE_DEPTH * BLK_M * BLK_K * tvm.DataType(dtype).bits // 8
    copy_bytes = BLK_M * BLK_K * tvm.DataType(dtype).bits // 8

    # fmt: off
    @T.prim_func
    def copy_async(A_ptr: T.handle, B_ptr: T.handle) -> None:
        M = T.int32()
        A = T.match_buffer(A_ptr, [M, K], dtype)
        B = T.match_buffer(B_ptr, [SMEM_PIPE_DEPTH, BLK_M, BLK_K], dtype)

        T.device_entry()
        cta_id = T.cta_id([1])
        tid = T.thread_id([thread_cnt])
        dyn = T.alloc_buffer([smem_bytes + 64], "uint8", scope="shared.dyn")
        A_smem = T.decl_buffer(
            [SMEM_PIPE_DEPTH, BLK_M, BLK_K], dtype, dyn.data, elem_offset=0, layout=shared_layout
        )
        mbarrier = T.decl_buffer([1], "uint64", dyn.data, elem_offset=smem_bytes // 8)
        mbar_ptr = T.meta_var(mbarrier.ptr_to([0]))

        if tid == 0:
            T.ptx.mbarrier.init(mbar_ptr, 1)
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.cta_sync()

                # Copy with pipeline index (like hgemm pattern)
        for ks in range(SMEM_PIPE_DEPTH):
            if tid == 0:
                Tx.copy_async(
                    A_smem[ks, :, :],
                    A[0:BLK_M, ks * BLK_K:(ks + 1) * BLK_K],
                    dispatch="tma",
                    mbar=mbar_ptr
                )
                T.ptx.mbarrier.arrive.expect_tx(mbar_ptr, copy_bytes)

            T.ptx.mbarrier.try_wait(mbar_ptr, ks % 2)

        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.cta_sync()
        for ks in range(SMEM_PIPE_DEPTH):
            Tx.cta.copy(
                B[ks, :, :],
                A_smem[ks, :, :]
            )
        # fmt: on

    np_dtype = tvm.testing.np_dtype_from_str(dtype)
    target = tvm.target.Target("cuda")

    with target:
        mod = tvm.IRModule({"main": copy_async})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

        np.random.seed(0)
        A_np = tvm.testing.generate_random_array(dtype, (M_CONCRETE, K))
        B_np = np.zeros((SMEM_PIPE_DEPTH, BLK_M, BLK_K), dtype=np_dtype)

        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np, dev)
        mod(A, B)

        # Verify: B[ks, :, :] should equal A[0:BLK_M, ks*BLK_K:(ks+1)*BLK_K]
        B_ref = np.zeros((SMEM_PIPE_DEPTH, BLK_M, BLK_K), dtype=np_dtype)
        for ks in range(SMEM_PIPE_DEPTH):
            B_ref[ks, :, :] = A_np[0:BLK_M, ks * BLK_K : (ks + 1) * BLK_K]
        np.testing.assert_allclose(B_ref, B.numpy())


@pytest.mark.cuda
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
@pytest.mark.parametrize("swizzle_len", [3])
@pytest.mark.parametrize("dtype", ["float16"])
def test_copy_tma_3d_with_view(dtype, swizzle_len):
    """Test 3D TMA copy using buffer view and swizzle layout (like flash attention pattern).

    This tests the pattern from FA4:
        Q_smem allocated as 4D: (SMEM_PIPE_DEPTH, NUM_BLK_K, BLK_M, BLK_K)
        Q_smem_3d = Q_smem.view(SMEM_PIPE_DEPTH, NUM_BLK_K, SEQ_TILE, GQA_RATIO, BLK_K)
        Tx.copy_async(Q_smem_3d[pipe_idx, blk_k_idx, :, :, :],
                      Q[batch, seq_start:seq_end, head_start:head_end, k_start:k_end], ...)
    """
    dev = tvm.cuda(0)
    smem_bytes = 2 * 2 * 128 * 64 * tvm.DataType(dtype).bits // 8
    copy_bytes_per_blk = 32 * 4 * 64 * tvm.DataType(dtype).bits // 8

    # Shared memory layout with swizzle
    shared_layout = T.ComposeLayout(
        T.SwizzleLayout(3, swizzle_len, 3, swizzle_inner=True),
        T.TileLayout(T.S[(2, 128, 128) : (128 * 128, 128, 1)]),
    )

    # fmt: off
    @T.prim_func
    def copy_async(Q_ptr: T.handle, B_ptr: T.handle) -> None:
        Q = T.match_buffer(Q_ptr, (2, 128, 8, 128), dtype)
        B = T.match_buffer(B_ptr, (32, 4, 64), dtype)

        T.device_entry()
        cta_id = T.cta_id([1])
        tid = T.thread_id([128])
        dyn = T.alloc_buffer([smem_bytes + 64], "uint8", scope="shared.dyn")
                # Allocate as 4D like FA4: (SMEM_PIPE_DEPTH, NUM_BLK_K, BLK_M, BLK_K)
        Q_smem = T.decl_buffer(
            (2, 2, 128, 64),
            dtype, dyn.data, elem_offset=0, layout=shared_layout
        )
        mbarrier = T.decl_buffer([1], "uint64", dyn.data, elem_offset=smem_bytes // 8)
        mbar_ptr = T.meta_var(mbarrier.ptr_to([0]))

                # Create 5D view for 3D copy pattern
        Q_smem_5d = Q_smem.view(2, 2, 32, 4, 64)

        if tid == 0:
            T.ptx.mbarrier.init(mbar_ptr, 1)
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.cta_sync()

        if tid == 0:
                    # 3D copy: [SEQ_Q_PER_TILE, GQA_RATIO, BLK_K]
            Tx.copy_async(
                Q_smem_5d[0, 0, :, :, :],
                Q[0, 0:32, 0:4, 0:64],
                dispatch="tma",
                mbar=mbar_ptr
            )
            T.ptx.mbarrier.arrive.expect_tx(mbar_ptr, copy_bytes_per_blk)

        T.ptx.mbarrier.try_wait(mbar_ptr, 0)

        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.cta_sync()
        Tx.cta.copy(
            B[:, :, :],
            Q_smem_5d[0, 0, :, :, :]
        )
        # fmt: on

    np_dtype = tvm.testing.np_dtype_from_str(dtype)
    target = tvm.target.Target("cuda")

    with target:
        mod = tvm.IRModule({"main": copy_async})

        # Verify that LowerTIRx generates exactly 1 TMA instruction
        lowered = tvm.tirx.transform.LowerTIRx()(mod)
        counter = TMACounter()
        counter.visit_stmt(lowered["main"].body)

        assert counter.total_tma_ops == 1, (
            f"Expected exactly 1 TMA operation, got {counter.total_tma_ops}. "
            "This indicates the 3D TMA copy with view is not generating optimal code."
        )

        # Now compile and verify correctness
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

        np.random.seed(0)
        Q_np = tvm.testing.generate_random_array(dtype, (2, 128, 8, 128))
        B_np = np.zeros((32, 4, 64), dtype=np_dtype)

        Q = tvm.runtime.tensor(Q_np, dev)
        B = tvm.runtime.tensor(B_np, dev)
        mod(Q, B)

        B_ref = np.zeros((32, 4, 64), dtype=np_dtype)
        B_ref[:, :, :] = Q_np[0, 0:32, 0:4, 0:64]
        np.testing.assert_allclose(B_ref, B.numpy())


# ===========================================================================
# Section 4: TMA GPU smoke tests (end-to-end compilation + correctness)
# ===========================================================================


@pytest.mark.cuda
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
@pytest.mark.parametrize(
    "task",
    [
        # (a) Basic 2D G2S: (8,256) full region
        pytest.param(
            (
                (8, 256),  # g_shape
                ((0, 8), (0, 256)),  # g_region
                (8, 256),  # s_shape
                ((0, 8), (0, 256)),  # s_region
                8,  # thread count per CTA
                TileLayout(S[8, 256]),  # A_layout
                TileLayout(S[8, 256]),  # B_layout
                lambda dtype: mma_shared_layout(dtype, 3, (8, 256)),
            ),
            id="g2s-2d-basic",
        ),
        # (b) 3D pipeline G2S: (3,8,256) → (8,256) per-phase
        pytest.param(
            (
                (3, 8, 256),
                None,  # multi-phase: region computed per-phase
                (8, 256),
                None,  # multi-phase
                8,
                TileLayout(S[3, 8, 256]),
                TileLayout(S[3, 8, 256]),
                lambda dtype: mma_shared_layout(dtype, 3, (8, 256)),
            ),
            id="g2s-3d-pipeline",
        ),
        # (c) 4D with unit dims: (2,2,128,64), copy (1,1,128,64) → 2D shared (128,64)
        pytest.param(
            (
                (2, 2, 128, 64),
                ((0, 1), (0, 1), (0, 128), (0, 64)),
                (128, 64),
                ((0, 128), (0, 64)),
                128,
                TileLayout(S[2, 2, 128, 64]).canonicalize(),
                TileLayout(S[2, 2, 128, 64]).canonicalize(),
                lambda dtype: mma_shared_layout(dtype, 3, (128, 64)),
            ),
            id="g2s-4d-unit-dims",
        ),
    ],
)
@pytest.mark.parametrize("dtype", ["float16"])
def test_copy_tma_gpu_smoke_g2s(task, dtype):
    """Smoke test: compile and run TMA G2S copy on GPU to verify end-to-end correctness."""
    g_shape, g_region, s_shape, s_region, thread_cnt, layoutA, layoutB, layoutS_fn = task
    dev = tvm.cuda(0)

    shared_layout = layoutS_fn(dtype)
    is_pipeline = g_region is None

    if is_pipeline:
        n = g_shape[0]
        smem_bytes = functools.reduce(lambda acc, e: acc * e, s_shape, 1)
        smem_bytes = smem_bytes * tvm.DataType(dtype).bits // 8

        r_smem = [slice(0, s) for s in s_shape]

        def r_gmem(stage):
            return [
                slice(stage, stage + 1),
                *[slice(0, g_shape[i]) for i in range(1, len(g_shape))],
            ]

        # fmt: off
        @T.prim_func
        def copy_async(A_ptr: T.handle, B_ptr: T.handle) -> None:
            A = T.match_buffer(A_ptr, g_shape, dtype, layout=layoutA)
            B = T.match_buffer(B_ptr, g_shape, dtype, layout=layoutB)

            T.device_entry()
            cta_id = T.cta_id([1])
            tid = T.thread_id([thread_cnt])
            dyn = T.alloc_buffer([smem_bytes + 8], "uint8", scope="shared.dyn")
            A_smem = T.decl_buffer(s_shape, dtype, dyn.data, elem_offset=0, layout=shared_layout)
            mbarrier = T.decl_buffer([1], "uint64", dyn.data, elem_offset=smem_bytes // 8)
            phase: T.int32

            phase = 0
            if tid == 0:
                T.ptx.mbarrier.init(mbarrier.ptr_to([0]), 1)
            T.ptx.fence.proxy_async("shared::cta")
            T.cuda.cta_sync()

            for stage in range(n):
                if tid == 0:
                    Tx.copy_async(A_smem[tuple(r_smem)], A[tuple(r_gmem(stage))], dispatch="tma", mbar=mbarrier.ptr_to([0]))  # noqa: E501
                    T.ptx.mbarrier.arrive.expect_tx(mbarrier.ptr_to([0]), smem_bytes)

                T.ptx.mbarrier.try_wait(mbarrier.ptr_to([0]), phase)
                phase = phase ^ 1

                T.ptx.fence.proxy_async("shared::cta")
                T.cuda.cta_sync()
                Tx.cta.copy(B[tuple(r_gmem(stage))], A_smem[tuple(r_smem)])
            # fmt: on

        np_dtype = tvm.testing.np_dtype_from_str(dtype)
        target = tvm.target.Target("cuda")
        with target:
            mod = tvm.IRModule({"main": copy_async})
            mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

            np.random.seed(0)
            A_np = tvm.testing.generate_random_array(dtype, g_shape)
            B_np = np.zeros(g_shape, dtype=np_dtype)

            A = tvm.runtime.tensor(A_np, dev)
            B = tvm.runtime.tensor(B_np, dev)
            mod(A, B)
            np.testing.assert_allclose(A_np, B.numpy())
    else:
        total_bytes = functools.reduce(
            lambda acc, region: acc * (region[1] - region[0]), s_region, 1
        )
        total_bytes = total_bytes * tvm.DataType(dtype).bits // 8

        smem_bytes = functools.reduce(lambda acc, e: acc * e, s_shape, 1)
        smem_bytes = smem_bytes * tvm.DataType(dtype).bits // 8

        r_smem = [slice(s_region[i][0], s_region[i][1]) for i in range(len(s_shape))]
        r_gmem = [slice(g_region[i][0], g_region[i][1]) for i in range(len(g_shape))]

        # fmt: off
        @T.prim_func
        def copy_async(A_ptr: T.handle, B_ptr: T.handle) -> None:
            A = T.match_buffer(A_ptr, g_shape, dtype, layout=layoutA)
            B = T.match_buffer(B_ptr, g_shape, dtype, layout=layoutB)

            T.device_entry()
            cta_id = T.cta_id([1])
            tid = T.thread_id([thread_cnt])
            dyn = T.alloc_buffer([smem_bytes + 64], "uint8", scope="shared.dyn")
            A_smem = T.decl_buffer(s_shape, dtype, dyn.data, elem_offset=0, layout=shared_layout)
            mbarrier = T.decl_buffer([1], "uint64", dyn.data, elem_offset=smem_bytes // 8)
            mbar_ptr = T.meta_var(mbarrier.ptr_to([0]))

            if tid == 0:
                T.ptx.mbarrier.init(mbar_ptr, 1)
            T.ptx.fence.proxy_async("shared::cta")
            T.cuda.cta_sync()

            if tid == 0:
                Tx.copy_async(A_smem[tuple(r_smem)], A[tuple(r_gmem)], dispatch="tma", mbar=mbar_ptr)  # noqa: E501
                T.ptx.mbarrier.arrive.expect_tx(mbar_ptr, total_bytes)
            T.ptx.mbarrier.try_wait(mbar_ptr, 0)
            T.cuda.cta_sync()
            Tx.cta.copy(B[tuple(r_gmem)], A_smem[tuple(r_smem)])
            # fmt: on

        np_dtype = tvm.testing.np_dtype_from_str(dtype)
        target = tvm.target.Target("cuda")
        with target:
            mod = tvm.IRModule({"main": copy_async})
            mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

            np.random.seed(0)
            A_np = tvm.testing.generate_random_array(dtype, g_shape)
            B_np = np.zeros(g_shape, dtype=np_dtype)

            A = tvm.runtime.tensor(A_np, dev)
            B = tvm.runtime.tensor(B_np, dev)
            mod(A, B)

            B_ref = np.zeros(g_shape, dtype=np_dtype)
            B_ref[tuple(r_gmem)] = A_np[tuple(r_gmem)]
            np.testing.assert_allclose(B_ref, B.numpy())


@pytest.mark.cuda
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
@pytest.mark.parametrize("dtype", ["float16"])
def test_copy_tma_gpu_smoke_s2g(dtype):
    """Smoke test: compile and run TMA S2G store on GPU."""
    g_shape = (3, 8, 256)
    s_shape = (8, 256)
    thread_cnt = 8
    n = g_shape[0]

    shared_layout = mma_shared_layout(dtype, 3, s_shape)

    smem_bytes = functools.reduce(lambda acc, e: acc * e, s_shape, 1)
    smem_bytes = smem_bytes * tvm.DataType(dtype).bits // 8

    r_smem = [slice(0, s) for s in s_shape]

    def r_gmem(stage):
        return [slice(stage, stage + 1), *[slice(0, g_shape[i]) for i in range(1, len(g_shape))]]

    layoutA = TileLayout(S[3, 8, 256])
    layoutB = TileLayout(S[3, 8, 256])

    # fmt: off
    @T.prim_func
    def copy_async(A_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, g_shape, dtype, layout=layoutA)
        B = T.match_buffer(B_ptr, g_shape, dtype, layout=layoutB)

        T.device_entry()
        cta_id = T.cta_id([1])
        tid = T.thread_id([thread_cnt])
        dyn = T.alloc_buffer([smem_bytes], "uint8", scope="shared.dyn")
        A_smem = T.decl_buffer(s_shape, dtype, dyn.data, elem_offset=0, layout=shared_layout)

        for stage in range(n):
            Tx.copy(A_smem[tuple(r_smem)], A[tuple(r_gmem(stage))])
            T.cuda.cta_sync()
            T.ptx.fence.proxy_async("shared::cta")
            if tid == 0:
                Tx.copy_async(B[tuple(r_gmem(stage))], A_smem[tuple(r_smem)], dispatch="tma")
                T.ptx.cp_async.bulk.commit_group()
                T.ptx.cp_async.bulk.wait_group()
            T.cuda.cta_sync()
        # fmt: on

    np_dtype = tvm.testing.np_dtype_from_str(dtype)
    target = tvm.target.Target("cuda")
    dev = tvm.cuda(0)

    with target:
        mod = tvm.IRModule({"main": copy_async})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

        np.random.seed(0)
        A_np = tvm.testing.generate_random_array(dtype, g_shape)
        B_np = np.zeros(g_shape, dtype=np_dtype)

        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np, dev)
        mod(A, B)

        np.testing.assert_allclose(A_np, B.numpy())


@pytest.mark.cuda
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
@pytest.mark.parametrize("dtype", ["float16"])
def test_copy_tma_dynamic_cta_mask(dtype):
    """Regression test for B00004: dynamic cta_mask expression in TMA multicast.

    Verifies that a TIR expression (depending on T.cta_id) used as cta_mask in
    copy_async compiles through the full TIRX pipeline without crashing.
    Previously, lower_tirx_scope_ids replaced scope-ID vars via Substitute,
    but Substitute didn't visit TilePrimitiveCall.config values, leaving stale var
    references that caused MakePackedAPI to fail with:
        "variables [...] are used, but are not passed in as API arguments"
    """
    CLUSTER_SIZE = 4
    CTA_GROUP = 2
    BLK_M = 64
    BLK_K = 64
    thread_cnt = 128

    smem_shape = (BLK_M, BLK_K)
    shared_layout = T.ComposeLayout(
        T.SwizzleLayout(3, 3, 3, swizzle_inner=True), T.TileLayout(T.S[smem_shape : (BLK_K, 1)])
    )
    smem_bytes = BLK_M * BLK_K * tvm.DataType(dtype).bits // 8
    copy_bytes = smem_bytes

    # fmt: off
    @T.prim_func
    def copy_async_dynamic_mask(A_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, [BLK_M, BLK_K], dtype)

        T.device_entry()
        cbx = T.cta_id_in_cluster([CLUSTER_SIZE])
        cta_id = T.cta_id([CLUSTER_SIZE])
        tid = T.thread_id([thread_cnt])

                # Dynamic cta_mask: exact expression from B00004 bug report
        cta_mask = T.meta_var(5 + 5 * cbx)
        dyn = T.alloc_buffer([smem_bytes + 64], "uint8", scope="shared.dyn")
        A_smem = T.decl_buffer(
            smem_shape, dtype, dyn.data, elem_offset=0, layout=shared_layout,
        )
        mbarrier = T.decl_buffer([1], "uint64", dyn.data, elem_offset=smem_bytes // 8)
        mbar_ptr = T.meta_var(mbarrier.ptr_to([0]))

        if tid == 0:
            T.ptx.mbarrier.init(mbar_ptr, 1)
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.cta_sync()

        if tid == 0:
            Tx.copy_async(
                A_smem[:, :],
                A[:, :],
                dispatch="tma",
                mbar=mbar_ptr,
                cta_mask=cta_mask,
                cta_group=CTA_GROUP,
            )
            T.ptx.mbarrier.arrive.expect_tx(mbar_ptr, copy_bytes)

        T.ptx.mbarrier.try_wait(mbar_ptr, 0)
        # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": copy_async_dynamic_mask})
        # This compilation crashed before the B00004 fix with:
        #   "variables [...] are used, but are not passed in as API arguments"
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

    # Verify multicast instruction was generated
    src = mod.mod.imports[0].inspect_source()
    assert "multicast" in src, "Expected multicast TMA instruction in generated code"


if __name__ == "__main__":
    tvm.testing.main()
