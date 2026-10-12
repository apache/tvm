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
"""CUDA tensor instruction interfaces and their fixed lowering contracts."""

from functools import cache

from tvm.ir import TensorRegion
from tvm.runtime import MemoryScope
from tvm.tirx.tile.instruction import Instruction, Operand, namespace

_SCOPE_GLOBAL = MemoryScope("global")
_SCOPE_LOCAL = MemoryScope("local")
_SCOPE_SHARED = MemoryScope("shared")
_SCOPE_SHARED_DYN = MemoryScope("shared.dyn")


@cache
def _scope_tmem():
    return MemoryScope("tmem")


def _check(result):
    if isinstance(result, tuple):
        ok, reason = result
    else:
        ok, reason = result, "incompatible operands or execution scope"
    if not ok:
        raise ValueError(reason)


def _copy_project(values, attrs):
    return (
        "copy",
        [values["dst"], values["src"]],
        {k: v for k, v in values.items() if k not in {"dst", "src"}},
    )


def _async_project(values, attrs):
    _, args, extra = _copy_project(values, attrs)
    return "copy_async", args, extra


def _memory(view, sctx):
    from .copy import ld_stmatrix, vec_auto_reg, vec_forced

    dst, src = view.args
    name = view.op.name.rsplit(".", 1)[-1]
    dst_scope, src_scope = dst.source.scope(), src.source.scope()
    if name in {"ld", "ldmatrix"}:
        if dst_scope != _SCOPE_LOCAL or src_scope not in {
            _SCOPE_GLOBAL,
            _SCOPE_SHARED,
            _SCOPE_SHARED_DYN,
        }:
            raise ValueError(f"{name} requires memory -> local operands")
    elif name in {"st", "stmatrix"}:
        if src_scope != _SCOPE_LOCAL or dst_scope not in {
            _SCOPE_GLOBAL,
            _SCOPE_SHARED,
            _SCOPE_SHARED_DYN,
        }:
            raise ValueError(f"{name} requires local -> memory operands")
    elif dst_scope != _SCOPE_LOCAL or src_scope != _SCOPE_LOCAL:
        raise ValueError("register mov requires local -> local operands")
    if name in {"ldmatrix", "stmatrix"}:
        _check(ld_stmatrix._is_ldstmatrix(view, sctx))
        return ld_stmatrix._emit(view, sctx)
    bits = int(view.call.attrs.vec_bits) if hasattr(view.call.attrs, "vec_bits") else 0
    if bits:
        if bits not in (16, 32, 64, 128, 256):
            raise ValueError("vec_bits must be 16, 32, 64, 128 or 256")
        _check(
            vec_forced._is_forced_vec_copy(view, sctx, variant=f"vec_{bits}b", num_bytes=bits // 8)
        )
        return vec_forced._emit_forced_vec_copy(view, sctx, bits // 8)
    _check(vec_auto_reg._is_reg_copy(view, sctx))
    return vec_auto_reg._emit_reg(view, sctx)


def _elementwise(view, sctx):
    from .elementwise.ops import ALL_OPS
    from .elementwise.reg import emit_reg, is_reg_ewise
    from .elementwise.smem import emit_smem, is_smem_ewise

    spec = ALL_OPS[view.kind]
    if view.args[0].source.scope() == _SCOPE_LOCAL:
        _check(is_reg_ewise(spec)(view, sctx))
        return emit_reg(view, spec, sctx)
    _check(is_smem_ewise(spec)(view, sctx))
    return emit_smem(view, spec, sctx)


def _mov_project(values, attrs):
    kind = "copy" if isinstance(values["src"], TensorRegion) else "fill"
    return kind, [values["dst"], values["src"]], {}


def _mov(view, sctx):
    if view.kind == "copy":
        if any(region.source.scope() != _SCOPE_LOCAL for region in view.args):
            raise ValueError("register mov requires local -> local operands")
        view.kind = "cast"
    return _elementwise(view, sctx)


def _ldgsts(view, sctx):
    from .copy_async.ldgsts import _emit_ldgsts, _is_ldgsts

    _check(_is_ldgsts(view, sctx))
    return _emit_ldgsts(view, sctx)


def _dsmem(view, sctx):
    from .common import validate_copy_op
    from .copy_async.dsmem import copy_dsmem_impl

    _check(validate_copy_op(view, sctx))
    if any(region.source.scope() not in (_SCOPE_SHARED, _SCOPE_SHARED_DYN) for region in view.args):
        raise ValueError("cp_async_bulk requires shared -> shared operands")
    return copy_dsmem_impl(view, sctx)


def _tma(view, sctx):
    from .copy_async.tma import (
        _validate_tma_copy_op,
        copy_tma_auto_impl,
        copy_tma_explicit_impl,
    )

    name = view.op.name.rsplit(".", 1)[-1]
    dst, src = view.args
    load = name == "cp_async_bulk_tensor_load"
    if (src.source.scope() == _SCOPE_GLOBAL) != load or (
        dst.source.scope() == _SCOPE_GLOBAL
    ) == load:
        raise ValueError(f"{name}: incompatible transfer direction")
    view.options.pop("descriptor_mode", None)
    reduction = view.options.pop("reduce_op", None)
    if name == "cp_reduce_async_bulk_tensor":
        if reduction is None:
            raise ValueError("cp_reduce_async_bulk_tensor requires reduce_op")
        view.options["use_tma_reduce"] = reduction
    elif reduction is not None:
        raise ValueError("reduce_op requires cp_reduce_async_bulk_tensor")
    policy = view.options.pop("cache_policy", None)
    if policy is not None:
        if view.call.attrs.cache_hint:
            raise ValueError("cache_hint and cache_policy are mutually exclusive")
        view.options["cache_hint"] = policy
    _check(_validate_tma_copy_op(view, sctx))
    mode = view.call.attrs.descriptor_mode
    if mode == "auto":
        return copy_tma_auto_impl(view, sctx)
    if mode == "explicit":
        return copy_tma_explicit_impl(view, sctx)
    raise ValueError("descriptor_mode must be auto or explicit")


def _tc_cp(view, sctx):
    from .copy_async.tcgen05_cp import _validate_smem_tmem_copy, copy_smem_tmem_impl

    _check(_validate_smem_tmem_copy(view, sctx))
    return copy_smem_tmem_impl(view, sctx)


def _tc_ldst(view, sctx):
    from .copy.utils import _is_valid_copy
    from .copy_async.tcgen05_ldst import copy_tmem_local_impl

    _check(_is_valid_copy(view, sctx))
    dst, src = view.args
    load = view.op.name.endswith(".ld")
    if (dst.source.scope(), src.source.scope()) != (
        (_SCOPE_LOCAL, _scope_tmem()) if load else (_scope_tmem(), _SCOPE_LOCAL)
    ):
        raise ValueError("tcgen05.ld/st requires the named TMEM transfer direction")
    return copy_tmem_local_impl(view, sctx)


def _mma(view, sctx):
    from .gemm.mma_m16n8k_ import (
        _full_active_lanes,
        _no_replica,
        gemm_cuda_mma_dispatch,
    )

    _check(_full_active_lanes(view, sctx))
    _check(_no_replica(view, sctx))
    return gemm_cuda_mma_dispatch(view, sctx)


def _tc_mma(view, sctx):
    from .gemm_async.tcgen05 import gemm_async_tcgen05_impl

    return gemm_async_tcgen05_impl(view, sctx)


def make_namespace():
    def spec(
        name,
        kind,
        operands,
        schema="ScopeAttrs",
        options=None,
        project=None,
        lower=None,
        *,
        scopes=("cluster", "cta", "warpgroup", "warp", "thread"),
    ):
        return Instruction(
            "tirx.cuda.tile." + name,
            kind,
            tuple(operands),
            scopes,
            schema,
            options or {},
            project=project,
            lower=lower,
        )

    pair = [Operand("dst", "region"), Operand("src", "region")]
    memory_options = dict(vec_bits=0, cache=None, l1_evict=None, l2_evict=None, prefetch_size=None)
    result = [
        spec(n, "copy", pair, "MemoryAttrs", memory_options, _copy_project, _memory)
        for n in ("ld", "st")
    ]
    result += [
        spec(n, "copy", pair, project=_copy_project, lower=_memory)
        for n in ("ldmatrix", "stmatrix")
    ]
    result += [
        spec(
            "mov",
            "fill",
            [Operand("dst", "region"), Operand("src")],
            project=_mov_project,
            lower=_mov,
        )
    ]
    result += [
        spec(
            "cp_async",
            "copy_async",
            [*pair, Operand("predicate", "scalar", -1)],
            "AsyncAttrs",
            dict(direct=False, fill_mode="", prefetch_size=-1),
            _async_project,
            _ldgsts,
        )
    ]
    result += [
        spec(
            "cp_async_bulk",
            "copy_async",
            [*pair, Operand("mbar", "address"), Operand("remote_cta_id", "scalar")],
            project=_async_project,
            lower=_dsmem,
            scopes=("thread",),
        )
    ]
    tma_options = dict(
        descriptor_mode="auto",
        cta_group=1,
        cache_hint="",
        tma_dtype=None,
        oob=None,
        prefetch_tensormap=False,
        tensormap_l2_promotion=None,
        reduce_op=None,
    )
    for name in (
        "cp_async_bulk_tensor_load",
        "cp_async_bulk_tensor_store",
        "cp_reduce_async_bulk_tensor",
    ):
        operands = (
            pair
            + (
                [
                    Operand("mbar", "address"),
                    Operand("cta_mask", "scalar", 0),
                    Operand("mbarrier_addr", "scalar", False),
                    Operand("gather4", "coordinates", None),
                    Operand("src_selector", "selectors", None),
                ]
                if name.endswith("_load")
                else []
            )
            + [Operand("cache_policy", "scalar", None)]
        )
        result.append(
            spec(
                name,
                "copy_async",
                operands,
                "TMAAttrs",
                tma_options,
                _async_project,
                _tma,
                scopes=("thread",),
            )
        )
    result += [
        spec(
            "tcgen05.cp",
            "copy_async",
            pair,
            "TcCopyAttrs",
            dict(shape=None, multicast=None, cta_group=1, decompress=None),
            _async_project,
            _tc_cp,
            scopes=("thread",),
        )
    ]
    result += [
        spec(
            "tcgen05." + n,
            "copy_async",
            pair,
            project=_async_project,
            lower=_tc_ldst,
            scopes=("warpgroup",),
        )
        for n in ("ld", "st")
    ]
    result += [
        spec(
            "mma_sync",
            "gemm",
            [
                Operand("D", "region"),
                Operand("A", "region"),
                Operand("B", "region"),
                Operand("C", "region"),
                Operand("transpose_A", "scalar", False),
                Operand("transpose_B", "scalar", False),
                Operand("alpha", "scalar", 1.0),
                Operand("beta", "scalar", 0.0),
            ],
            lower=_mma,
        )
    ]
    mma_options = dict(
        cta_group=1,
        mma_m=None,
        mma_n=None,
        smem_desc="hoist",
        is_AB_tf32=False,
        weight_stationary=None,
    )
    for scaled in (False, True):
        operands = [Operand("C", "region"), Operand("A", "region"), Operand("B", "region")]
        if scaled:
            operands += [Operand("SFA", "region"), Operand("SFB", "region")]
        operands += [
            Operand("transA", "scalar", False),
            Operand("transB", "scalar", False),
            Operand("accum", "scalar", False),
        ]
        # Descriptor I is a runtime operand, separate from static instruction qualifiers.
        n_base = len(operands)
        operands += [Operand("descI", "scalar", None)]

        def project(values, attrs, names=tuple(a.name for a in operands[:n_base])):
            return "gemm_async", [values[n] for n in names], {"descI": values["descI"]}

        result += [
            spec(
                "tcgen05.mma_block_scale" if scaled else "tcgen05.mma",
                "gemm_async",
                operands,
                "TcMmaAttrs",
                mma_options,
                project,
                _tc_mma,
                scopes=("thread", "warp"),
            )
        ]
    for name, kind in {
        "add": "add",
        "sub": "sub",
        "mul": "mul",
        "div": "fdiv",
        "max": "maximum",
    }.items():
        result += [
            spec(
                name,
                kind,
                [Operand("dst", "region"), Operand("lhs"), Operand("rhs")],
                "MathAttrs",
                dict(rounding_mode=None),
                lower=_elementwise,
            )
        ]
    for name, kind in {
        "cvt": "cast",
        "sqrt": "sqrt",
        "ex2": "exp2",
        "lg2": "log2",
        "compose.exp": "exp",
        "compose.silu": "silu",
    }.items():
        result += [
            spec(
                name,
                kind,
                [Operand("dst", "region"), Operand("src", "region", "@dst")],
                lower=_elementwise,
            )
        ]
    result += [
        spec(
            "fma",
            "fma",
            [Operand("dst", "region"), Operand("src"), Operand("scale"), Operand("bias")],
            lower=_elementwise,
        )
    ]
    for op in ("exp", "exp2", "log2", "sqrt"):
        name = op + "_with_scale_bias"
        result += [
            spec(
                "compose." + name,
                name,
                [
                    Operand("dst", "region"),
                    Operand("src", "region"),
                    Operand("scale", "scalar"),
                    Operand("bias"),
                ],
                lower=_elementwise,
            )
        ]
    from .patterns import apply

    result = namespace("cuda", result)
    result.apply = apply
    return result
