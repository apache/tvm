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

from tvm.ir import TensorRegion
from tvm.tirx.tensor_instruction import Instruction, Operand, namespace


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
    from .tile_primitive.copy import ld_stmatrix, vec_auto_reg, vec_forced

    dst, src = view.args
    name = view.op.name.rsplit(".", 1)[-1]
    dst_scope, src_scope = dst.source.scope(), src.source.scope()
    if name in {"ld", "ldmatrix"}:
        if dst_scope != "local" or src_scope not in {"global", "shared", "shared.dyn"}:
            raise ValueError(f"{name} requires memory -> local operands")
    elif name in {"st", "stmatrix"}:
        if src_scope != "local" or dst_scope not in {"global", "shared", "shared.dyn"}:
            raise ValueError(f"{name} requires local -> memory operands")
    elif dst_scope != "local" or src_scope != "local":
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
    from .tile_primitive.elementwise.ops import ALL_OPS
    from .tile_primitive.elementwise.reg import emit_reg, is_reg_ewise
    from .tile_primitive.elementwise.smem import emit_smem, is_smem_ewise

    spec = ALL_OPS[view.kind]
    if view.args[0].source.scope() == "local":
        _check(is_reg_ewise(spec)(view, sctx))
        return emit_reg(view, spec, sctx)
    _check(is_smem_ewise(spec)(view, sctx))
    return emit_smem(view, spec, sctx)


def _mov_project(values, attrs):
    kind = "copy" if isinstance(values["src"], TensorRegion) else "fill"
    return kind, [values["dst"], values["src"]], {}


def _mov(view, sctx):
    if view.kind == "copy":
        if any(region.source.scope() != "local" for region in view.args):
            raise ValueError("register mov requires local -> local operands")
        view.kind = "cast"
    return _elementwise(view, sctx)


def _ldgsts(view, sctx):
    from .tile_primitive.copy_async.ldgsts import _emit_ldgsts, _is_ldgsts

    _check(_is_ldgsts(view, sctx))
    return _emit_ldgsts(view, sctx)


def _dsmem(view, sctx):
    from .tile_primitive.common import validate_copy_op
    from .tile_primitive.copy_async.dsmem import copy_dsmem_impl

    _check(validate_copy_op(view, sctx))
    if not sctx.is_thread or any(
        not region.source.scope().startswith("shared") for region in view.args
    ):
        raise ValueError("cp_async_bulk requires shared -> shared operands at thread scope")
    return copy_dsmem_impl(view, sctx)


def _tma(view, sctx):
    from .tile_primitive.copy_async.tma import (
        _validate_tma_copy_op,
        copy_tma_auto_impl,
        copy_tma_explicit_impl,
    )

    if not sctx.is_thread:
        raise ValueError("TMA tensor instructions require thread scope")
    name = view.op.name.rsplit(".", 1)[-1]
    dst, src = view.args
    load = name == "cp_async_bulk_tensor_load"
    if (src.source.scope() == "global") != load or (dst.source.scope() == "global") == load:
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
    from .tile_primitive.copy_async.tcgen05_cp import _validate_smem_tmem_copy, copy_smem_tmem_impl

    if not sctx.is_thread:
        raise ValueError("tcgen05.cp requires thread scope")
    _check(_validate_smem_tmem_copy(view, sctx))
    return copy_smem_tmem_impl(view, sctx)


def _tc_ldst(view, sctx):
    from .tile_primitive.copy.utils import _is_valid_copy
    from .tile_primitive.copy_async.tcgen05_ldst import copy_tmem_local_impl

    _check(_is_valid_copy(view, sctx))
    dst, src = view.args
    load = view.op.name.endswith(".ld")
    if (dst.source.scope(), src.source.scope()) != (
        ("local", "tmem") if load else ("tmem", "local")
    ):
        raise ValueError("tcgen05.ld/st requires the named TMEM transfer direction")
    if not sctx.is_warpgroup:
        raise ValueError("tcgen05.ld/st requires warpgroup scope")
    return copy_tmem_local_impl(view, sctx)


def _mma(view, sctx):
    from .tile_primitive.gemm.mma_m16n8k_ import (
        _full_active_lanes,
        _no_replica,
        gemm_cuda_mma_dispatch,
    )

    _check(_full_active_lanes(view, sctx))
    _check(_no_replica(view, sctx))
    return gemm_cuda_mma_dispatch(view, sctx)


def _tc_mma(view, sctx):
    from .tile_primitive.gemm_async.tcgen05 import gemm_async_tcgen05_impl

    if not (sctx.is_thread or sctx.is_warp):
        raise ValueError("tcgen05.mma requires thread or warp scope")
    return gemm_async_tcgen05_impl(view, sctx)


def make_namespace():
    def spec(name, kind, operands, schema="ScopeAttrs", options=None, project=None, lower=None):
        return Instruction(
            "tirx.cuda.tile." + name,
            kind,
            tuple(operands),
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
            spec(name, "copy_async", operands, "TMAAttrs", tma_options, _async_project, _tma)
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
        )
    ]
    result += [
        spec("tcgen05." + n, "copy_async", pair, project=_async_project, lower=_tc_ldst)
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
    return namespace("cuda", result)
