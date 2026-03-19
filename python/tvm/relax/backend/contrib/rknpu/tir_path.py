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
"""Experimental RKNPU TIR-first lowering path.

This module provides a minimal proof-of-concept path that keeps RKNPU lowering
inside TVM's Relax/TIR flow instead of emitting opaque BYOC modules.

Current scope (intentional):
- static-shape float16:
  - 2D `matmul`, `add` (tensor-tensor or tensor-bias), `nn.relu`
  - NCHW/OIHW `nn.conv2d` + `nn.relu`
- extern-backed TIR PrimFuncs (stage submits)
- FuseOps + FuseTIR integration
- a TIR pass that annotates fused functions with a PC-chain candidate count
"""

import base64
import json
import os
import struct
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import tvm
from tvm import relax, tir, topi
from tvm.relax.backend.contrib.rknpu.codegen import (
    PLACEHOLDER_BIAS,
    PLACEHOLDER_INPUT,
    PLACEHOLDER_OUTPUT,
    PLACEHOLDER_WEIGHT,
    RELOC_BIAS,
    RELOC_INPUT,
    RELOC_OUTPUT,
    RELOC_WEIGHT,
    _build_relocation_table,
)
from tvm.relax.backend.contrib.rknpu.npu_core.abstract import (
    AbstractConv2DTask,
    AbstractElementwiseTask,
    AbstractMatmulTask,
)
from tvm.relax.backend.contrib.rknpu.npu_core.alignment import align_up, pad_m
from tvm.relax.backend.contrib.rknpu.npu_core.handles import TensorHandle
from tvm.relax.backend.contrib.rknpu.npu_core.regcmd_gen import (
    RegCmdGenerator,
    compute_n_tile,
    generate_ppu_task,
)
from tvm.relax.backend.contrib.rknpu.npu_core.lut_tables import (
    EXP_LE_TABLE,
    EXP_LO_TABLE,
    EXP_LUT_PARAMS,
    GELU_LE_TABLE,
    GELU_LO_TABLE,
    GELU_LUT_PARAMS,
    build_reciprocal_tables,
)
from tvm.relax.backend.contrib.rknpu.transforms import (
    DecomposeLayerNormForRKNPU,
    DecomposeSoftmaxForRKNPU,
)
from tvm.relax.op.base import call_tir
from tvm.relax.struct_info import TensorStructInfo
from tvm.relax.expr_functor import PyExprMutator, mutator
from tvm.relax.transform.legalize_ops.common import LegalizeFunc
from tvm.script import tir as T


_K_ELEMWISE = 0
_K_OUT_EWISE_FUSABLE = 4
_K_OPAQUE = 8

_SUPPORTED_RKNPU_STAGE_IDS: Set[int] = {
    1,
    2,
    3,
    4,
    5,
    6,
    8,
    9,
    10,
    11,
    12,
    13,
}

# Stage-level ABI contracts for chain-boundary prechecks.
# Contract decisions are authoritative for submit boundary splitting/chaining.
@dataclass(frozen=True)
class StageABIContract:
    stage_id: int
    stage_name: str
    rank: int
    layout_family: str
    output_class: str
    input_accept_classes: Tuple[str, ...]
    force_materialize_before_consumers: Tuple[str, ...] = ()


_STAGE_ABI_CONTRACTS: Dict[int, StageABIContract] = {
    # 2D dense producers/consumers.
    1: StageABIContract(1, "matmul", 2, "matrix2d", "dense2d", ("dense2d", "lut2d")),
    2: StageABIContract(2, "add", 2, "matrix2d", "dense2d", ("dense2d", "lut2d")),
    3: StageABIContract(3, "relu", 2, "matrix2d", "dense2d", ("dense2d", "lut2d")),
    6: StageABIContract(
        6,
        "matmul_bias_relu",
        2,
        "matrix2d",
        "dense2d",
        ("dense2d", "lut2d"),
        force_materialize_before_consumers=("add", "relu", "add_relu"),
    ),
    9: StageABIContract(9, "mul", 2, "matrix2d", "dense2d", ("dense2d", "lut2d")),
    10: StageABIContract(10, "exp", 2, "matrix2d", "lut2d", ("dense2d",)),
    11: StageABIContract(11, "matmul_bias", 2, "matrix2d", "dense2d", ("dense2d", "lut2d")),
    12: StageABIContract(12, "reciprocal", 2, "matrix2d", "lut2d", ("dense2d", "lut2d")),
    13: StageABIContract(13, "gelu", 2, "matrix2d", "lut2d", ("dense2d",)),
    # 4D conv/relu family.
    4: StageABIContract(4, "relu4d", 4, "nchw4d", "dense4d", ("dense4d",)),
    5: StageABIContract(5, "conv2d", 4, "nchw4d", "dense4d", ("dense4d",)),
    8: StageABIContract(8, "conv2d_relu", 4, "nchw4d", "dense4d", ("dense4d",)),
}

_missing_stage_contracts = _SUPPORTED_RKNPU_STAGE_IDS.difference(_STAGE_ABI_CONTRACTS)
_extra_stage_contracts = set(_STAGE_ABI_CONTRACTS).difference(_SUPPORTED_RKNPU_STAGE_IDS)
if _missing_stage_contracts or _extra_stage_contracts:
    raise AssertionError(
        "RKNPU supported-stage ABI contract table is out of sync: "
        f"missing={sorted(_missing_stage_contracts)} extra={sorted(_extra_stage_contracts)}"
    )


@dataclass(frozen=True)
class StageBoundaryInfo:
    contract: StageABIContract
    m: Optional[int] = None
    n: Optional[int] = None
    k: Optional[int] = None
    mode: Optional[int] = None


def _stage_boundary_info(
    stage: Tuple[int, List[tvm.tir.PrimExpr]]
) -> Optional[StageBoundaryInfo]:
    sid, args = stage
    contract = _STAGE_ABI_CONTRACTS.get(sid)
    if contract is None:
        return None
    if sid in (1, 6, 11):
        if len(args) >= 6:
            return StageBoundaryInfo(
                contract=contract,
                m=_try_int_imm(args[-3]),
                k=_try_int_imm(args[-2]),
                n=_try_int_imm(args[-1]),
            )
        return StageBoundaryInfo(contract=contract)
    if sid in (2, 7, 9):
        if len(args) >= 6:
            return StageBoundaryInfo(
                contract=contract,
                m=_try_int_imm(args[3]),
                n=_try_int_imm(args[4]),
                mode=_try_int_imm(args[5]),
            )
        return StageBoundaryInfo(contract=contract)
    if sid in (3, 10, 12, 13):
        if len(args) >= 4:
            return StageBoundaryInfo(
                contract=contract,
                m=_try_int_imm(args[2]),
                n=_try_int_imm(args[3]),
            )
        return StageBoundaryInfo(contract=contract)
    if sid in (4, 5, 8):
        return StageBoundaryInfo(contract=contract)
    return StageBoundaryInfo(contract=contract)


def _pc_chain_contract_compatible(
    producer: Tuple[int, List[tvm.tir.PrimExpr]],
    consumer: Tuple[int, List[tvm.tir.PrimExpr]],
) -> Tuple[bool, str]:
    """Contract precheck for submit boundaries.

    Returns:
      - ``(True, reason)`` if chaining is allowed by contract.
      - ``(False, reason)`` if chaining must be split by contract.
    """
    prod_info = _stage_boundary_info(producer)
    cons_info = _stage_boundary_info(consumer)
    if prod_info is None or cons_info is None:
        return False, "contract_unknown_stage"
    prod_contract = prod_info.contract
    cons_contract = cons_info.contract
    sid = prod_contract.stage_id
    next_sid = cons_contract.stage_id

    if prod_contract.rank != cons_contract.rank:
        return (
            False,
            f"contract_rank_mismatch_{prod_contract.rank}d_to_{cons_contract.rank}d",
        )
    if prod_contract.layout_family != cons_contract.layout_family:
        return (
            False,
            "contract_layout_family_mismatch_"
            f"{prod_contract.layout_family}_to_{cons_contract.layout_family}",
        )
    if prod_contract.output_class not in cons_contract.input_accept_classes:
        return (
            False,
            "contract_input_class_mismatch_"
            f"{prod_contract.output_class}_to_{cons_contract.stage_name}",
        )

    # Contract-level semantic constraints migrated from legacy policy.
    if cons_contract.stage_name in prod_contract.force_materialize_before_consumers:
        return False, "contract_requires_materialization_before_consumer"
    if sid == 2 and next_sid == 10:
        if prod_info.m is None or prod_info.n is None or prod_info.m * prod_info.n <= 1024:
            return False, "contract_add_to_exp_small_or_dynamic_scores"
    if sid == 2 and next_sid == 13:
        return False, "contract_add_to_gelu_real_submit_unstable"
    if sid == 13 and next_sid == 1:
        return False, "contract_gelu_to_matmul_real_submit_unstable"
    if sid == 12 and next_sid == 9:
        allow_recip_mul = os.getenv(
            "TVM_RKNPU_PC_CHAIN_ALLOW_RECIPROCAL_TO_MUL", ""
        ).lower() in ("1", "true", "yes", "on")
        if not allow_recip_mul:
            return False, "contract_reciprocal_to_mul_not_enabled"
    # High-confidence correctness policy: avoid direct matmul->reciprocal and
    # conv->relu4d chaining until contract evidence is promoted.
    if sid in (1, 6, 11) and next_sid == 12:
        return False, "contract_matmul_to_reciprocal_requires_materialization"
    if sid in (5, 8) and next_sid == 4:
        return False, "contract_conv_to_relu4d_requires_materialization"
    if sid == 4 and next_sid == 5:
        return False, "contract_relu4d_to_conv_requires_materialization"
    if sid in (1, 6, 11) and next_sid in (2, 9):
        if cons_info.mode == 0 and prod_info.k is not None and prod_info.n is not None:
            if prod_info.k < prod_info.n:
                op_name = "add" if next_sid == 2 else "mul"
                return False, f"contract_matmul_expand_to_{op_name}_tensor_unstable"
    return True, "contract_ok"


def _try_int_imm(expr) -> Optional[int]:
    if isinstance(expr, tvm.tir.IntImm):
        return int(expr.value)
    return None


def _parse_stage_id_allowlist(raw: str) -> Set[int]:
    out: Set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.add(int(token))
        except ValueError:
            continue
    return out


def _pc_chain_stage_shape_hint(stage_id: int, args: List[tvm.tir.PrimExpr]) -> str:
    if stage_id in (1, 6, 11):
        if len(args) >= 6:
            m = _try_int_imm(args[-3])
            k = _try_int_imm(args[-2])
            n = _try_int_imm(args[-1])
            return f"m={m},k={k},n={n}"
    if stage_id in (2, 7, 9):
        if len(args) >= 5:
            m = _try_int_imm(args[3])
            n = _try_int_imm(args[4])
            mode = _try_int_imm(args[5]) if len(args) >= 6 else None
            return f"m={m},n={n},mode={mode}"
    if stage_id in (3, 10, 12, 13):
        if len(args) >= 4:
            m = _try_int_imm(args[2])
            n = _try_int_imm(args[3])
            return f"m={m},n={n}"
    return "shape=unknown"


def _pc_chain_boundary_compatible(
    producer: Tuple[int, List[tvm.tir.PrimExpr]],
    consumer: Tuple[int, List[tvm.tir.PrimExpr]],
) -> Tuple[bool, str]:
    """Return (is_compatible, reason) for a submit boundary."""
    compatible, reason, _used_legacy = _pc_chain_boundary_decide(producer, consumer)
    return compatible, reason


def _pc_chain_boundary_decide(
    producer: Tuple[int, List[tvm.tir.PrimExpr]],
    consumer: Tuple[int, List[tvm.tir.PrimExpr]],
) -> Tuple[bool, str, bool]:
    """Return (is_compatible, reason, used_legacy_fallback=False)."""
    compatible, reason = _pc_chain_contract_compatible(producer, consumer)
    return compatible, reason, False


def _shape_as_ints(sinfo: relax.TensorStructInfo) -> Optional[List[int]]:
    shape = sinfo.shape
    if shape is None:
        return None
    out: List[int] = []
    for dim in shape:
        if not isinstance(dim, tir.IntImm):
            return None
        out.append(int(dim))
    return out


def _is_all_ones_const(expr: relax.Expr) -> bool:
    if not isinstance(expr, relax.Constant):
        return False
    try:
        arr = expr.data.numpy()
    except Exception:  # pylint: disable=broad-exception-caught
        return False
    return bool(arr.size > 0 and (arr == 1).all())


def _default_matmul_legalize(bb: relax.BlockBuilder, call: relax.Call) -> relax.Expr:
    return bb.call_te(topi.matmul, call.args[0], call.args[1], primfunc_name_hint="matmul")


def _default_add_legalize(bb: relax.BlockBuilder, call: relax.Call) -> relax.Expr:
    return bb.call_te(topi.add, call.args[0], call.args[1], primfunc_name_hint="add")


def _default_multiply_legalize(bb: relax.BlockBuilder, call: relax.Call) -> relax.Expr:
    return bb.call_te(topi.multiply, call.args[0], call.args[1], primfunc_name_hint="mul")


def _default_divide_legalize(bb: relax.BlockBuilder, call: relax.Call) -> relax.Expr:
    return bb.call_te(topi.divide, call.args[0], call.args[1], primfunc_name_hint="divide")


def _default_exp_legalize(bb: relax.BlockBuilder, call: relax.Call) -> relax.Expr:
    return bb.call_te(topi.exp, call.args[0], primfunc_name_hint="exp")


def _default_rsqrt_legalize(bb: relax.BlockBuilder, call: relax.Call) -> relax.Expr:
    return bb.call_te(topi.rsqrt, call.args[0], primfunc_name_hint="rsqrt")


def _default_gelu_legalize(bb: relax.BlockBuilder, call: relax.Call) -> relax.Expr:
    return bb.call_te(topi.nn.gelu, call.args[0], primfunc_name_hint="gelu")


def _default_relu_legalize(bb: relax.BlockBuilder, call: relax.Call) -> relax.Expr:
    return bb.call_te(topi.nn.relu, call.args[0], primfunc_name_hint="relu")


def _default_conv2d_legalize(bb: relax.BlockBuilder, call: relax.Call) -> relax.Expr:
    attrs = call.attrs
    return bb.call_te(
        topi.nn.conv2d,
        call.args[0],
        call.args[1],
        attrs.strides,
        attrs.padding,
        attrs.dilation,
        attrs.data_layout,
        attrs.kernel_layout,
        attrs.out_layout,
        attrs.out_dtype,
        attrs.groups,
        primfunc_name_hint="conv2d",
    )


def _normalize_pair(value) -> Optional[List[int]]:
    values = [int(v) for v in value]
    if len(values) == 1:
        return [values[0], values[0]]
    if len(values) == 2:
        return values
    return None


def _normalize_padding(value) -> Optional[List[int]]:
    values = [int(v) for v in value]
    if len(values) == 1:
        return [values[0], values[0], values[0], values[0]]
    if len(values) == 2:
        return [values[0], values[1], values[0], values[1]]
    if len(values) == 4:
        return values
    return None


def _matmul_submit_primfunc(m: int, k: int, n: int) -> tir.PrimFunc:
    # N==1 matmul often feeds scalar-like normalization steps (e.g. softmax sum/reciprocal).
    # Keep it non-fusable to avoid a known FuseTIR buffer-mapping inconsistency on
    # matmul->divide chains while preserving normal fusion for larger-N matmuls.
    op_pattern = _K_OUT_EWISE_FUSABLE if n != 1 else _K_OPAQUE

    @T.prim_func(private=True)
    def primfunc(a: T.handle, b: T.handle, c: T.handle):
        T.func_attr({"tir.noalias": True, "op_pattern": op_pattern})
        A = T.match_buffer(a, (m, k), "float16")
        B = T.match_buffer(b, (k, n), "float16")
        C = T.match_buffer(c, (m, n), "float16")
        with T.sblock("root"):
            T.reads(A[0:m, 0:k], B[0:k, 0:n])
            T.writes(C[0:m, 0:n])
            T.evaluate(
                T.call_extern(
                    "int32",
                    "rknpu_submit_matmul_stage",
                    A.data,
                    B.data,
                    C.data,
                    T.int32(m),
                    T.int32(k),
                    T.int32(n),
                )
            )

    return primfunc


def _exp_submit_primfunc(m: int, n: int) -> tir.PrimFunc:
    @T.prim_func(private=True)
    def primfunc(a: T.handle, c: T.handle):
        T.func_attr({"tir.noalias": True, "op_pattern": _K_ELEMWISE})
        A = T.match_buffer(a, (m, n), "float16")
        C = T.match_buffer(c, (m, n), "float16")
        with T.sblock("root"):
            T.reads(A[0:m, 0:n])
            T.writes(C[0:m, 0:n])
            T.evaluate(
                T.call_extern(
                    "int32",
                    "rknpu_submit_exp_stage",
                    A.data,
                    C.data,
                    T.int32(m),
                    T.int32(n),
                )
            )

    return primfunc


def _reciprocal_submit_primfunc(m: int, n: int) -> tir.PrimFunc:
    @T.prim_func(private=True)
    def primfunc(a: T.handle, c: T.handle):
        T.func_attr({"tir.noalias": True, "op_pattern": _K_ELEMWISE})
        A = T.match_buffer(a, (m, n), "float16")
        C = T.match_buffer(c, (m, n), "float16")
        with T.sblock("root"):
            T.reads(A[0:m, 0:n])
            T.writes(C[0:m, 0:n])
            T.evaluate(
                T.call_extern(
                    "int32",
                    "rknpu_submit_reciprocal_stage",
                    A.data,
                    C.data,
                    T.int32(m),
                    T.int32(n),
                )
            )

    return primfunc


def _gelu_submit_primfunc(m: int, n: int) -> tir.PrimFunc:
    @T.prim_func(private=True)
    def primfunc(a: T.handle, c: T.handle):
        # Keep GELU as its own bridge submit. Fusing it into surrounding
        # call_tir PrimFuncs reintroduces real-submit correctness drift on the
        # encoder path via local alloc_buffer handoff between submits.
        T.func_attr({"tir.noalias": True, "op_pattern": _K_OPAQUE})
        A = T.match_buffer(a, (m, n), "float16")
        C = T.match_buffer(c, (m, n), "float16")
        with T.sblock("root"):
            T.reads(A[0:m, 0:n])
            T.writes(C[0:m, 0:n])
            T.evaluate(
                T.call_extern(
                    "int32",
                    "rknpu_submit_gelu_stage",
                    A.data,
                    C.data,
                    T.int32(m),
                    T.int32(n),
                )
            )

    return primfunc


def _rsqrt_opaque_primfunc(m: int, n: int) -> tir.PrimFunc:
    @T.prim_func(private=True)
    def primfunc(a: T.handle, c: T.handle):
        T.func_attr({"tir.noalias": True, "op_pattern": _K_OPAQUE})
        A = T.match_buffer(a, (m, n), "float16")
        C = T.match_buffer(c, (m, n), "float16")
        for i, j in T.grid(m, n):
            with T.sblock("rsqrt"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(A[vi, vj])
                T.writes(C[vi, vj])
                C[vi, vj] = T.rsqrt(A[vi, vj])

    return primfunc


def _gelu_opaque_primfunc(m: int, n: int) -> tir.PrimFunc:
    sqrt_half = tir.FloatImm("float16", 0.7071067811865476)
    half = tir.FloatImm("float16", 0.5)

    @T.prim_func(private=True)
    def primfunc(a: T.handle, c: T.handle):
        T.func_attr({"tir.noalias": True, "op_pattern": _K_OPAQUE})
        A = T.match_buffer(a, (m, n), "float16")
        C = T.match_buffer(c, (m, n), "float16")
        for i, j in T.grid(m, n):
            with T.sblock("gelu"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(A[vi, vj])
                T.writes(C[vi, vj])
                erf_arg = T.Cast("float32", A[vi, vj] * sqrt_half)
                erf_val = T.erf(erf_arg)
                C[vi, vj] = A[vi, vj] * (half + T.Cast("float16", erf_val) * half)

    return primfunc


def _divide_opaque_primfunc(am: int, an: int, bm: int, bn: int, m: int, n: int) -> tir.PrimFunc:
    @T.prim_func(private=True)
    def primfunc(a: T.handle, b: T.handle, c: T.handle):
        T.func_attr({"tir.noalias": True, "op_pattern": _K_OPAQUE})
        A = T.match_buffer(a, (am, an), "float16")
        B = T.match_buffer(b, (bm, bn), "float16")
        C = T.match_buffer(c, (m, n), "float16")
        for i, j in T.grid(m, n):
            with T.sblock("divide"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(A[(0 if am == 1 else vi), (0 if an == 1 else vj)],
                        B[(0 if bm == 1 else vi), (0 if bn == 1 else vj)])
                T.writes(C[vi, vj])
                C[vi, vj] = (
                    A[(0 if am == 1 else vi), (0 if an == 1 else vj)]
                    / B[(0 if bm == 1 else vi), (0 if bn == 1 else vj)]
                )

    return primfunc


def _add_submit_primfunc(m: int, n: int, b_mode: int) -> tir.PrimFunc:
    if b_mode == 1:

        @T.prim_func(private=True)
        def primfunc(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr({"tir.noalias": True, "op_pattern": _K_ELEMWISE})
            A = T.match_buffer(a, (m, n), "float16")
            B = T.match_buffer(b, (n,), "float16")
            C = T.match_buffer(c, (m, n), "float16")
            with T.sblock("root"):
                T.reads(A[0:m, 0:n], B[0:n])
                T.writes(C[0:m, 0:n])
                T.evaluate(
                    T.call_extern(
                        "int32",
                        "rknpu_submit_add_stage",
                        A.data,
                        B.data,
                        C.data,
                        T.int32(m),
                        T.int32(n),
                        T.int32(1),
                    )
                )

    elif b_mode == 2:

        @T.prim_func(private=True)
        def primfunc(a: T.handle, b: T.handle, c: T.handle):
            # Allow output-side fusion (e.g. add -> exp) but avoid declaring this
            # as a generic elementwise producer. Full elementwise fusion on the
            # [m,1] operand previously tripped FuseTIR buffer-equality checks.
            T.func_attr({"tir.noalias": True, "op_pattern": _K_OUT_EWISE_FUSABLE})
            A = T.match_buffer(a, (m, n), "float16")
            B = T.match_buffer(b, (m, 1), "float16")
            C = T.match_buffer(c, (m, n), "float16")
            with T.sblock("root"):
                T.reads(A[0:m, 0:n], B[0:m, 0:1])
                T.writes(C[0:m, 0:n])
                T.evaluate(
                    T.call_extern(
                        "int32",
                        "rknpu_submit_add_stage",
                        A.data,
                        B.data,
                        C.data,
                        T.int32(m),
                        T.int32(n),
                        T.int32(2),
                    )
                )

    else:

        @T.prim_func(private=True)
        def primfunc(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr({"tir.noalias": True, "op_pattern": _K_ELEMWISE})
            A = T.match_buffer(a, (m, n), "float16")
            B = T.match_buffer(b, (m, n), "float16")
            C = T.match_buffer(c, (m, n), "float16")
            with T.sblock("root"):
                T.reads(A[0:m, 0:n], B[0:m, 0:n])
                T.writes(C[0:m, 0:n])
                T.evaluate(
                    T.call_extern(
                        "int32",
                        "rknpu_submit_add_stage",
                        A.data,
                        B.data,
                        C.data,
                        T.int32(m),
                        T.int32(n),
                        T.int32(0),
                    )
                )

    return primfunc


def _mul_submit_primfunc(m: int, n: int, b_mode: int) -> tir.PrimFunc:
    if b_mode == 1:

        @T.prim_func(private=True)
        def primfunc(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr({"tir.noalias": True, "op_pattern": _K_ELEMWISE})
            A = T.match_buffer(a, (m, n), "float16")
            B = T.match_buffer(b, (n,), "float16")
            C = T.match_buffer(c, (m, n), "float16")
            with T.sblock("root"):
                T.reads(A[0:m, 0:n], B[0:n])
                T.writes(C[0:m, 0:n])
                T.evaluate(
                    T.call_extern(
                        "int32",
                        "rknpu_submit_mul_stage",
                        A.data,
                        B.data,
                        C.data,
                        T.int32(m),
                        T.int32(n),
                        T.int32(1),
                    )
                )

    elif b_mode == 2:

        @T.prim_func(private=True)
        def primfunc(a: T.handle, b: T.handle, c: T.handle):
            # Same rationale as column-broadcast add: allow consumer-side fusion
            # without advertising fully generic elementwise fusibility.
            T.func_attr({"tir.noalias": True, "op_pattern": _K_OUT_EWISE_FUSABLE})
            A = T.match_buffer(a, (m, n), "float16")
            B = T.match_buffer(b, (m, 1), "float16")
            C = T.match_buffer(c, (m, n), "float16")
            with T.sblock("root"):
                T.reads(A[0:m, 0:n], B[0:m, 0:1])
                T.writes(C[0:m, 0:n])
                T.evaluate(
                    T.call_extern(
                        "int32",
                        "rknpu_submit_mul_stage",
                        A.data,
                        B.data,
                        C.data,
                        T.int32(m),
                        T.int32(n),
                        T.int32(2),
                    )
                )

    else:

        @T.prim_func(private=True)
        def primfunc(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr({"tir.noalias": True, "op_pattern": _K_ELEMWISE})
            A = T.match_buffer(a, (m, n), "float16")
            B = T.match_buffer(b, (m, n), "float16")
            C = T.match_buffer(c, (m, n), "float16")
            with T.sblock("root"):
                T.reads(A[0:m, 0:n], B[0:m, 0:n])
                T.writes(C[0:m, 0:n])
                T.evaluate(
                    T.call_extern(
                        "int32",
                        "rknpu_submit_mul_stage",
                        A.data,
                        B.data,
                        C.data,
                        T.int32(m),
                        T.int32(n),
                        T.int32(0),
                    )
                )

    return primfunc


def _mul_square_submit_primfunc(m: int, n: int) -> tir.PrimFunc:
    @T.prim_func(private=True)
    def primfunc(a: T.handle, c: T.handle):
        T.func_attr({"tir.noalias": True, "op_pattern": _K_ELEMWISE})
        A = T.match_buffer(a, (m, n), "float16")
        C = T.match_buffer(c, (m, n), "float16")
        with T.sblock("root"):
            T.reads(A[0:m, 0:n])
            T.writes(C[0:m, 0:n])
            T.evaluate(
                T.call_extern(
                    "int32",
                    "rknpu_submit_mul_stage",
                    A.data,
                    A.data,
                    C.data,
                    T.int32(m),
                    T.int32(n),
                    T.int32(0),
                )
            )

    return primfunc


def _relu_submit_primfunc(m: int, n: int) -> tir.PrimFunc:
    @T.prim_func(private=True)
    def primfunc(a: T.handle, c: T.handle):
        T.func_attr({"tir.noalias": True, "op_pattern": _K_ELEMWISE})
        A = T.match_buffer(a, (m, n), "float16")
        C = T.match_buffer(c, (m, n), "float16")
        with T.sblock("root"):
            T.reads(A[0:m, 0:n])
            T.writes(C[0:m, 0:n])
            T.evaluate(
                T.call_extern(
                    "int32",
                    "rknpu_submit_relu_stage",
                    A.data,
                    C.data,
                    T.int32(m),
                    T.int32(n),
                )
            )

    return primfunc


def _relu_submit_primfunc_4d(n: int, c: int, h: int, w: int) -> tir.PrimFunc:
    @T.prim_func(private=True)
    def primfunc(a: T.handle, c_out: T.handle):
        T.func_attr({"tir.noalias": True, "op_pattern": _K_ELEMWISE})
        A = T.match_buffer(a, (n, c, h, w), "float16")
        C_out = T.match_buffer(c_out, (n, c, h, w), "float16")
        with T.sblock("root"):
            T.reads(A[0:n, 0:c, 0:h, 0:w])
            T.writes(C_out[0:n, 0:c, 0:h, 0:w])
            T.evaluate(
                T.call_extern(
                    "int32",
                    "rknpu_submit_relu_stage_4d",
                    A.data,
                    C_out.data,
                    T.int32(n),
                    T.int32(c),
                    T.int32(h),
                    T.int32(w),
                )
            )

    return primfunc


def _conv2d_submit_primfunc(
    n: int,
    c: int,
    h: int,
    w: int,
    oc: int,
    kh: int,
    kw: int,
    oh: int,
    ow: int,
    stride_h: int,
    stride_w: int,
    pad_top: int,
    pad_left: int,
    pad_bottom: int,
    pad_right: int,
) -> tir.PrimFunc:
    @T.prim_func(private=True)
    def primfunc(data: T.handle, weight: T.handle, out: T.handle):
        T.func_attr({"tir.noalias": True, "op_pattern": _K_OUT_EWISE_FUSABLE})
        Data = T.match_buffer(data, (n, c, h, w), "float16")
        Weight = T.match_buffer(weight, (oc, c, kh, kw), "float16")
        Out = T.match_buffer(out, (n, oc, oh, ow), "float16")
        with T.sblock("root"):
            T.reads(Data[0:n, 0:c, 0:h, 0:w], Weight[0:oc, 0:c, 0:kh, 0:kw])
            T.writes(Out[0:n, 0:oc, 0:oh, 0:ow])
            T.evaluate(
                T.call_extern(
                    "int32",
                    "rknpu_submit_conv2d_stage",
                    Data.data,
                    Weight.data,
                    Out.data,
                    T.int32(n),
                    T.int32(c),
                    T.int32(h),
                    T.int32(w),
                    T.int32(oc),
                    T.int32(kh),
                    T.int32(kw),
                    T.int32(oh),
                    T.int32(ow),
                    T.int32(stride_h),
                    T.int32(stride_w),
                    T.int32(pad_top),
                    T.int32(pad_left),
                    T.int32(pad_bottom),
                    T.int32(pad_right),
                )
            )

    return primfunc


def _matmul_bias_relu_submit_primfunc(m: int, k: int, n: int) -> tir.PrimFunc:
    @T.prim_func(private=True)
    def primfunc(a: T.handle, b: T.handle, bias: T.handle, c: T.handle):
        T.func_attr({"tir.noalias": True, "op_pattern": _K_OPAQUE})
        A = T.match_buffer(a, (m, k), "float16")
        B = T.match_buffer(b, (k, n), "float16")
        Bias = T.match_buffer(bias, (n,), "float16")
        C = T.match_buffer(c, (m, n), "float16")
        with T.sblock("root"):
            T.reads(A[0:m, 0:k], B[0:k, 0:n], Bias[0:n])
            T.writes(C[0:m, 0:n])
            T.evaluate(
                T.call_extern(
                    "int32",
                    "rknpu_submit_chain_stage_v2",
                    T.int32(1),
                    T.int32(6),
                    A.data,
                    B.data,
                    Bias.data,
                    C.data,
                    T.int32(m),
                    T.int32(k),
                    T.int32(n),
                )
            )

    return primfunc


def _matmul_bias_submit_primfunc(m: int, k: int, n: int) -> tir.PrimFunc:
    @T.prim_func(private=True)
    def primfunc(a: T.handle, b: T.handle, bias: T.handle, c: T.handle):
        T.func_attr({"tir.noalias": True, "op_pattern": _K_OPAQUE})
        A = T.match_buffer(a, (m, k), "float16")
        B = T.match_buffer(b, (k, n), "float16")
        Bias = T.match_buffer(bias, (n,), "float16")
        C = T.match_buffer(c, (m, n), "float16")
        with T.sblock("root"):
            T.reads(A[0:m, 0:k], B[0:k, 0:n], Bias[0:n])
            T.writes(C[0:m, 0:n])
            T.evaluate(
                T.call_extern(
                    "int32",
                    "rknpu_submit_chain_stage_v2",
                    T.int32(1),
                    T.int32(11),
                    A.data,
                    B.data,
                    Bias.data,
                    C.data,
                    T.int32(m),
                    T.int32(k),
                    T.int32(n),
                )
            )

    return primfunc


def _conv2d_relu_submit_primfunc(
    n: int,
    c: int,
    h: int,
    w: int,
    oc: int,
    kh: int,
    kw: int,
    oh: int,
    ow: int,
    stride_h: int,
    stride_w: int,
    pad_top: int,
    pad_left: int,
    pad_bottom: int,
    pad_right: int,
) -> tir.PrimFunc:
    @T.prim_func(private=True)
    def primfunc(data: T.handle, weight: T.handle, out: T.handle):
        T.func_attr({"tir.noalias": True, "op_pattern": _K_OPAQUE})
        Data = T.match_buffer(data, (n, c, h, w), "float16")
        Weight = T.match_buffer(weight, (oc, c, kh, kw), "float16")
        Out = T.match_buffer(out, (n, oc, oh, ow), "float16")
        with T.sblock("root"):
            T.reads(Data[0:n, 0:c, 0:h, 0:w], Weight[0:oc, 0:c, 0:kh, 0:kw])
            T.writes(Out[0:n, 0:oc, 0:oh, 0:ow])
            T.evaluate(
                T.call_extern(
                    "int32",
                    "rknpu_submit_chain_stage_v2",
                    T.int32(1),
                    T.int32(8),
                    Data.data,
                    Weight.data,
                    Out.data,
                    T.int32(n),
                    T.int32(c),
                    T.int32(h),
                    T.int32(w),
                    T.int32(oc),
                    T.int32(kh),
                    T.int32(kw),
                    T.int32(oh),
                    T.int32(ow),
                    T.int32(stride_h),
                    T.int32(stride_w),
                    T.int32(pad_top),
                    T.int32(pad_left),
                    T.int32(pad_bottom),
                    T.int32(pad_right),
                )
            )

    return primfunc


def _legalize_matmul_to_rknpu_stage(bb: relax.BlockBuilder, call: relax.Call) -> relax.Expr:
    lhs_sinfo = call.args[0].struct_info
    rhs_sinfo = call.args[1].struct_info
    if not isinstance(lhs_sinfo, relax.TensorStructInfo) or not isinstance(
        rhs_sinfo, relax.TensorStructInfo
    ):
        return _default_matmul_legalize(bb, call)
    lhs_shape = _shape_as_ints(lhs_sinfo)
    rhs_shape = _shape_as_ints(rhs_sinfo)
    if (
        lhs_shape is None
        or rhs_shape is None
        or len(lhs_shape) != 2
        or len(rhs_shape) != 2
        or lhs_sinfo.dtype != "float16"
        or rhs_sinfo.dtype != "float16"
    ):
        return _default_matmul_legalize(bb, call)

    m, k = lhs_shape
    rk, n = rhs_shape
    if k != rk:
        return _default_matmul_legalize(bb, call)
    gvar = bb.add_func(_matmul_submit_primfunc(m, k, n), f"rknpu_submit_matmul_m{m}_k{k}_n{n}")
    return call_tir(gvar, [call.args[0], call.args[1]], [TensorStructInfo((m, n), "float16")])


def _legalize_add_to_rknpu_stage(bb: relax.BlockBuilder, call: relax.Call) -> relax.Expr:
    a_sinfo = call.args[0].struct_info
    b_sinfo = call.args[1].struct_info
    if not isinstance(a_sinfo, relax.TensorStructInfo) or not isinstance(
        b_sinfo, relax.TensorStructInfo
    ):
        return _default_add_legalize(bb, call)
    a_shape = _shape_as_ints(a_sinfo)
    b_shape = _shape_as_ints(b_sinfo)
    if (
        a_shape is None
        or b_shape is None
        or len(a_shape) != 2
        or a_sinfo.dtype != "float16"
        or b_sinfo.dtype != "float16"
    ):
        return _default_add_legalize(bb, call)

    m, n = a_shape
    bias_1d = len(b_shape) == 1 and b_shape[0] == n
    col_2d = len(b_shape) == 2 and b_shape[0] == m and b_shape[1] == 1
    tensor_2d = len(b_shape) == 2 and b_shape[0] == m and b_shape[1] == n
    if not (bias_1d or col_2d or tensor_2d):
        return _default_add_legalize(bb, call)

    b_mode = 1 if bias_1d else 2 if col_2d else 0
    suffix = "bias" if b_mode == 1 else "col" if b_mode == 2 else "tensor"
    gvar = bb.add_func(
        _add_submit_primfunc(m, n, b_mode),
        f"rknpu_submit_add_m{m}_n{n}_{suffix}",
    )
    return call_tir(gvar, [call.args[0], call.args[1]], [TensorStructInfo((m, n), "float16")])


def _legalize_multiply_to_rknpu_stage(bb: relax.BlockBuilder, call: relax.Call) -> relax.Expr:
    a_sinfo = call.args[0].struct_info
    b_sinfo = call.args[1].struct_info
    if not isinstance(a_sinfo, relax.TensorStructInfo) or not isinstance(
        b_sinfo, relax.TensorStructInfo
    ):
        return _default_multiply_legalize(bb, call)
    a_shape = _shape_as_ints(a_sinfo)
    b_shape = _shape_as_ints(b_sinfo)
    if (
        a_shape is None
        or b_shape is None
        or len(a_shape) != 2
        or a_sinfo.dtype != "float16"
        or b_sinfo.dtype != "float16"
    ):
        return _default_multiply_legalize(bb, call)

    m, n = a_shape
    if len(b_shape) == 2 and b_shape[0] == m and b_shape[1] == n and call.args[0].same_as(call.args[1]):
        gvar = bb.add_func(
            _mul_square_submit_primfunc(m, n),
            f"rknpu_submit_mul_square_m{m}_n{n}",
        )
        return call_tir(gvar, [call.args[0]], [TensorStructInfo((m, n), "float16")])

    bias_1d = len(b_shape) == 1 and b_shape[0] == n
    col_2d = len(b_shape) == 2 and b_shape[0] == m and b_shape[1] == 1
    tensor_2d = len(b_shape) == 2 and b_shape[0] == m and b_shape[1] == n
    if not (bias_1d or col_2d or tensor_2d):
        return _default_multiply_legalize(bb, call)

    b_mode = 1 if bias_1d else 2 if col_2d else 0
    suffix = "bias" if b_mode == 1 else "col" if b_mode == 2 else "tensor"
    gvar = bb.add_func(
        _mul_submit_primfunc(m, n, b_mode),
        f"rknpu_submit_mul_m{m}_n{n}_{suffix}",
    )
    return call_tir(gvar, [call.args[0], call.args[1]], [TensorStructInfo((m, n), "float16")])


def _legalize_relu_to_rknpu_stage(bb: relax.BlockBuilder, call: relax.Call) -> relax.Expr:
    x_sinfo = call.args[0].struct_info
    if not isinstance(x_sinfo, relax.TensorStructInfo):
        return _default_relu_legalize(bb, call)
    shape = _shape_as_ints(x_sinfo)
    if shape is None or x_sinfo.dtype != "float16":
        return _default_relu_legalize(bb, call)
    if len(shape) == 2:
        m, n = shape
        gvar = bb.add_func(_relu_submit_primfunc(m, n), f"rknpu_submit_relu_m{m}_n{n}")
        return call_tir(gvar, [call.args[0]], [TensorStructInfo((m, n), "float16")])
    if len(shape) == 4:
        n, c, h, w = shape
        gvar = bb.add_func(
            _relu_submit_primfunc_4d(n, c, h, w),
            f"rknpu_submit_relu_n{n}_c{c}_h{h}_w{w}",
        )
        return call_tir(gvar, [call.args[0]], [TensorStructInfo((n, c, h, w), "float16")])
    return _default_relu_legalize(bb, call)


def _legalize_exp_to_rknpu_stage(bb: relax.BlockBuilder, call: relax.Call) -> relax.Expr:
    sinfo = call.args[0].struct_info
    if not isinstance(sinfo, TensorStructInfo):
        return _default_exp_legalize(bb, call)
    shp = _shape_as_ints(sinfo)
    if shp is None or len(shp) != 2 or sinfo.dtype != "float16":
        return _default_exp_legalize(bb, call)
    m, n = shp
    gvar = bb.add_func(_exp_submit_primfunc(m, n), f"rknpu_submit_exp_m{m}_n{n}")
    return call_tir(gvar, [call.args[0]], [TensorStructInfo((m, n), "float16")])


def _emit_reciprocal_stage(bb: relax.BlockBuilder, x: relax.Expr) -> Optional[relax.Expr]:
    sinfo = x.struct_info
    if not isinstance(sinfo, TensorStructInfo):
        return None
    shp = _shape_as_ints(sinfo)
    if shp is None or len(shp) != 2 or sinfo.dtype != "float16":
        return None
    m, n = shp
    gvar = bb.add_func(_reciprocal_submit_primfunc(m, n), f"rknpu_submit_reciprocal_m{m}_n{n}")
    return bb.emit(call_tir(gvar, [x], [TensorStructInfo((m, n), "float16")]))


def _emit_multiply_stage(
    bb: relax.BlockBuilder, a: relax.Expr, b: relax.Expr, out_shape: List[int]
) -> Optional[relax.Expr]:
    a_sinfo = a.struct_info
    b_sinfo = b.struct_info
    if not isinstance(a_sinfo, TensorStructInfo) or not isinstance(b_sinfo, TensorStructInfo):
        return None
    a_shape = _shape_as_ints(a_sinfo)
    b_shape = _shape_as_ints(b_sinfo)
    if (
        a_shape is None
        or b_shape is None
        or len(out_shape) != 2
        or a_sinfo.dtype != "float16"
        or b_sinfo.dtype != "float16"
    ):
        return None
    m, n = out_shape
    bias_1d = len(b_shape) == 1 and b_shape[0] == n
    col_2d = len(b_shape) == 2 and b_shape[0] == m and b_shape[1] == 1
    tensor_2d = len(b_shape) == 2 and b_shape[0] == m and b_shape[1] == n
    if not (bias_1d or col_2d or tensor_2d):
        return None
    b_mode = 1 if bias_1d else 2 if col_2d else 0
    suffix = "bias" if b_mode == 1 else "col" if b_mode == 2 else "tensor"
    gvar = bb.add_func(
        _mul_submit_primfunc(m, n, b_mode),
        f"rknpu_submit_mul_m{m}_n{n}_{suffix}",
    )
    return call_tir(gvar, [a, b], [TensorStructInfo((m, n), "float16")])


def _legalize_rsqrt_opaque_stage(bb: relax.BlockBuilder, call: relax.Call) -> relax.Expr:
    sinfo = call.args[0].struct_info
    if not isinstance(sinfo, TensorStructInfo):
        return _default_rsqrt_legalize(bb, call)
    shp = _shape_as_ints(sinfo)
    if shp is None or len(shp) != 2 or sinfo.dtype != "float16":
        return _default_rsqrt_legalize(bb, call)
    m, n = shp
    gvar = bb.add_func(_rsqrt_opaque_primfunc(m, n), f"tir_rsqrt_opaque_m{m}_n{n}")
    return call_tir(gvar, [call.args[0]], [TensorStructInfo((m, n), "float16")])


def _legalize_gelu_opaque_stage(bb: relax.BlockBuilder, call: relax.Call) -> relax.Expr:
    sinfo = call.args[0].struct_info
    if not isinstance(sinfo, TensorStructInfo):
        return _default_gelu_legalize(bb, call)
    shp = _shape_as_ints(sinfo)
    if shp is None or len(shp) != 2 or sinfo.dtype != "float16":
        return _default_gelu_legalize(bb, call)
    m, n = shp
    gvar = bb.add_func(_gelu_submit_primfunc(m, n), f"rknpu_submit_gelu_m{m}_n{n}")
    return call_tir(gvar, [call.args[0]], [TensorStructInfo((m, n), "float16")])


def _legalize_divide_opaque_stage(bb: relax.BlockBuilder, call: relax.Call) -> relax.Expr:
    a_sinfo = call.args[0].struct_info
    b_sinfo = call.args[1].struct_info
    if not (isinstance(a_sinfo, TensorStructInfo) and isinstance(b_sinfo, TensorStructInfo)):
        return _default_divide_legalize(bb, call)
    a_shp = _shape_as_ints(a_sinfo)
    b_shp = _shape_as_ints(b_sinfo)
    if (
        a_shp is None
        or b_shp is None
        or len(a_shp) != 2
        or len(b_shp) != 2
        or a_sinfo.dtype != "float16"
        or b_sinfo.dtype != "float16"
    ):
        return _default_divide_legalize(bb, call)
    am, an = a_shp
    bm, bn = b_shp
    if am not in (1, bm) and bm not in (1, am):
        return _default_divide_legalize(bb, call)
    if an not in (1, bn) and bn not in (1, an):
        return _default_divide_legalize(bb, call)
    m = max(am, bm)
    n = max(an, bn)
    recip_b = _emit_reciprocal_stage(bb, call.args[1])
    if recip_b is not None:
        if _is_all_ones_const(call.args[0]) and bm == m and bn == n:
            return recip_b
        # Stage-12 -> stage-9 with b_mode=2 remains layout-fragile in real submit.
        # Optional workaround: materialize reciprocal [M,1] to [M,N] via matmul
        # so the final multiply runs as tensor-tensor (b_mode=0).
        expand_col_via_matmul = os.getenv(
            "TVM_RKNPU_DIVIDE_COL_EXPAND_VIA_MATMUL", ""
        ).lower() in ("1", "true", "yes", "on")
        if not expand_col_via_matmul:
            # Auto-enable the safer path when reciprocal->mul chain opt-in is set.
            expand_col_via_matmul = os.getenv(
                "TVM_RKNPU_PC_CHAIN_ALLOW_RECIPROCAL_TO_MUL", ""
            ).lower() in ("1", "true", "yes", "on")
        if expand_col_via_matmul and bm == m and bn == 1 and n > 1:
            ones_row = relax.const(np.ones((1, n), dtype=np.float16), "float16")
            recip_full = bb.emit(relax.op.matmul(recip_b, ones_row))
            mul = _emit_multiply_stage(bb, call.args[0], recip_full, [m, n])
            if mul is not None:
                return mul
        mul = _emit_multiply_stage(bb, call.args[0], recip_b, [m, n])
        if mul is not None:
            return mul
    gvar = bb.add_func(
        _divide_opaque_primfunc(am, an, bm, bn, m, n),
        f"tir_divide_opaque_a{am}x{an}_b{bm}x{bn}",
    )
    return call_tir(gvar, [call.args[0], call.args[1]], [TensorStructInfo((m, n), "float16")])


def _legalize_conv2d_to_rknpu_stage(bb: relax.BlockBuilder, call: relax.Call) -> relax.Expr:
    x_sinfo = call.args[0].struct_info
    w_sinfo = call.args[1].struct_info
    if not isinstance(x_sinfo, relax.TensorStructInfo) or not isinstance(
        w_sinfo, relax.TensorStructInfo
    ):
        return _default_conv2d_legalize(bb, call)
    x_shape = _shape_as_ints(x_sinfo)
    w_shape = _shape_as_ints(w_sinfo)
    if (
        x_shape is None
        or w_shape is None
        or len(x_shape) != 4
        or len(w_shape) != 4
        or x_sinfo.dtype != "float16"
        or w_sinfo.dtype != "float16"
    ):
        return _default_conv2d_legalize(bb, call)

    attrs = call.attrs
    if attrs.data_layout != "NCHW" or attrs.kernel_layout != "OIHW":
        return _default_conv2d_legalize(bb, call)
    if attrs.out_layout not in ("", "NCHW"):
        return _default_conv2d_legalize(bb, call)
    if int(attrs.groups) != 1:
        return _default_conv2d_legalize(bb, call)

    stride = _normalize_pair(attrs.strides)
    dilation = _normalize_pair(attrs.dilation)
    padding = _normalize_padding(attrs.padding)
    if stride is None or dilation is None or padding is None:
        return _default_conv2d_legalize(bb, call)
    if dilation != [1, 1]:
        return _default_conv2d_legalize(bb, call)

    out_sinfo = call.struct_info
    if not isinstance(out_sinfo, relax.TensorStructInfo):
        return _default_conv2d_legalize(bb, call)
    out_shape = _shape_as_ints(out_sinfo)
    if out_shape is None or len(out_shape) != 4 or out_sinfo.dtype != "float16":
        return _default_conv2d_legalize(bb, call)

    n, c, h, w = x_shape
    oc, wc, kh, kw = w_shape
    on, oo, oh, ow = out_shape
    if on != n or oo != oc or wc != c:
        return _default_conv2d_legalize(bb, call)

    stride_h, stride_w = stride
    pad_top, pad_left, pad_bottom, pad_right = padding
    gvar = bb.add_func(
        _conv2d_submit_primfunc(
            n,
            c,
            h,
            w,
            oc,
            kh,
            kw,
            oh,
            ow,
            stride_h,
            stride_w,
            pad_top,
            pad_left,
            pad_bottom,
            pad_right,
        ),
        (
            f"rknpu_submit_conv2d_n{n}_c{c}_h{h}_w{w}"
            f"_oc{oc}_kh{kh}_kw{kw}_oh{oh}_ow{ow}"
        ),
    )
    return call_tir(
        gvar,
        [call.args[0], call.args[1]],
        [TensorStructInfo((n, oc, oh, ow), "float16")],
    )


def _rknpu_tir_legalize_map() -> Dict[str, LegalizeFunc]:
    return {
        "relax.matmul": _legalize_matmul_to_rknpu_stage,
        "relax.nn.conv2d": _legalize_conv2d_to_rknpu_stage,
        "relax.add": _legalize_add_to_rknpu_stage,
        "relax.multiply": _legalize_multiply_to_rknpu_stage,
        "relax.divide": _legalize_divide_opaque_stage,
        "relax.exp": _legalize_exp_to_rknpu_stage,
        "relax.rsqrt": _legalize_rsqrt_opaque_stage,
        "relax.nn.gelu": _legalize_gelu_opaque_stage,
        "relax.nn.relu": _legalize_relu_to_rknpu_stage,
    }


def legalize_to_rknpu_tir_stages(mod: tvm.ir.IRModule) -> tvm.ir.IRModule:
    """Legalize supported Relax ops into extern-backed RKNPU stage PrimFuncs."""
    seq = tvm.transform.Sequential(
        [
            relax.transform.LegalizeOps(_rknpu_tir_legalize_map()),
            relax.transform.AnnotateTIROpPattern(),
        ]
    )
    return seq(mod)


def annotate_pc_chain_candidates() -> tvm.ir.transform.Pass:
    """Annotate PrimFuncs with count of chainable RKNPU stage submits.

    This is intentionally a hook pass for future lowering. If a fused TIR
    function contains more than one `rknpu_submit_*` extern call, it gains
    attribute `rknpu.pc_chain_candidates`.
    """

    @tir.transform.prim_func_pass(opt_level=0, name="AnnotateRKNPUPCChainCandidates")
    def _annotate(func: tir.PrimFunc, _mod, _ctx):
        count = 0
        call_extern_op = tvm.ir.Op.get("tir.call_extern")

        def _visit(node):
            nonlocal count
            if not isinstance(node, tir.Call):
                return
            if node.op != call_extern_op:
                return
            if not node.args:
                return
            name = node.args[0]
            if isinstance(name, tir.StringImm) and name.value.startswith("rknpu_submit_"):
                count += 1

        tir.stmt_functor.post_order_visit(func.body, _visit)
        if count > 1:
            return func.with_attr("rknpu.pc_chain_candidates", tir.IntImm("int64", count))
        return func

    return _annotate


def lower_pc_chain_submits() -> tvm.ir.transform.Pass:
    """Lower multiple stage submits inside a PrimFunc to one chain submit call.

    This is an experimental structural lowering pass intended to validate the
    TIR architecture path. It rewrites:
      rknpu_submit_stage_a(...)
      rknpu_submit_stage_b(...)
      ...
    into:
      rknpu_submit_chain_stage_v2(num_tasks, stage0_id, stage0_args..., stage1_id, ...)
    while preserving the fused function boundary and metadata.
    """

    @tir.transform.prim_func_pass(opt_level=0, name="LowerRKNPUPCChainSubmits")
    def _lower(func: tir.PrimFunc, _mod, _ctx):
        call_extern_op = tvm.ir.Op.get("tir.call_extern")
        stage_id_map = {
            "rknpu_submit_matmul_stage": 1,
            "rknpu_submit_add_stage": 2,
            "rknpu_submit_mul_stage": 9,
            "rknpu_submit_exp_stage": 10,
            "rknpu_submit_reciprocal_stage": 12,
            "rknpu_submit_gelu_stage": 13,
            "rknpu_submit_relu_stage": 3,
            "rknpu_submit_relu_stage_4d": 4,
            "rknpu_submit_conv2d_stage": 5,
        }
        stage_calls = []

        def _collect(node):
            if not isinstance(node, tir.Call):
                return
            if node.op != call_extern_op or not node.args:
                return
            name = node.args[0]
            if isinstance(name, tir.StringImm):
                if name.value.startswith("rknpu_submit_") and name.value != "rknpu_submit_chain_stage":
                    stage_calls.append(node)

        tir.stmt_functor.post_order_visit(func.body, _collect)
        if len(stage_calls) <= 0:
            return func
        raw_stages = []
        for stage_call in stage_calls:
            name = stage_call.args[0]
            if not isinstance(name, tir.StringImm) or name.value not in stage_id_map:
                return func
            raw_stages.append((stage_id_map[name.value], list(stage_call.args[1:])))
        include_singletons = os.getenv("TVM_RKNPU_PC_CHAIN_INCLUDE_SINGLETONS", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        singleton_allowlist = _parse_stage_id_allowlist(
            os.getenv("TVM_RKNPU_PC_CHAIN_SINGLETON_ALLOWLIST", "")
        )
        if len(stage_calls) == 1 and not include_singletons:
            singleton_stage_id = raw_stages[0][0]
            if singleton_stage_id not in singleton_allowlist:
                return func
        # Correctness-first mode: split every stage into its own chain submit so
        # stage boundaries materialize writeback buffers before consumers run.
        split_every_stage = os.getenv("TVM_RKNPU_PC_CHAIN_SPLIT_STAGES", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        # Hardware-aware collapse: matmul + bias-add + relu -> one fused stage.
        # Set TVM_RKNPU_PC_CHAIN_DISABLE_FUSION=1 to keep one task per stage.
        disable_fusion = split_every_stage or os.getenv(
            "TVM_RKNPU_PC_CHAIN_DISABLE_FUSION", ""
        ).lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        disable_add_relu_raw = os.getenv("TVM_RKNPU_PC_CHAIN_DISABLE_ADD_RELU_FUSION")
        if disable_add_relu_raw is not None and disable_add_relu_raw.lower() in (
            "0",
            "false",
            "no",
            "off",
        ):
            raise RuntimeError(
                "RKNPU add+relu fused stage (stage 7) is unsupported: real-submit "
                "correctness is wrong. Keep add and relu as separate stages."
            )
        enable_matmul_bias_fusion_raw = os.getenv(
            "TVM_RKNPU_PC_CHAIN_ENABLE_MATMUL_BIAS_FUSION"
        )
        # Stage-11 fusion is part of the normal supported 2-D dense policy.
        # Keep the env var only as an explicit opt-out for experiments.
        if enable_matmul_bias_fusion_raw is None:
            enable_matmul_bias_fusion = True
        else:
            enable_matmul_bias_fusion = enable_matmul_bias_fusion_raw.lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
        fused_stages = []
        i = 0
        while i < len(raw_stages):
            if (not disable_fusion) and i + 2 < len(raw_stages):
                sid0, a0 = raw_stages[i]
                sid1, a1 = raw_stages[i + 1]
                sid2, a2 = raw_stages[i + 2]
                bias_1d = len(a1) > 5 and isinstance(a1[5], tir.IntImm) and int(a1[5]) == 1
                if (
                    sid0 == 1
                    and sid1 == 2
                    and sid2 == 3
                    and bias_1d
                    and tvm.ir.structural_equal(a1[0], a0[2])   # add input is matmul output
                    and tvm.ir.structural_equal(a2[0], a1[2])   # relu input is add output
                    and tvm.ir.structural_equal(a1[3], a0[3])   # m
                    and tvm.ir.structural_equal(a1[4], a0[5])   # n
                    and tvm.ir.structural_equal(a2[2], a0[3])   # m
                    and tvm.ir.structural_equal(a2[3], a0[5])   # n
                ):
                    fused_stages.append((6, [a0[0], a0[1], a1[1], a2[1], a0[3], a0[4], a0[5]]))
                    i += 3
                    continue
            if enable_matmul_bias_fusion and (not disable_fusion) and i + 1 < len(raw_stages):
                sid0, a0 = raw_stages[i]
                sid1, a1 = raw_stages[i + 1]
                bias_1d = len(a1) > 5 and isinstance(a1[5], tir.IntImm) and int(a1[5]) == 1
                if (
                    sid0 == 1
                    and sid1 == 2
                    and bias_1d
                    and tvm.ir.structural_equal(a1[0], a0[2])   # add input is matmul output
                    and tvm.ir.structural_equal(a1[3], a0[3])   # m
                    and tvm.ir.structural_equal(a1[4], a0[5])   # n
                ):
                    fused_stages.append((11, [a0[0], a0[1], a1[1], a1[2], a0[3], a0[4], a0[5]]))
                    i += 2
                    continue
            if (not disable_fusion) and i + 1 < len(raw_stages):
                sid0, a0 = raw_stages[i]
                sid1, a1 = raw_stages[i + 1]
                if (
                    sid0 == 5
                    and sid1 == 4
                    and tvm.ir.structural_equal(a0[2], a1[0])   # relu input is conv output
                    and tvm.ir.structural_equal(a1[2], a0[3])   # n
                    and tvm.ir.structural_equal(a1[3], a0[7])   # ch == oc
                    and tvm.ir.structural_equal(a1[4], a0[10])  # h == oh
                    and tvm.ir.structural_equal(a1[5], a0[11])  # w == ow
                ):
                    # Fuse conv2d + relu4d into one conv stage with DPU relu enabled.
                    fused_stages.append(
                        (
                            8,
                            [
                                a0[0],  # data
                                a0[1],  # weight
                                a1[1],  # final output
                                a0[3],  # n
                                a0[4],  # c
                                a0[5],  # h
                                a0[6],  # w
                                a0[7],  # oc
                                a0[8],  # kh
                                a0[9],  # kw
                                a0[10], # oh
                                a0[11], # ow
                                a0[12], # stride_h
                                a0[13], # stride_w
                                a0[14], # pad_top
                                a0[15], # pad_left
                                a0[16], # pad_bottom
                                a0[17], # pad_right
                            ],
                        )
                    )
                    i += 2
                    continue
            fused_stages.append(raw_stages[i])
            i += 1

        fail_on_incompatible = os.getenv(
            "TVM_RKNPU_PC_CHAIN_FAIL_ON_INCOMPATIBLE", ""
        ).lower() in ("1", "true", "yes", "on")
        blocked_boundaries: List[Dict[str, object]] = []
        if split_every_stage:
            submit_groups = [[stage] for stage in fused_stages]
        else:
            submit_groups = []
            current_group: List[tuple[int, List[tvm.tir.PrimExpr]]] = []
            for idx, stage in enumerate(fused_stages):
                current_group.append(stage)
                next_stage = fused_stages[idx + 1] if idx + 1 < len(fused_stages) else None
                if next_stage is not None:
                    compatible, reason = _pc_chain_boundary_compatible(stage, next_stage)
                    sid, stage_args = stage
                    next_sid, next_args = next_stage
                    if not compatible:
                        blocked_boundaries.append(
                            {
                                "boundary_index": idx,
                                "producer_stage_id": sid,
                                "consumer_stage_id": next_sid,
                                "producer_shape_hint": _pc_chain_stage_shape_hint(sid, stage_args),
                                "consumer_shape_hint": _pc_chain_stage_shape_hint(next_sid, next_args),
                                "reason": reason,
                            }
                        )
                        if fail_on_incompatible:
                            raise RuntimeError(
                                "RKNPU chain ABI incompatibility at boundary "
                                f"{idx}: stage {sid} -> {next_sid} ({reason}); "
                                f"producer[{_pc_chain_stage_shape_hint(sid, stage_args)}], "
                                f"consumer[{_pc_chain_stage_shape_hint(next_sid, next_args)}]"
                            )
                        submit_groups.append(current_group)
                        current_group = []
            if current_group:
                submit_groups.append(current_group)
        seen = {"count": 0}

        def _rewrite(expr):
            if not isinstance(expr, tir.Call):
                return expr
            if expr.op != call_extern_op or not expr.args:
                return expr
            name = expr.args[0]
            if not isinstance(name, tir.StringImm):
                return expr
            if not name.value.startswith("rknpu_submit_") or name.value.startswith("rknpu_submit_chain_stage"):
                return expr

            seen["count"] += 1
            submit_idx = seen["count"] - 1
            if submit_idx < len(submit_groups):
                submit_stages = submit_groups[submit_idx]
                chain_args = [
                    tir.StringImm("rknpu_submit_chain_stage_v2"),
                    tir.IntImm("int32", len(submit_stages)),
                ]
                for stage_id, stage_args in submit_stages:
                    chain_args.append(tir.IntImm("int32", stage_id))
                    chain_args.extend(stage_args)
                return tir.Call(
                    expr.dtype,
                    call_extern_op,
                    chain_args,
                    expr.span,
                )
            return tir.const(0, expr.dtype)

        new_body = tir.stmt_functor.ir_transform(func.body, None, _rewrite, ["tir.Call"])
        lowered = (
            func.with_body(new_body)
            .with_attr("rknpu.pc_chain_lowered", tir.IntImm("int64", 1))
            .with_attr("rknpu.pc_chained_tasks", tir.IntImm("int64", len(fused_stages)))
        )
        lowered = lowered.with_attr(
            "rknpu.pc_chain_compat_blocked_count",
            tir.IntImm("int64", len(blocked_boundaries)),
        )
        if blocked_boundaries:
            lowered = lowered.with_attr(
                "rknpu.pc_chain_compat_blocked_json",
                tvm.runtime.String(json.dumps(blocked_boundaries)),
            )
        return lowered

    return _lower


def lower_to_rknpu_tir(mod: tvm.ir.IRModule) -> tvm.ir.IRModule:
    """Lower a Relax module using the experimental RKNPU TIR path."""
    mod = DecomposeLayerNormForRKNPU()(mod)
    if os.getenv("TVM_RKNPU_TIR_DECOMPOSE_SOFTMAX", "").lower() in ("1", "true", "yes", "on"):
        mod = DecomposeSoftmaxForRKNPU()(mod)
    mod = legalize_to_rknpu_tir_stages(mod)
    seq = tvm.transform.Sequential(
        [
            relax.transform.FuseOps(),
            relax.transform.FuseTIR(),
            annotate_pc_chain_candidates(),
        ]
    )
    return seq(mod)


def _extract_direct_rknpu_submit_calls(func: tir.PrimFunc) -> Optional[List[tir.Call]]:
    call_extern_op = tvm.ir.Op.get("tir.call_extern")
    calls: List[tir.Call] = []

    def _collect(node):
        if not isinstance(node, tir.Call):
            return
        if node.op != call_extern_op or not node.args:
            return
        name = node.args[0]
        if not isinstance(name, tir.StringImm):
            return
        if not name.value.startswith("rknpu_submit_"):
            return
        if name.value.startswith("rknpu_submit_chain_stage"):
            return
        calls.append(node)

    tir.stmt_functor.post_order_visit(func.body, _collect)
    if not calls:
        return None
    return calls


def _direct_rknpu_stage_ids(func: tir.PrimFunc) -> Optional[List[int]]:
    calls = _extract_direct_rknpu_submit_calls(func)
    if calls is None:
        return None
    stage_id_map = {
        "rknpu_submit_matmul_stage": 1,
        "rknpu_submit_add_stage": 2,
        "rknpu_submit_mul_stage": 9,
        "rknpu_submit_exp_stage": 10,
        "rknpu_submit_reciprocal_stage": 12,
        "rknpu_submit_gelu_stage": 13,
        "rknpu_submit_relu_stage": 3,
        "rknpu_submit_relu_stage_4d": 4,
        "rknpu_submit_conv2d_stage": 5,
    }
    out: List[int] = []
    for call in calls:
        name = call.args[0]
        if not isinstance(name, tir.StringImm) or name.value not in stage_id_map:
            return None
        out.append(stage_id_map[name.value])
    return out


def _extract_allocate_slots(func: tir.PrimFunc) -> Dict[tir.Var, tuple[str, int]]:
    slots: Dict[tir.Var, tuple[str, int]] = {}

    def _collect(node):
        if not isinstance(node, tir.Allocate):
            return
        numel = 1
        for extent in node.extents:
            val = _as_int(extent)
            if val is None:
                return
            numel *= int(val)
        slots[node.buffer_var] = (node.dtype, int(numel))

    tir.stmt_functor.post_order_visit(func.body, _collect)
    return slots


def _infer_stage_temp_slot(call: tir.Call) -> Optional[tuple[tir.Var, str, int]]:
    if not isinstance(call, tir.Call) or not call.args:
        return None
    name = call.args[0]
    if not isinstance(name, tir.StringImm):
        return None
    args = list(call.args)
    dtype = "float16"
    if name.value == "rknpu_submit_matmul_stage" and len(args) >= 7:
        out = args[3]
        m = _as_int(args[4])
        n = _as_int(args[6])
        if isinstance(out, tir.Var) and m is not None and n is not None:
            return out, dtype, int(m) * int(n)
    if name.value in ("rknpu_submit_add_stage", "rknpu_submit_mul_stage") and len(args) >= 6:
        out = args[3]
        m = _as_int(args[4])
        n = _as_int(args[5])
        if isinstance(out, tir.Var) and m is not None and n is not None:
            return out, dtype, int(m) * int(n)
    if name.value in (
        "rknpu_submit_exp_stage",
        "rknpu_submit_reciprocal_stage",
        "rknpu_submit_gelu_stage",
        "rknpu_submit_relu_stage",
    ) and len(args) >= 5:
        out = args[2]
        m = _as_int(args[3])
        n = _as_int(args[4])
        if isinstance(out, tir.Var) and m is not None and n is not None:
            return out, dtype, int(m) * int(n)
    return None


def _collect_relax_var_use_counts(func: relax.Function) -> Dict[relax.Var, int]:
    counts: Dict[relax.Var, int] = {}

    def _visit(expr):
        if isinstance(expr, relax.Var):
            counts[expr] = counts.get(expr, 0) + 1

    relax.analysis.post_order_visit(func.body, _visit)
    for param in func.params:
        counts.pop(param, None)
    return counts


def _is_rknpu_call_tir(mod: tvm.ir.IRModule, value: relax.Expr) -> bool:
    if not isinstance(value, relax.Call):
        return False
    if value.op != tvm.ir.Op.get("relax.call_tir") or not value.args:
        return False
    callee = value.args[0]
    if not isinstance(callee, tvm.ir.GlobalVar):
        return False
    func = mod[callee]
    return isinstance(func, tir.PrimFunc) and _extract_direct_rknpu_submit_calls(func) is not None


def _call_tir_inputs(value: relax.Call) -> Optional[List[relax.Expr]]:
    if len(value.args) < 2 or not isinstance(value.args[1], relax.Tuple):
        return None
    return list(value.args[1].fields)


def _tensor_buffer_from_sinfo(name: str, sinfo: TensorStructInfo, data=None) -> Optional[tir.Buffer]:
    shape = _shape_as_ints(sinfo)
    if shape is None:
        return None
    return tir.decl_buffer(tuple(shape), sinfo.dtype, name=name, data=data)


def _wrap_stmt_in_root_block(body: tir.Stmt) -> tir.Stmt:
    root = tir.SBlock([], [], [], "root", body)
    return tir.SBlockRealize([], tir.const(True, "bool"), root)


def _merge_linear_rknpu_call_tir_groups(mod: tvm.ir.IRModule) -> tvm.ir.IRModule:
    call_tir_op = tvm.ir.Op.get("relax.call_tir")
    bb = relax.BlockBuilder(mod)

    def _build_group_primfunc(
        group: List[relax.VarBinding], use_count: Dict[relax.Var, int]
    ) -> Optional[tuple[tir.PrimFunc, List[relax.Expr]]]:
        final_var = group[-1].var
        final_sinfo = final_var.struct_info
        if not isinstance(final_sinfo, TensorStructInfo):
            return None
        final_buf = _tensor_buffer_from_sinfo("out", final_sinfo)
        if final_buf is None:
            return None

        produced = {binding.var for binding in group}
        external_exprs: List[relax.Expr] = []
        external_buffers: Dict[relax.Expr, tir.Buffer] = {}
        internal_buffers: Dict[relax.Var, tir.Buffer] = {}
        internal_allocs: List[tuple[tir.Var, str, int]] = []
        bound_data_vars: set[tir.Var] = {final_buf.data}

        def _buffer_for_expr(expr: relax.Expr) -> Optional[tir.Buffer]:
            if isinstance(expr, relax.Var):
                if expr.same_as(final_var):
                    return final_buf
                if expr in internal_buffers:
                    return internal_buffers[expr]
                if expr in produced:
                    return None
            if expr not in external_buffers:
                sinfo = getattr(expr, "struct_info", None)
                if not isinstance(sinfo, TensorStructInfo):
                    return None
                buf = _tensor_buffer_from_sinfo(f"arg{len(external_exprs)}", sinfo)
                if buf is None:
                    return None
                external_exprs.append(expr)
                external_buffers[expr] = buf
                bound_data_vars.add(buf.data)
            return external_buffers[expr]

        for binding in group[:-1]:
            sinfo = binding.var.struct_info
            if not isinstance(sinfo, TensorStructInfo):
                return None
            shape = _shape_as_ints(sinfo)
            if shape is None:
                return None
            storage = tir.Var(
                f"{binding.var.name_hint}_data",
                tvm.ir.PointerType(tvm.ir.PrimType(sinfo.dtype)),
            )
            buf = _tensor_buffer_from_sinfo(binding.var.name_hint, sinfo, data=storage)
            if buf is None:
                return None
            internal_buffers[binding.var] = buf
            internal_allocs.append((storage, sinfo.dtype, int(math.prod(shape))))
            bound_data_vars.add(buf.data)

        stmts: List[tir.Stmt] = []
        temp_slots: Dict[tir.Var, tuple[str, int]] = {}
        for binding in group:
            value = binding.value
            if not isinstance(value, relax.Call) or value.op != call_tir_op:
                return None
            callee = value.args[0]
            if not isinstance(callee, tvm.ir.GlobalVar):
                return None
            callee_func = mod[callee]
            if not isinstance(callee_func, tir.PrimFunc):
                return None
            submit_calls = _extract_direct_rknpu_submit_calls(callee_func)
            inputs = _call_tir_inputs(value)
            if submit_calls is None or inputs is None:
                return None
            if len(callee_func.params) != len(inputs) + 1:
                return None
            subst: Dict[tir.Var, tir.PrimExpr] = {}
            for inp_expr, param in zip(inputs, callee_func.params[:-1]):
                old_buf = callee_func.buffer_map[param]
                new_buf = _buffer_for_expr(inp_expr)
                if new_buf is None:
                    return None
                subst[old_buf.data] = new_buf.data
            out_buf = _buffer_for_expr(binding.var)
            if out_buf is None:
                return None
            old_out_buf = callee_func.buffer_map[callee_func.params[-1]]
            subst[old_out_buf.data] = out_buf.data
            for alloc_idx, (old_var, (dtype, numel)) in enumerate(_extract_allocate_slots(callee_func).items()):
                new_var = tir.Var(
                    f"{callee.name_hint}_tmp{len(internal_allocs) + alloc_idx}",
                    tvm.ir.PointerType(tvm.ir.PrimType(dtype)),
                )
                subst[old_var] = new_var
                internal_allocs.append((new_var, dtype, numel))
            for submit_call in submit_calls:
                rewritten = tir.stmt_functor.substitute(submit_call, subst)
                inferred = _infer_stage_temp_slot(rewritten)
                if inferred is not None:
                    temp_var, dtype, numel = inferred
                    if temp_var not in bound_data_vars and temp_var not in temp_slots:
                        temp_slots[temp_var] = (dtype, numel)
                        internal_allocs.append((temp_var, dtype, numel))
                stmts.append(tir.Evaluate(rewritten))

        if not stmts:
            return None
        body: tir.Stmt = stmts[0] if len(stmts) == 1 else tir.SeqStmt(stmts)
        for storage, dtype, numel in reversed(internal_allocs):
            body = tir.Allocate(storage, dtype, [tir.IntImm("int64", numel)], tir.const(True, "bool"), body)
        body = _wrap_stmt_in_root_block(body)

        params: List = [*external_buffers.values(), final_buf]
        prim = tir.PrimFunc(
            params,
            body,
            attrs=tvm.ir.make_node(
                "ir.DictAttrs",
                **{
                    "tir.noalias": True,
                    "op_pattern": _K_OPAQUE,
                },
            ),
        )
        prim = prim.with_attr("private", tir.IntImm("int64", 1))
        prim = prim.with_attr("rknpu.linear_chain_group", tir.IntImm("int64", len(group)))
        return prim, external_exprs

    updates: Dict[tvm.ir.GlobalVar, relax.Function] = {}
    for gv, func in mod.functions_items():
        if not isinstance(func, relax.Function):
            continue
        if not isinstance(func.body, relax.SeqExpr):
            continue
        use_count = _collect_relax_var_use_counts(func)
        blocks = []
        changed = False
        for block in func.body.blocks:
            if not isinstance(block, relax.DataflowBlock):
                blocks.append(block)
                continue
            new_bindings: List[relax.Binding] = []
            bindings = list(block.bindings)
            i = 0
            while i < len(bindings):
                binding = bindings[i]
                if not isinstance(binding, relax.VarBinding) or not _is_rknpu_call_tir(mod, binding.value):
                    new_bindings.append(binding)
                    i += 1
                    continue
                group = [binding]
                j = i + 1
                prev_var = binding.var
                while j < len(bindings):
                    nxt = bindings[j]
                    if not isinstance(nxt, relax.VarBinding) or not _is_rknpu_call_tir(mod, nxt.value):
                        break
                    if use_count.get(prev_var, 0) != 1:
                        break
                    nxt_inputs = _call_tir_inputs(nxt.value)
                    if nxt_inputs is None or not any(
                        isinstance(arg, relax.Var) and arg.same_as(prev_var) for arg in nxt_inputs
                    ):
                        break
                    group.append(nxt)
                    prev_var = nxt.var
                    j += 1
                def _collect_group_stage_ids(group_bindings: List[relax.VarBinding]) -> Optional[List[int]]:
                    stage_ids_out: List[int] = []
                    for group_binding in group_bindings:
                        group_value = group_binding.value
                        if not isinstance(group_value, relax.Call):
                            return None
                        group_callee = group_value.args[0]
                        if not isinstance(group_callee, tvm.ir.GlobalVar):
                            return None
                        group_func = mod[group_callee]
                        if not isinstance(group_func, tir.PrimFunc):
                            return None
                        stage_ids = _direct_rknpu_stage_ids(group_func)
                        if stage_ids is None:
                            return None
                        stage_ids_out.extend(stage_ids)
                    return stage_ids_out

                combined_stage_ids = _collect_group_stage_ids(group)
                if len(group) <= 1:
                    new_bindings.append(binding)
                    i += 1
                    continue
                built = _build_group_primfunc(group, use_count)
                if built is None:
                    # Generic fallback: if the maximal chain is not legal/buildable,
                    # merge the longest prefix that can be lowered.
                    merged_prefix = False
                    for prefix_len in range(len(group) - 1, 1, -1):
                        prefix = group[:prefix_len]
                        built_prefix = _build_group_primfunc(prefix, use_count)
                        if built_prefix is None:
                            continue
                        prim, inputs = built_prefix
                        name = "fused_" + "_".join(
                            str(c.value.args[0].name_hint)
                            for c in prefix
                            if isinstance(c.value, relax.Call)
                            and isinstance(c.value.args[0], tvm.ir.GlobalVar)
                        )
                        new_gv = bb.add_func(prim, bb.get_unique_name(name))
                        new_call = call_tir(new_gv, inputs, [prefix[-1].var.struct_info])
                        new_bindings.append(relax.VarBinding(prefix[-1].var, new_call))
                        changed = True
                        i += prefix_len
                        merged_prefix = True
                        break
                    if merged_prefix:
                        continue
                    new_bindings.append(binding)
                    i += 1
                    continue
                prim, inputs = built
                name = "fused_" + "_".join(
                    str(c.value.args[0].name_hint)
                    for c in group
                    if isinstance(c.value, relax.Call) and isinstance(c.value.args[0], tvm.ir.GlobalVar)
                )
                new_gv = bb.add_func(prim, bb.get_unique_name(name))
                new_call = call_tir(new_gv, inputs, [group[-1].var.struct_info])
                new_bindings.append(relax.VarBinding(group[-1].var, new_call))
                changed = True
                i = j
            blocks.append(relax.DataflowBlock(new_bindings))
        if changed:
            updates[gv] = relax.Function(
                func.params,
                relax.SeqExpr(blocks, func.body.body),
                func.ret_struct_info,
                func.is_pure,
                func.attrs,
                func.span,
            )
    if not updates:
        return mod
    for gv, new_func in updates.items():
        bb.update_func(gv, new_func)
    return bb.get()


def _merge_deferred_rknpu_call_tir_groups(mod: tvm.ir.IRModule) -> tvm.ir.IRModule:
    call_tir_op = tvm.ir.Op.get("relax.call_tir")
    bb = relax.BlockBuilder(mod)
    allowed_stage_patterns = {
        (1, 2, 1),  # q/v projection -> scores/ctx
        (2, 1, 2),  # layernorm centered add -> var+eps
    }

    def _build_group_primfunc(
        group: List[relax.VarBinding], use_count: Dict[relax.Var, int]
    ) -> Optional[tuple[tir.PrimFunc, List[relax.Expr]]]:
        final_var = group[-1].var
        final_sinfo = final_var.struct_info
        if not isinstance(final_sinfo, TensorStructInfo):
            return None
        final_buf = _tensor_buffer_from_sinfo("out", final_sinfo)
        if final_buf is None:
            return None

        produced = {binding.var for binding in group}
        external_exprs: List[relax.Expr] = []
        external_buffers: Dict[relax.Expr, tir.Buffer] = {}
        internal_buffers: Dict[relax.Var, tir.Buffer] = {}
        internal_allocs: List[tuple[tir.Var, str, int]] = []
        bound_data_vars: set[tir.Var] = {final_buf.data}

        def _buffer_for_expr(expr: relax.Expr) -> Optional[tir.Buffer]:
            if isinstance(expr, relax.Var):
                if expr.same_as(final_var):
                    return final_buf
                if expr in internal_buffers:
                    return internal_buffers[expr]
                if expr in produced:
                    return None
            if expr not in external_buffers:
                sinfo = getattr(expr, "struct_info", None)
                if not isinstance(sinfo, TensorStructInfo):
                    return None
                buf = _tensor_buffer_from_sinfo(f"arg{len(external_exprs)}", sinfo)
                if buf is None:
                    return None
                external_exprs.append(expr)
                external_buffers[expr] = buf
                bound_data_vars.add(buf.data)
            return external_buffers[expr]

        for binding in group[:-1]:
            sinfo = binding.var.struct_info
            if not isinstance(sinfo, TensorStructInfo):
                return None
            shape = _shape_as_ints(sinfo)
            if shape is None:
                return None
            storage = tir.Var(
                f"{binding.var.name_hint}_data",
                tvm.ir.PointerType(tvm.ir.PrimType(sinfo.dtype)),
            )
            buf = _tensor_buffer_from_sinfo(binding.var.name_hint, sinfo, data=storage)
            if buf is None:
                return None
            internal_buffers[binding.var] = buf
            internal_allocs.append((storage, sinfo.dtype, int(math.prod(shape))))
            bound_data_vars.add(buf.data)

        stmts: List[tir.Stmt] = []
        temp_slots: Dict[tir.Var, tuple[str, int]] = {}
        for binding in group:
            value = binding.value
            if not isinstance(value, relax.Call) or value.op != call_tir_op:
                return None
            callee = value.args[0]
            if not isinstance(callee, tvm.ir.GlobalVar):
                return None
            callee_func = mod[callee]
            if not isinstance(callee_func, tir.PrimFunc):
                return None
            submit_calls = _extract_direct_rknpu_submit_calls(callee_func)
            inputs = _call_tir_inputs(value)
            if submit_calls is None or inputs is None:
                return None
            if len(callee_func.params) != len(inputs) + 1:
                return None
            subst: Dict[tir.Var, tir.PrimExpr] = {}
            for inp_expr, param in zip(inputs, callee_func.params[:-1]):
                old_buf = callee_func.buffer_map[param]
                new_buf = _buffer_for_expr(inp_expr)
                if new_buf is None:
                    return None
                subst[old_buf.data] = new_buf.data
            out_buf = _buffer_for_expr(binding.var)
            if out_buf is None:
                return None
            old_out_buf = callee_func.buffer_map[callee_func.params[-1]]
            subst[old_out_buf.data] = out_buf.data
            for alloc_idx, (old_var, (dtype, numel)) in enumerate(_extract_allocate_slots(callee_func).items()):
                new_var = tir.Var(
                    f"{callee.name_hint}_tmp{len(internal_allocs) + alloc_idx}",
                    tvm.ir.PointerType(tvm.ir.PrimType(dtype)),
                )
                subst[old_var] = new_var
                internal_allocs.append((new_var, dtype, numel))
            for submit_call in submit_calls:
                rewritten = tir.stmt_functor.substitute(submit_call, subst)
                inferred = _infer_stage_temp_slot(rewritten)
                if inferred is not None:
                    temp_var, dtype, numel = inferred
                    if temp_var not in bound_data_vars and temp_var not in temp_slots:
                        temp_slots[temp_var] = (dtype, numel)
                        internal_allocs.append((temp_var, dtype, numel))
                stmts.append(tir.Evaluate(rewritten))

        if not stmts:
            return None
        body: tir.Stmt = stmts[0] if len(stmts) == 1 else tir.SeqStmt(stmts)
        for storage, dtype, numel in reversed(internal_allocs):
            body = tir.Allocate(storage, dtype, [tir.IntImm("int64", numel)], tir.const(True, "bool"), body)
        body = _wrap_stmt_in_root_block(body)

        params: List = [*external_buffers.values(), final_buf]
        prim = tir.PrimFunc(
            params,
            body,
            attrs=tvm.ir.make_node(
                "ir.DictAttrs",
                **{
                    "tir.noalias": True,
                    "op_pattern": _K_OPAQUE,
                },
            ),
        )
        prim = prim.with_attr("private", tir.IntImm("int64", 1))
        prim = prim.with_attr("rknpu.deferred_chain_group", tir.IntImm("int64", len(group)))
        return prim, external_exprs

    updates: Dict[tvm.ir.GlobalVar, relax.Function] = {}
    for gv, func in mod.functions_items():
        if not isinstance(func, relax.Function):
            continue
        if not isinstance(func.body, relax.SeqExpr):
            continue
        use_count = _collect_relax_var_use_counts(func)
        blocks = []
        changed = False
        for block in func.body.blocks:
            if not isinstance(block, relax.DataflowBlock):
                blocks.append(block)
                continue
            bindings = list(block.bindings)
            planned: Dict[int, tuple[relax.VarBinding, tir.PrimFunc, List[relax.Expr]]] = {}
            removed: set[int] = set()
            for i, binding in enumerate(bindings):
                if i in removed or not isinstance(binding, relax.VarBinding) or not _is_rknpu_call_tir(mod, binding.value):
                    continue
                if use_count.get(binding.var, 0) != 1:
                    continue
                value = binding.value
                if not isinstance(value, relax.Call):
                    continue
                callee = value.args[0]
                if not isinstance(callee, tvm.ir.GlobalVar):
                    continue
                func_i = mod[callee]
                if not isinstance(func_i, tir.PrimFunc):
                    continue
                producer_ids = _direct_rknpu_stage_ids(func_i)
                if producer_ids not in ([1, 2], [2]):
                    continue
                consumer_idx = None
                consumer_binding = None
                for j in range(i + 1, len(bindings)):
                    if j in removed:
                        continue
                    nxt = bindings[j]
                    if not isinstance(nxt, relax.VarBinding) or not _is_rknpu_call_tir(mod, nxt.value):
                        continue
                    nxt_inputs = _call_tir_inputs(nxt.value)
                    if nxt_inputs is None:
                        continue
                    if any(isinstance(arg, relax.Var) and arg.same_as(binding.var) for arg in nxt_inputs):
                        consumer_idx = j
                        consumer_binding = nxt
                        break
                if consumer_idx is None or consumer_binding is None or consumer_idx in planned:
                    continue
                consumer_value = consumer_binding.value
                if not isinstance(consumer_value, relax.Call):
                    continue
                consumer_callee = consumer_value.args[0]
                if not isinstance(consumer_callee, tvm.ir.GlobalVar):
                    continue
                func_j = mod[consumer_callee]
                if not isinstance(func_j, tir.PrimFunc):
                    continue
                consumer_ids = _direct_rknpu_stage_ids(func_j)
                if consumer_ids is None:
                    continue
                if tuple([*producer_ids, *consumer_ids]) not in allowed_stage_patterns:
                    continue
                built = _build_group_primfunc([binding, consumer_binding], use_count)
                if built is None:
                    continue
                prim, inputs = built
                planned[consumer_idx] = (consumer_binding, prim, inputs)
                removed.add(i)
                changed = True

            if not changed:
                blocks.append(block)
                continue

            new_bindings: List[relax.Binding] = []
            for idx, binding in enumerate(bindings):
                if idx in removed:
                    continue
                if idx in planned:
                    consumer_binding, prim, inputs = planned[idx]
                    name = "fused_deferred_" + consumer_binding.var.name_hint
                    new_gv = bb.add_func(prim, bb.get_unique_name(name))
                    new_call = call_tir(new_gv, inputs, [consumer_binding.var.struct_info])
                    new_bindings.append(relax.VarBinding(consumer_binding.var, new_call))
                    continue
                new_bindings.append(binding)
            blocks.append(relax.DataflowBlock(new_bindings))
        if changed:
            updates[gv] = relax.Function(
                func.params,
                relax.SeqExpr(blocks, func.body.body),
                func.ret_struct_info,
                func.is_pure,
                func.attrs,
                func.span,
            )
    if not updates:
        return mod
    for gv, new_func in updates.items():
        bb.update_func(gv, new_func)
    return bb.get()


def _merge_deferred_tuple_projection_groups(mod: tvm.ir.IRModule) -> tvm.ir.IRModule:
    call_tir_op = tvm.ir.Op.get("relax.call_tir")
    bb = relax.BlockBuilder(mod)

    def _build_group_primfunc(
        producer: relax.VarBinding, consumer: relax.VarBinding
    ) -> Optional[tuple[tir.PrimFunc, List[relax.Expr], List[TensorStructInfo]]]:
        producer_sinfo = producer.var.struct_info
        consumer_sinfo = consumer.var.struct_info
        if not isinstance(producer_sinfo, TensorStructInfo):
            return None
        if not isinstance(consumer_sinfo, tvm.relax.TupleStructInfo):
            return None
        out_sinfos: List[TensorStructInfo] = []
        out_bufs: List[tir.Buffer] = []
        for idx, field in enumerate(consumer_sinfo.fields):
            if not isinstance(field, TensorStructInfo):
                return None
            out_sinfos.append(field)
            buf = _tensor_buffer_from_sinfo(f"out{idx}", field)
            if buf is None:
                return None
            out_bufs.append(buf)

        prod_shape = _shape_as_ints(producer_sinfo)
        if prod_shape is None:
            return None
        prod_storage = tir.Var(
            f"{producer.var.name_hint}_data",
            tvm.ir.PointerType(tvm.ir.PrimType(producer_sinfo.dtype)),
        )
        prod_buf = _tensor_buffer_from_sinfo(producer.var.name_hint, producer_sinfo, data=prod_storage)
        if prod_buf is None:
            return None

        external_exprs: List[relax.Expr] = []
        external_buffers: Dict[relax.Expr, tir.Buffer] = {}
        internal_allocs: List[tuple[tir.Var, str, int]] = [
            (prod_storage, producer_sinfo.dtype, int(math.prod(prod_shape)))
        ]
        bound_data_vars: set[tir.Var] = {prod_buf.data, *(buf.data for buf in out_bufs)}

        def _buffer_for_expr(expr: relax.Expr) -> Optional[tir.Buffer]:
            if isinstance(expr, relax.Var) and expr.same_as(producer.var):
                return prod_buf
            if expr not in external_buffers:
                sinfo = getattr(expr, "struct_info", None)
                if not isinstance(sinfo, TensorStructInfo):
                    return None
                buf = _tensor_buffer_from_sinfo(f"arg{len(external_exprs)}", sinfo)
                if buf is None:
                    return None
                external_exprs.append(expr)
                external_buffers[expr] = buf
                bound_data_vars.add(buf.data)
            return external_buffers[expr]

        def _append_rewritten_calls(
            binding: relax.VarBinding,
            output_buffers: Optional[List[tir.Buffer]] = None,
        ) -> Optional[List[tir.Stmt]]:
            value = binding.value
            if not isinstance(value, relax.Call) or value.op != call_tir_op:
                return None
            callee = value.args[0]
            if not isinstance(callee, tvm.ir.GlobalVar):
                return None
            callee_func = mod[callee]
            if not isinstance(callee_func, tir.PrimFunc):
                return None
            submit_calls = _extract_direct_rknpu_submit_calls(callee_func)
            inputs = _call_tir_inputs(value)
            if submit_calls is None or inputs is None:
                return None
            n_outputs = 1 if output_buffers is None else len(output_buffers)
            if len(callee_func.params) != len(inputs) + n_outputs:
                return None
            subst: Dict[tir.Var, tir.PrimExpr] = {}
            for inp_expr, param in zip(inputs, callee_func.params[:-n_outputs]):
                old_buf = callee_func.buffer_map[param]
                new_buf = _buffer_for_expr(inp_expr)
                if new_buf is None:
                    return None
                subst[old_buf.data] = new_buf.data
            if output_buffers is None:
                old_out_buf = callee_func.buffer_map[callee_func.params[-1]]
                subst[old_out_buf.data] = prod_buf.data
            else:
                for out_idx, out_buf in enumerate(output_buffers):
                    old_out_buf = callee_func.buffer_map[callee_func.params[len(inputs) + out_idx]]
                    subst[old_out_buf.data] = out_buf.data
            local_stmts: List[tir.Stmt] = []
            for alloc_idx, (old_var, (dtype, numel)) in enumerate(_extract_allocate_slots(callee_func).items()):
                new_var = tir.Var(
                    f"{callee.name_hint}_tmp{len(internal_allocs) + alloc_idx}",
                    tvm.ir.PointerType(tvm.ir.PrimType(dtype)),
                )
                subst[old_var] = new_var
                internal_allocs.append((new_var, dtype, numel))
            temp_slots: Dict[tir.Var, tuple[str, int]] = {}
            for submit_call in submit_calls:
                rewritten = tir.stmt_functor.substitute(submit_call, subst)
                inferred = _infer_stage_temp_slot(rewritten)
                if inferred is not None:
                    temp_var, dtype, numel = inferred
                    if temp_var not in bound_data_vars and temp_var not in temp_slots:
                        temp_slots[temp_var] = (dtype, numel)
                        internal_allocs.append((temp_var, dtype, numel))
                local_stmts.append(tir.Evaluate(rewritten))
            return local_stmts

        stmts: List[tir.Stmt] = []
        prod_stmts = _append_rewritten_calls(producer)
        cons_stmts = _append_rewritten_calls(consumer, out_bufs)
        if prod_stmts is None or cons_stmts is None:
            return None
        stmts.extend(prod_stmts)
        stmts.extend(cons_stmts)

        body: tir.Stmt = stmts[0] if len(stmts) == 1 else tir.SeqStmt(stmts)
        for storage, dtype, numel in reversed(internal_allocs):
            body = tir.Allocate(storage, dtype, [tir.IntImm("int64", numel)], tir.const(True, "bool"), body)
        body = _wrap_stmt_in_root_block(body)

        params: List = [*external_buffers.values(), *out_bufs]
        prim = tir.PrimFunc(
            params,
            body,
            attrs=tvm.ir.make_node(
                "ir.DictAttrs",
                **{
                    "tir.noalias": True,
                    "op_pattern": _K_OPAQUE,
                },
            ),
        )
        prim = prim.with_attr("private", tir.IntImm("int64", 1))
        prim = prim.with_attr("rknpu.deferred_tuple_projection_group", tir.IntImm("int64", 1))
        return prim, external_exprs, out_sinfos

    updates: Dict[tvm.ir.GlobalVar, relax.Function] = {}
    for gv, func in mod.functions_items():
        if not isinstance(func, relax.Function):
            continue
        if not isinstance(func.body, relax.SeqExpr):
            continue
        use_count = _collect_relax_var_use_counts(func)
        blocks = []
        changed = False
        for block in func.body.blocks:
            if not isinstance(block, relax.DataflowBlock):
                blocks.append(block)
                continue
            bindings = list(block.bindings)
            new_bindings: List[relax.Binding] = []
            i = 0
            while i < len(bindings):
                binding = bindings[i]
                if (
                    i + 1 < len(bindings)
                    and isinstance(binding, relax.VarBinding)
                    and isinstance(bindings[i + 1], relax.VarBinding)
                    and _is_rknpu_call_tir(mod, binding.value)
                    and _is_rknpu_call_tir(mod, bindings[i + 1].value)
                    and use_count.get(binding.var, 0) == 1
                ):
                    producer = binding
                    consumer = bindings[i + 1]
                    prod_value = producer.value
                    cons_value = consumer.value
                    if isinstance(prod_value, relax.Call) and isinstance(cons_value, relax.Call):
                        prod_callee = prod_value.args[0]
                        cons_callee = cons_value.args[0]
                        if isinstance(prod_callee, tvm.ir.GlobalVar) and isinstance(cons_callee, tvm.ir.GlobalVar):
                            prod_func = mod[prod_callee]
                            cons_func = mod[cons_callee]
                            prod_ids = _direct_rknpu_stage_ids(prod_func) if isinstance(prod_func, tir.PrimFunc) else None
                            cons_ids = _direct_rknpu_stage_ids(cons_func) if isinstance(cons_func, tir.PrimFunc) else None
                            cons_inputs = _call_tir_inputs(cons_value)
                            if (
                                prod_ids in ([1, 9, 9, 2], [9, 9, 2])
                                and cons_ids == [1, 2, 1, 2, 1, 2]
                                and cons_inputs is not None
                                and len(cons_inputs) >= 1
                                and isinstance(cons_inputs[0], relax.Var)
                                and cons_inputs[0].same_as(producer.var)
                            ):
                                built = _build_group_primfunc(producer, consumer)
                                if built is not None:
                                    prim, inputs, out_sinfos = built
                                    new_gv = bb.add_func(prim, bb.get_unique_name("fused_deferred_proj_group"))
                                    new_call = bb.normalize(call_tir(new_gv, inputs, out_sinfos))
                                    new_bindings.append(relax.VarBinding(consumer.var, new_call))
                                    changed = True
                                    i += 2
                                    continue
                new_bindings.append(binding)
                i += 1
            blocks.append(relax.DataflowBlock(new_bindings))
        if changed:
            updates[gv] = relax.Function(
                func.params,
                relax.SeqExpr(blocks, func.body.body),
                func.ret_struct_info,
                func.is_pure,
                func.attrs,
                func.span,
            )
    if not updates:
        return mod
    for gv, new_func in updates.items():
        bb.update_func(gv, new_func)
    return bb.get()


def _merge_sibling_projection_groups(mod: tvm.ir.IRModule) -> tvm.ir.IRModule:
    call_tir_op = tvm.ir.Op.get("relax.call_tir")
    bb = relax.BlockBuilder(mod)

    def _build_group_primfunc(
        group: List[relax.VarBinding],
    ) -> Optional[tuple[tir.PrimFunc, List[relax.Expr], List[TensorStructInfo]]]:
        out_bufs: List[tir.Buffer] = []
        out_sinfos: List[TensorStructInfo] = []
        for idx, binding in enumerate(group):
            sinfo = binding.var.struct_info
            if not isinstance(sinfo, TensorStructInfo):
                return None
            buf = _tensor_buffer_from_sinfo(f"out{idx}", sinfo)
            if buf is None:
                return None
            out_bufs.append(buf)
            out_sinfos.append(sinfo)

        external_exprs: List[relax.Expr] = []
        external_buffers: Dict[relax.Expr, tir.Buffer] = {}
        internal_allocs: List[tuple[tir.Var, str, int]] = []
        bound_data_vars: set[tir.Var] = {buf.data for buf in out_bufs}

        def _buffer_for_expr(expr: relax.Expr) -> Optional[tir.Buffer]:
            if expr not in external_buffers:
                sinfo = getattr(expr, "struct_info", None)
                if not isinstance(sinfo, TensorStructInfo):
                    return None
                buf = _tensor_buffer_from_sinfo(f"arg{len(external_exprs)}", sinfo)
                if buf is None:
                    return None
                external_exprs.append(expr)
                external_buffers[expr] = buf
                bound_data_vars.add(buf.data)
            return external_buffers[expr]

        stmts: List[tir.Stmt] = []
        temp_slots: Dict[tir.Var, tuple[str, int]] = {}
        for out_idx, binding in enumerate(group):
            value = binding.value
            if not isinstance(value, relax.Call) or value.op != call_tir_op:
                return None
            callee = value.args[0]
            if not isinstance(callee, tvm.ir.GlobalVar):
                return None
            callee_func = mod[callee]
            if not isinstance(callee_func, tir.PrimFunc):
                return None
            submit_calls = _extract_direct_rknpu_submit_calls(callee_func)
            inputs = _call_tir_inputs(value)
            if submit_calls is None or inputs is None:
                return None
            if len(callee_func.params) != len(inputs) + 1:
                return None
            subst: Dict[tir.Var, tir.PrimExpr] = {}
            for inp_expr, param in zip(inputs, callee_func.params[:-1]):
                old_buf = callee_func.buffer_map[param]
                new_buf = _buffer_for_expr(inp_expr)
                if new_buf is None:
                    return None
                subst[old_buf.data] = new_buf.data
            old_out_buf = callee_func.buffer_map[callee_func.params[-1]]
            subst[old_out_buf.data] = out_bufs[out_idx].data
            for alloc_idx, (old_var, (dtype, numel)) in enumerate(_extract_allocate_slots(callee_func).items()):
                new_var = tir.Var(
                    f"{callee.name_hint}_tmp{len(internal_allocs) + alloc_idx}",
                    tvm.ir.PointerType(tvm.ir.PrimType(dtype)),
                )
                subst[old_var] = new_var
                internal_allocs.append((new_var, dtype, numel))
            for submit_call in submit_calls:
                rewritten = tir.stmt_functor.substitute(submit_call, subst)
                inferred = _infer_stage_temp_slot(rewritten)
                if inferred is not None:
                    temp_var, dtype, numel = inferred
                    if temp_var not in bound_data_vars and temp_var not in temp_slots:
                        temp_slots[temp_var] = (dtype, numel)
                        internal_allocs.append((temp_var, dtype, numel))
                stmts.append(tir.Evaluate(rewritten))

        if not stmts:
            return None
        body: tir.Stmt = stmts[0] if len(stmts) == 1 else tir.SeqStmt(stmts)
        for storage, dtype, numel in reversed(internal_allocs):
            body = tir.Allocate(storage, dtype, [tir.IntImm("int64", numel)], tir.const(True, "bool"), body)
        body = _wrap_stmt_in_root_block(body)

        params: List = [*external_buffers.values(), *out_bufs]
        prim = tir.PrimFunc(
            params,
            body,
            attrs=tvm.ir.make_node(
                "ir.DictAttrs",
                **{
                    "tir.noalias": True,
                    "op_pattern": _K_OPAQUE,
                },
            ),
        )
        prim = prim.with_attr("private", tir.IntImm("int64", 1))
        prim = prim.with_attr("rknpu.sibling_projection_group", tir.IntImm("int64", len(group)))
        return prim, external_exprs, out_sinfos

    updates: Dict[tvm.ir.GlobalVar, relax.Function] = {}
    for gv, func in mod.functions_items():
        if not isinstance(func, relax.Function):
            continue
        if not isinstance(func.body, relax.SeqExpr):
            continue
        blocks = []
        changed = False
        for block in func.body.blocks:
            if not isinstance(block, relax.DataflowBlock):
                blocks.append(block)
                continue
            bindings = list(block.bindings)
            new_bindings: List[relax.Binding] = []
            i = 0
            while i < len(bindings):
                binding = bindings[i]
                if not isinstance(binding, relax.VarBinding) or not _is_rknpu_call_tir(mod, binding.value):
                    new_bindings.append(binding)
                    i += 1
                    continue
                value = binding.value
                if not isinstance(value, relax.Call):
                    new_bindings.append(binding)
                    i += 1
                    continue
                inputs = _call_tir_inputs(value)
                callee = value.args[0]
                if inputs is None or not isinstance(callee, tvm.ir.GlobalVar):
                    new_bindings.append(binding)
                    i += 1
                    continue
                callee_func = mod[callee]
                stage_ids = _direct_rknpu_stage_ids(callee_func) if isinstance(callee_func, tir.PrimFunc) else None
                if stage_ids != [1, 2] or len(inputs) != 3:
                    new_bindings.append(binding)
                    i += 1
                    continue
                base_input = inputs[0]
                group = [binding]
                j = i + 1
                while j < len(bindings):
                    nxt = bindings[j]
                    if not isinstance(nxt, relax.VarBinding) or not _is_rknpu_call_tir(mod, nxt.value):
                        break
                    nxt_value = nxt.value
                    if not isinstance(nxt_value, relax.Call):
                        break
                    nxt_inputs = _call_tir_inputs(nxt_value)
                    nxt_callee = nxt_value.args[0]
                    if nxt_inputs is None or len(nxt_inputs) != 3 or not isinstance(nxt_callee, tvm.ir.GlobalVar):
                        break
                    nxt_func = mod[nxt_callee]
                    nxt_stage_ids = _direct_rknpu_stage_ids(nxt_func) if isinstance(nxt_func, tir.PrimFunc) else None
                    if nxt_stage_ids != [1, 2]:
                        break
                    if not tvm.ir.structural_equal(nxt_inputs[0], base_input):
                        break
                    group.append(nxt)
                    j += 1
                if len(group) < 2:
                    new_bindings.append(binding)
                    i += 1
                    continue
                built = _build_group_primfunc(group)
                if built is None:
                    new_bindings.append(binding)
                    i += 1
                    continue
                prim, ext_inputs, out_sinfos = built
                new_gv = bb.add_func(prim, bb.get_unique_name("fused_sibling_proj"))
                tuple_call = bb.normalize(call_tir(new_gv, ext_inputs, out_sinfos))
                tuple_var = relax.Var(bb.get_unique_name("lv_proj_group"), tuple_call.struct_info_)
                new_bindings.append(relax.VarBinding(tuple_var, tuple_call))
                for idx, old_binding in enumerate(group):
                    item = bb.normalize(relax.TupleGetItem(tuple_var, idx))
                    new_bindings.append(relax.VarBinding(old_binding.var, item))
                changed = True
                i = j
            blocks.append(relax.DataflowBlock(new_bindings))
        if changed:
            updates[gv] = relax.Function(
                func.params,
                relax.SeqExpr(blocks, func.body.body),
                func.ret_struct_info,
                func.is_pure,
                func.attrs,
                func.span,
            )
    if not updates:
        return mod
    for gv, new_func in updates.items():
        bb.update_func(gv, new_func)
    return bb.get()


def lower_to_rknpu_tir_with_pc_chain(mod: tvm.ir.IRModule) -> tvm.ir.IRModule:
    """Lower to fused RKNPU TIR and apply experimental PC-chain submit lowering."""
    mod = lower_to_rknpu_tir(mod)
    mod = _merge_sibling_projection_groups(mod)
    mod = _merge_linear_rknpu_call_tir_groups(mod)
    mod = _merge_deferred_rknpu_call_tir_groups(mod)
    mod = _merge_deferred_tuple_projection_groups(mod)
    # Deferred merges can expose new linear-chain opportunities that were
    # previously blocked by transient fan-out.
    mod = _merge_linear_rknpu_call_tir_groups(mod)
    mod = relax.transform.NormalizeGlobalVar()(mod)
    mod = lower_pc_chain_submits()(mod)
    mod = attach_bridge_chain_blob_attr(mod)
    mod = attach_schedule_report_attr(mod)
    return mod


def _as_int(expr) -> Optional[int]:
    if isinstance(expr, tvm.tir.IntImm):
        return int(expr.value)
    return None


def _stage_arg_count(stage_id: int) -> int:
    if stage_id == 1:
        return 6  # matmul
    if stage_id == 2:
        return 6  # add
    if stage_id == 3:
        return 4  # relu 2d
    if stage_id == 4:
        return 6  # relu 4d
    if stage_id == 5:
        return 18  # conv2d
    if stage_id == 6:
        return 7  # matmul + bias + relu
    if stage_id == 7:
        return 6  # add + relu
    if stage_id == 8:
        return 18  # conv2d + relu
    if stage_id == 9:
        return 6  # mul
    if stage_id == 10:
        return 4  # exp
    if stage_id == 12:
        return 4  # reciprocal
    if stage_id == 13:
        return 4  # gelu lut
    if stage_id == 11:
        return 7  # matmul + bias
    raise ValueError(f"unsupported stage id: {stage_id}")


def _infer_reciprocal_max_domain(
    stages: List[tuple[int, List[tvm.tir.PrimExpr]]], stage_index: int, m: int, n: int
) -> float:
    """Infer reciprocal input domain from the producer stage when available."""
    if stage_index > 0:
        prev_sid, prev_payload = stages[stage_index - 1]
        # Softmax decomposition pattern: sum via matmul(M,K)x(K,1) -> reciprocal(M,1).
        if prev_sid == 1 and len(prev_payload) >= 6:
            prev_k = _as_int(prev_payload[4])
            prev_n = _as_int(prev_payload[5])
            if prev_k is not None and prev_k > 0 and prev_n == 1:
                return float(prev_k)
    # Fallback for singleton reciprocal(M,1): row sums are often bounded by
    # sequence length in attention paths (commonly matching M).
    if n == 1 and m > 0:
        return float(max(64, m))
    return 64.0


_BRIDGE_CHAIN_BLOB_V4_MAGIC = 0x34424352  # "RCB4"
_RELOC_WRITE_BACK = 1
_PTR_REGS = {0x1070, 0x4020, 0x5018, 0x5020, 0x5038, 0x6070, 0x701C}
_PLACEHOLDER_RELOC_BASES = {
    PLACEHOLDER_INPUT: RELOC_INPUT,
    PLACEHOLDER_WEIGHT: RELOC_WEIGHT,
    PLACEHOLDER_OUTPUT: RELOC_OUTPUT,
    PLACEHOLDER_BIAS: RELOC_BIAS,
}
_PLACEHOLDER_RELOC_MAX_DELTA = 0x08000000  # 128 MiB safety window per placeholder family


def _split_matmul_tiles(m: int, k: int, n: int) -> List[tuple[int, int, int, int]]:
    if m <= 0 or n <= 0:
        raise ValueError(f"invalid matmul dims m={m} n={n}")
    tile_task = AbstractMatmulTask(
        op_name="matmul_tile",
        M=m,
        K=k,
        N=n,
        precision="float16",
        relu=False,
        has_bias=False,
        output_fp16=True,
    )
    m_tile_size = int(tile_task.compute_m_tile_size())
    if m_tile_size <= 0:
        m_tile_size = m
    n_tile_size = int(compute_n_tile(n))
    if n_tile_size <= 0:
        n_tile_size = n
    out: List[tuple[int, int, int, int]] = []
    m_tiles: List[tuple[int, int]] = []
    if m > m_tile_size:
        m_off = 0
        while m_off < m:
            tile_m = min(m_tile_size, m - m_off)
            m_tiles.append((m_off, tile_m))
            m_off += tile_m
        # Avoid pathological tiny tail tiles like 1020+4 by rebalancing when
        # exactly two M tiles are needed. This keeps task count constant while
        # making both tasks substantial enough to amortize submit overhead.
        if len(m_tiles) == 2 and m_tiles[1][1] < 64 and (m % 4 == 0):
            first = (m // 2)
            first = (first // 4) * 4
            second = m - first
            if (
                first > 0
                and second > 0
                and first <= m_tile_size
                and second <= m_tile_size
                and first % 4 == 0
                and second % 4 == 0
            ):
                m_tiles = [(0, first), (first, second)]
    else:
        m_tiles = [(0, m)]
    for m_off, tile_m in m_tiles:
        for n_off in range(0, n, n_tile_size):
            tile_n = min(n_tile_size, n - n_off)
            out.append((m_off, tile_m, n_off, tile_n))
    return out


def _patch_task_reloc_deltas(task, reloc_deltas: Dict[int, int]) -> None:
    if not reloc_deltas:
        return
    patched = list(task.regcmds)
    for cmd_idx, reloc_type in _build_relocation_table(task):
        delta = int(reloc_deltas.get(reloc_type, 0))
        if delta == 0:
            continue
        cmd = patched[cmd_idx]
        new_value = int(cmd.value) + delta
        if new_value < 0 or new_value > 0xFFFFFFFF:
            raise RuntimeError(
                f"reloc delta overflow: cmd={cmd_idx} reloc_type={reloc_type} "
                f"value=0x{int(cmd.value):08x} delta={delta}"
            )
        patched[cmd_idx] = cmd.patch_value(new_value)
    task.regcmds = patched


def _extract_chained_submits_from_primfunc(
    func: tvm.tir.PrimFunc,
) -> List[List[tuple[int, List[tvm.tir.PrimExpr]]]]:
    call_extern = tvm.ir.Op.get("tir.call_extern")
    local_submits: List[List[tuple[int, List[tvm.tir.PrimExpr]]]] = []

    def _visit(node):
        if not isinstance(node, tvm.tir.Call):
            return
        if node.op != call_extern or len(node.args) < 2:
            return
        fn = node.args[0]
        if not isinstance(fn, tvm.tir.StringImm) or fn.value != "rknpu_submit_chain_stage_v2":
            return
        num_tasks = _as_int(node.args[1])
        if num_tasks is None or num_tasks <= 0:
            return
        cursor = 2
        stages: List[tuple[int, List[tvm.tir.PrimExpr]]] = []
        for _ in range(num_tasks):
            sid = _as_int(node.args[cursor])
            if sid is None:
                raise RuntimeError("non-constant stage id in chained submit")
            nargs = _stage_arg_count(sid)
            payload = list(node.args[cursor + 1 : cursor + 1 + nargs])
            stages.append((sid, payload))
            cursor += 1 + nargs
        local_submits.append(stages)

    tvm.tir.stmt_functor.post_order_visit(func.body, _visit)
    return local_submits


def _extract_chain_submits(
    mod: tvm.ir.IRModule,
) -> List[List[tuple[int, List[tvm.tir.PrimExpr]]]]:

    submits: List[List[tuple[int, List[tvm.tir.PrimExpr]]]] = []
    call_order = _collect_main_call_tir_order(mod)
    if call_order:
        for gv in call_order:
            func = mod[gv]
            if isinstance(func, tvm.tir.PrimFunc):
                submits.extend(_extract_chained_submits_from_primfunc(func))
        if submits:
            return submits

    for _, base_func in mod.functions.items():
        if isinstance(base_func, tvm.tir.PrimFunc):
            submits.extend(_extract_chained_submits_from_primfunc(base_func))
    return submits


_SUBMIT_NAME_TO_STAGE_ID: Dict[str, int] = {
    "rknpu_submit_matmul_stage": 1,
    "rknpu_submit_add_stage": 2,
    "rknpu_submit_mul_stage": 9,
    "rknpu_submit_exp_stage": 10,
    "rknpu_submit_reciprocal_stage": 12,
    "rknpu_submit_gelu_stage": 13,
    "rknpu_submit_relu_stage": 3,
    "rknpu_submit_relu_stage_4d": 4,
    "rknpu_submit_conv2d_stage": 5,
}


def _collect_main_call_tir_order(mod: tvm.ir.IRModule) -> List[tvm.ir.GlobalVar]:
    try:
        main_gv = mod.get_global_var("main")
    except Exception:  # pylint: disable=broad-exception-caught
        main_gv = None
    if main_gv is None or not isinstance(mod[main_gv], tvm.relax.Function):
        return []

    main_func = mod[main_gv]
    body = main_func.body
    call_tir_op = tvm.ir.Op.get("relax.call_tir")
    call_order: List[tvm.ir.GlobalVar] = []
    if isinstance(body, tvm.relax.SeqExpr):
        for block in body.blocks:
            for binding in block.bindings:
                if not isinstance(binding, tvm.relax.VarBinding):
                    continue
                value = binding.value
                if not isinstance(value, tvm.relax.Call):
                    continue
                if value.op != call_tir_op or not value.args:
                    continue
                callee = value.args[0]
                if isinstance(callee, tvm.ir.GlobalVar) and isinstance(mod[callee], tvm.tir.PrimFunc):
                    call_order.append(callee)
    return call_order


def _extract_direct_submit_singletons(
    func: tvm.tir.PrimFunc,
) -> List[List[tuple[int, List[tvm.tir.PrimExpr]]]]:
    calls = _extract_direct_rknpu_submit_calls(func)
    if calls is None:
        return []
    submits: List[List[tuple[int, List[tvm.tir.PrimExpr]]]] = []
    for call in calls:
        name = call.args[0]
        if not isinstance(name, tvm.tir.StringImm):
            continue
        sid = _SUBMIT_NAME_TO_STAGE_ID.get(name.value)
        if sid is None:
            continue
        submits.append([(sid, list(call.args[1:]))])
    return submits


def _extract_schedule_submits(
    mod: tvm.ir.IRModule,
) -> List[List[tuple[int, List[tvm.tir.PrimExpr]]]]:
    """Extract runtime submit groups for reporting.

    Unlike bridge-chain blob extraction, this includes direct single-stage
    submits when a PrimFunc contains no chained submit call.
    """
    submits: List[List[tuple[int, List[tvm.tir.PrimExpr]]]] = []
    call_order = _collect_main_call_tir_order(mod)
    if call_order:
        for gv in call_order:
            func = mod[gv]
            if not isinstance(func, tvm.tir.PrimFunc):
                continue
            chained = _extract_chained_submits_from_primfunc(func)
            if chained:
                submits.extend(chained)
                continue
            submits.extend(_extract_direct_submit_singletons(func))
        if submits:
            return submits

    for _, base_func in mod.functions.items():
        if not isinstance(base_func, tvm.tir.PrimFunc):
            continue
        chained = _extract_chained_submits_from_primfunc(base_func)
        if chained:
            submits.extend(chained)
            continue
        submits.extend(_extract_direct_submit_singletons(base_func))
    return submits


def _reloc_for_type(stage_id: int, reloc_type: int) -> tuple[int, int]:
    if stage_id == 1:  # matmul: a,b,c
        if reloc_type == RELOC_INPUT:
            return 0, 0
        if reloc_type == RELOC_WEIGHT:
            return 1, 0
        if reloc_type == RELOC_OUTPUT:
            return 2, _RELOC_WRITE_BACK
        return 0, 0
    if stage_id == 2:  # add: a,b,c
        if reloc_type == RELOC_INPUT:
            return 0, 0
        if reloc_type in (RELOC_WEIGHT, RELOC_BIAS):
            return 1, 0
        if reloc_type == RELOC_OUTPUT:
            return 2, _RELOC_WRITE_BACK
        return 0, 0
    if stage_id == 9:  # mul: a,b,c
        if reloc_type == RELOC_INPUT:
            return 0, 0
        if reloc_type in (RELOC_WEIGHT, RELOC_BIAS):
            return 1, 0
        if reloc_type == RELOC_OUTPUT:
            return 2, _RELOC_WRITE_BACK
        return 0, 0
    if stage_id == 10:  # exp: a,c
        if reloc_type == RELOC_INPUT:
            return 0, 0
        if reloc_type == RELOC_OUTPUT:
            return 1, _RELOC_WRITE_BACK
        return 0, 0
    if stage_id == 12:  # reciprocal: a,c
        if reloc_type == RELOC_INPUT:
            return 0, 0
        if reloc_type == RELOC_OUTPUT:
            return 1, _RELOC_WRITE_BACK
        return 0, 0
    if stage_id == 13:  # gelu lut: a,c
        if reloc_type == RELOC_INPUT:
            return 0, 0
        if reloc_type == RELOC_OUTPUT:
            return 1, _RELOC_WRITE_BACK
        return 0, 0
    if stage_id == 3:  # relu: a,c (generated as ewise with synthetic 2nd input)
        if reloc_type == RELOC_OUTPUT:
            return 1, _RELOC_WRITE_BACK
        return 0, 0
    if stage_id == 4:  # relu4d: a,c (generated as ewise with synthetic 2nd input)
        if reloc_type == RELOC_OUTPUT:
            return 1, _RELOC_WRITE_BACK
        return 0, 0
    if stage_id == 5:  # conv2d: data,weight,out
        if reloc_type == RELOC_INPUT:
            return 0, 0
        if reloc_type == RELOC_WEIGHT:
            return 1, 0
        if reloc_type == RELOC_OUTPUT:
            return 2, _RELOC_WRITE_BACK
        return 0, 0
    if stage_id == 8:  # conv2d+relu: data,weight,out
        if reloc_type == RELOC_INPUT:
            return 0, 0
        if reloc_type == RELOC_WEIGHT:
            return 1, 0
        if reloc_type == RELOC_OUTPUT:
            return 2, _RELOC_WRITE_BACK
        return 0, 0
    if stage_id == 6:  # matmul+bias+relu: a,b,bias,c
        if reloc_type == RELOC_INPUT:
            return 0, 0
        if reloc_type == RELOC_WEIGHT:
            return 1, 0
        if reloc_type == RELOC_BIAS:
            return 2, 0
        if reloc_type == RELOC_OUTPUT:
            return 3, _RELOC_WRITE_BACK
        return 0, 0
    if stage_id == 11:  # matmul+bias: a,b,bias,c
        if reloc_type == RELOC_INPUT:
            return 0, 0
        if reloc_type == RELOC_WEIGHT:
            return 1, 0
        if reloc_type == RELOC_BIAS:
            return 2, 0
        if reloc_type == RELOC_OUTPUT:
            return 3, _RELOC_WRITE_BACK
        return 0, 0
    if stage_id == 7:  # add+relu: a,b,c
        if reloc_type == RELOC_INPUT:
            return 0, 0
        if reloc_type in (RELOC_WEIGHT, RELOC_BIAS):
            return 1, 0
        if reloc_type == RELOC_OUTPUT:
            return 2, _RELOC_WRITE_BACK
        return 0, 0
    return 0, 0


def _infer_reloc_type_from_cmd_u64(cmd_u64: int) -> Optional[int]:
    # Packed regcmd uses [addr32 in bits 16..47][reg in bits 0..15].
    addr = int((cmd_u64 >> 16) & 0xFFFFFFFF)
    best = None
    for base, reloc_type in _PLACEHOLDER_RELOC_BASES.items():
        if addr < base:
            continue
        delta = addr - base
        if delta >= _PLACEHOLDER_RELOC_MAX_DELTA:
            continue
        if best is None or delta < best[0]:
            best = (delta, reloc_type)
    if best is None:
        return None
    return best[1]


def _task_to_record(task, stage_index: int, stage_id: int) -> dict:
    reg_u64 = [cmd.to_u64() for cmd in task.regcmds]
    reloc_by_cmd: Dict[int, tuple[int, int]] = {}
    for idx, reloc_type in _build_relocation_table(task):
        arg_index, flags = _reloc_for_type(stage_id, reloc_type)
        reloc_by_cmd[idx] = (arg_index, flags)
    for idx, u in enumerate(reg_u64):
        if idx in reloc_by_cmd:
            continue
        reg = int(u & 0xFFFF)
        if reg not in _PTR_REGS:
            continue
        reloc_type = _infer_reloc_type_from_cmd_u64(u)
        if reloc_type is None:
            continue
        arg_index, flags = _reloc_for_type(stage_id, reloc_type)
        reloc_by_cmd[idx] = (arg_index, flags)
    relocs = [(idx, reloc_by_cmd[idx][0], reloc_by_cmd[idx][1]) for idx in sorted(reloc_by_cmd)]
    blocks: List[str] = []
    seen_blocks: Set[str] = set()
    for cmd in task.regcmds:
        block_name = getattr(cmd.block, "name", None)
        if not isinstance(block_name, str) or not block_name or block_name == "PC":
            continue
        if block_name in seen_blocks:
            continue
        seen_blocks.add(block_name)
        blocks.append(block_name)
    partition = getattr(task, "_rknpu_partition", None)
    return {
        "stage_index": stage_index,
        "regcmd_blob": b"".join(struct.pack("<Q", u) for u in reg_u64),
        "enable_mask": int(task.enable_mask),
        "int_mask": int(task.int_mask),
        "int_clear": int(task.int_clear),
        "regcfg_amount": len(task.regcmds) - 4,
        "relocs": relocs,
        "blocks": blocks,
        "partition": partition if isinstance(partition, dict) else None,
    }


def _build_matmul_task_records(
    gen: RegCmdGenerator, op_name: str, m: int, k: int, n: int, relu: bool, has_bias: bool
) -> List:
    tasks: List = []
    full_task = AbstractMatmulTask(op_name=op_name, M=m, K=k, N=n, relu=relu)
    m_tile_size = int(full_task.compute_m_tile_size())
    n_tile_size = int(compute_n_tile(n))
    needs_m_tiling = m > m_tile_size
    needs_n_tiling = n > n_tile_size
    k_aligned = int(align_up(k, 32))
    m_padded_full = int(pad_m(m))
    dma_row_bytes = 8 * 2
    for m_off, tile_m, n_off, tile_n in _split_matmul_tiles(m, k, n):
        task = AbstractMatmulTask(op_name=op_name, M=tile_m, K=k, N=tile_n, relu=relu)
        if needs_m_tiling:
            task.is_mtile = True
            task.M_tile = tile_m
            task.M_full = m
            task.m_offset = 0
        if needs_n_tiling:
            task.is_ntile = True
            task.N_tile = tile_n
        tile_n_aligned = int(align_up(tile_n, 16))
        tile_m_padded = int(pad_m(tile_m))
        bias = (
            TensorHandle("bias", (tile_n,), "float32", tile_n_aligned * 4, dma_addr=PLACEHOLDER_BIAS)
            if has_bias
            else None
        )
        gen_task = gen.generate_matmul(
            task,
            TensorHandle(
                "a", (tile_m, k), "float16", tile_m_padded * k_aligned * 2, dma_addr=PLACEHOLDER_INPUT
            ),
            TensorHandle(
                "b", (k, tile_n), "float16", k_aligned * tile_n_aligned * 2, dma_addr=PLACEHOLDER_WEIGHT
            ),
            TensorHandle(
                "c",
                (tile_m, tile_n),
                "float16",
                tile_m_padded * tile_n_aligned * 2,
                dma_addr=PLACEHOLDER_OUTPUT,
            ),
            bias_handle=bias,
        )
        _patch_task_reloc_deltas(
            gen_task,
            {
                RELOC_INPUT: m_off * dma_row_bytes,
                RELOC_WEIGHT: n_off * k_aligned * 2,
                RELOC_OUTPUT: m_off * dma_row_bytes + n_off * m_padded_full * 2,
                RELOC_BIAS: n_off * 4,
            },
        )
        gen_task._rknpu_partition = {
            "rows": [int(m_off), int(m_off + tile_m)],
            "cols": [int(n_off), int(n_off + tile_n)],
        }
        tasks.append(gen_task)
    return tasks


def _json_shape(shape: Tuple[int, ...]) -> List[int]:
    return [int(x) for x in shape]


def _io_spec(
    role: str,
    shape: Tuple[int, ...],
    dtype: str = "float16",
    partition_axes: Optional[Tuple[Optional[str], ...]] = None,
) -> Dict[str, object]:
    return {
        "role": role,
        "dtype": dtype,
        "shape": _json_shape(shape),
        "partition_axes": list(partition_axes) if partition_axes is not None else None,
    }


def _stage_signature_for_payload(sid: int, payload) -> Optional[Dict[str, object]]:
    if sid in (1, 6, 11):
        offset = 3 if sid == 1 else 4
        m = _as_int(payload[offset])
        k = _as_int(payload[offset + 1])
        n = _as_int(payload[offset + 2])
        if None in (m, k, n):
            return None
        inputs = [
            _io_spec("lhs", (m, k), partition_axes=("rows", None)),
            _io_spec("rhs", (k, n), partition_axes=(None, "cols")),
        ]
        if sid in (6, 11):
            inputs.append(_io_spec("bias_cols", (n,), partition_axes=("cols",)))
        return {
            "rank": 2,
            "inputs": inputs,
            "output": _io_spec("output", (m, n), partition_axes=("rows", "cols")),
        }
    if sid in (2, 7, 9):
        m = _as_int(payload[3])
        n = _as_int(payload[4])
        b_mode = _as_int(payload[5])
        if None in (m, n, b_mode):
            return None
        if b_mode == 1:
            rhs = _io_spec("rhs_cols_bias", (n,), partition_axes=("cols",))
        elif b_mode == 2:
            rhs = _io_spec("rhs_rows_bias", (m, 1), partition_axes=("rows", None))
        else:
            rhs = _io_spec("rhs_tensor", (m, n), partition_axes=("rows", "cols"))
        return {
            "rank": 2,
            "inputs": [
                _io_spec("lhs_tensor", (m, n), partition_axes=("rows", "cols")),
                rhs,
            ],
            "output": _io_spec("output", (m, n), partition_axes=("rows", "cols")),
        }
    if sid in (3, 10, 12, 13):
        m = _as_int(payload[2])
        n = _as_int(payload[3])
        if None in (m, n):
            return None
        return {
            "rank": 2,
            "inputs": [_io_spec("input", (m, n), partition_axes=("rows", "cols"))],
            "output": _io_spec("output", (m, n), partition_axes=("rows", "cols")),
        }
    if sid == 4:
        n0 = _as_int(payload[2])
        ch = _as_int(payload[3])
        h = _as_int(payload[4])
        w = _as_int(payload[5])
        if None in (n0, ch, h, w):
            return None
        shape = (n0, ch, h, w)
        return {
            "rank": 4,
            "inputs": [_io_spec("input", shape)],
            "output": _io_spec("output", shape),
        }
    if sid in (5, 8):
        n0 = _as_int(payload[3])
        c = _as_int(payload[4])
        h = _as_int(payload[5])
        w = _as_int(payload[6])
        oc = _as_int(payload[7])
        kh = _as_int(payload[8])
        kw = _as_int(payload[9])
        oh = _as_int(payload[10])
        ow = _as_int(payload[11])
        if None in (n0, c, h, w, oc, kh, kw, oh, ow):
            return None
        return {
            "rank": 4,
            "inputs": [
                _io_spec("input", (n0, c, h, w)),
                _io_spec("weight", (oc, c, kh, kw)),
            ],
            "output": _io_spec("output", (n0, oc, oh, ow)),
        }
    return None


def _stage_task_summaries(stage_tasks: List[dict], task_start: int) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for local_index, task in enumerate(stage_tasks):
        summary: Dict[str, object] = {
            "submit_task_index": int(task_start + local_index),
            "stage_task_index": int(local_index),
            "reloc_count": int(len(task.get("relocs", []))),
            "regcfg_amount": int(task.get("regcfg_amount", 0)),
            "blocks": [str(x) for x in task.get("blocks", [])],
        }
        partition = task.get("partition")
        if isinstance(partition, dict):
            rows = partition.get("rows")
            cols = partition.get("cols")
            if isinstance(rows, list) and len(rows) == 2 and isinstance(cols, list) and len(cols) == 2:
                summary["partition"] = {
                    "rows": [int(rows[0]), int(rows[1])],
                    "cols": [int(cols[0]), int(cols[1])],
                }
        out.append(summary)
    return out


def _stage_partition_summary(task_summaries: List[Dict[str, object]]) -> Optional[Dict[str, List[List[int]]]]:
    rows: List[List[int]] = []
    cols: List[List[int]] = []
    seen_rows: Set[Tuple[int, int]] = set()
    seen_cols: Set[Tuple[int, int]] = set()
    for task in task_summaries:
        partition = task.get("partition")
        if not isinstance(partition, dict):
            continue
        raw_rows = partition.get("rows")
        raw_cols = partition.get("cols")
        if isinstance(raw_rows, list) and len(raw_rows) == 2:
            key = (int(raw_rows[0]), int(raw_rows[1]))
            if key not in seen_rows:
                seen_rows.add(key)
                rows.append([key[0], key[1]])
        if isinstance(raw_cols, list) and len(raw_cols) == 2:
            key = (int(raw_cols[0]), int(raw_cols[1]))
            if key not in seen_cols:
                seen_cols.add(key)
                cols.append([key[0], key[1]])
    if not rows and not cols:
        return None
    return {"rows": rows, "cols": cols}


def _stage_block_list(task_summaries: List[Dict[str, object]]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for task in task_summaries:
        for block in task.get("blocks", []):
            block_name = str(block)
            if block_name in seen:
                continue
            seen.add(block_name)
            out.append(block_name)
    return out


def _build_conv2d_task_record(
    gen: RegCmdGenerator,
    *,
    n0: int,
    c: int,
    h: int,
    w: int,
    oc: int,
    kh: int,
    kw: int,
    oh: int,
    ow: int,
    stride_h: int,
    stride_w: int,
    pad_top: int,
    pad_left: int,
    pad_bottom: int,
    pad_right: int,
    relu: bool = False,
):
    if n0 != 1:
        raise RuntimeError(f"conv2d stage only supports batch=1 in chain blob, got n={n0}")
    if stride_h != stride_w:
        raise RuntimeError(
            "conv2d stage requires equal h/w stride in chain blob: "
            f"stride_h={stride_h} stride_w={stride_w}"
        )
    task = AbstractConv2DTask(
        op_name="conv2d",
        C=c,
        H=h,
        W=w,
        N=oc,
        kH=kh,
        kW=kw,
        stride=stride_h,
        pad_top=pad_top,
        pad_left=pad_left,
        pad_bottom=pad_bottom,
        pad_right=pad_right,
        relu=relu,
        has_bias=False,
    )
    if task.H_out != oh or task.W_out != ow:
        raise RuntimeError(
            "conv2d output mismatch in chain blob: "
            f"expected ({task.H_out},{task.W_out}) from attrs but got ({oh},{ow})"
        )
    c_aligned = align_up(c, 32)
    k_eff_aligned = align_up(align_up(c, 32) * kh * kw, 32)
    n_aligned = align_up(oc, 16)
    m_padded = pad_m(oh * ow)
    return gen.generate_conv_mode0(
        task,
        TensorHandle("input", (c, h, w), "float16", c_aligned * h * w * 2, dma_addr=PLACEHOLDER_INPUT),
        TensorHandle(
            "weight",
            (oc, c, kh, kw),
            "float16",
            k_eff_aligned * n_aligned * 2,
            dma_addr=PLACEHOLDER_WEIGHT,
        ),
        TensorHandle("output", (oc, oh, ow), "float16", n_aligned * m_padded * 2, dma_addr=PLACEHOLDER_OUTPUT),
        bias_handle=None,
    )


def build_bridge_chain_blob(mod: tvm.ir.IRModule) -> bytes:
    """Build chain blob payload for runtime bridge from lowered chained TIR."""
    gen = RegCmdGenerator()
    submits = _extract_chain_submits(mod)
    submit_tasks: List[List[dict]] = []
    for stages in submits:
        tasks: List[dict] = []
        for stage_index, (sid, payload) in enumerate(stages):
            if sid == 1:
                m = _as_int(payload[3])
                k = _as_int(payload[4])
                n = _as_int(payload[5])
                if None in (m, k, n):
                    raise RuntimeError("matmul stage has dynamic dims")
                for t in _build_matmul_task_records(
                        gen, "matmul", m=m, k=k, n=n, relu=False, has_bias=False
                ):
                    tasks.append(_task_to_record(t, stage_index, sid))
            elif sid == 2:
                m = _as_int(payload[3])
                n = _as_int(payload[4])
                b_mode = _as_int(payload[5])
                if None in (m, n, b_mode):
                    raise RuntimeError("add stage has dynamic dims")
                b_shape = (n,) if b_mode == 1 else (m, 1) if b_mode == 2 else (m, n)
                b_elems = b_shape[0] * (b_shape[1] if len(b_shape) == 2 else 1)
                task = gen.generate_elementwise(
                    AbstractElementwiseTask(
                        op_name="add",
                        op_type="Add",
                        n_inputs=2,
                        shape=(m, n),
                        broadcast_b=False,
                    ),
                    TensorHandle("a", (m, n), "float16", m * n * 2, dma_addr=PLACEHOLDER_INPUT),
                    TensorHandle(
                        "b", b_shape, "float16", b_elems * 2, dma_addr=PLACEHOLDER_WEIGHT
                    ),
                    TensorHandle("c", (m, n), "float16", m * n * 2, dma_addr=PLACEHOLDER_OUTPUT),
                )
                tasks.append(_task_to_record(task, stage_index, sid))
            elif sid == 9:
                m = _as_int(payload[3])
                n = _as_int(payload[4])
                b_mode = _as_int(payload[5])
                if None in (m, n, b_mode):
                    raise RuntimeError("mul stage has dynamic dims")
                b_shape = (n,) if b_mode == 1 else (m, 1) if b_mode == 2 else (m, n)
                b_elems = b_shape[0] * (b_shape[1] if len(b_shape) == 2 else 1)
                task = gen.generate_elementwise(
                    AbstractElementwiseTask(
                        op_name="mul",
                        op_type="Mul",
                        n_inputs=2,
                        shape=(m, n),
                        broadcast_b=False,
                    ),
                    TensorHandle("a", (m, n), "float16", m * n * 2, dma_addr=PLACEHOLDER_INPUT),
                    TensorHandle(
                        "b", b_shape, "float16", b_elems * 2, dma_addr=PLACEHOLDER_WEIGHT
                    ),
                    TensorHandle("c", (m, n), "float16", m * n * 2, dma_addr=PLACEHOLDER_OUTPUT),
                )
                tasks.append(_task_to_record(task, stage_index, sid))
            elif sid == 10:
                m = _as_int(payload[2])
                n = _as_int(payload[3])
                if None in (m, n):
                    raise RuntimeError("exp stage has dynamic dims")
                task = gen.generate_lut_combined_task(
                    EXP_LE_TABLE,
                    EXP_LO_TABLE,
                    shape=(m, n),
                    src_dma=PLACEHOLDER_INPUT,
                    dst_dma=PLACEHOLDER_OUTPUT,
                    lut_params=EXP_LUT_PARAMS,
                )
                tasks.append(_task_to_record(task, stage_index, sid))
            elif sid == 12:
                m = _as_int(payload[2])
                n = _as_int(payload[3])
                if None in (m, n):
                    raise RuntimeError("reciprocal stage has dynamic dims")
                recip_le, recip_lo, recip_params = build_reciprocal_tables(
                    _infer_reciprocal_max_domain(stages, stage_index, m, n)
                )
                task = gen.generate_lut_combined_task(
                    recip_le,
                    recip_lo,
                    shape=(m, n),
                    src_dma=PLACEHOLDER_INPUT,
                    dst_dma=PLACEHOLDER_OUTPUT,
                    lut_params=recip_params,
                )
                tasks.append(_task_to_record(task, stage_index, sid))
            elif sid == 13:
                m = _as_int(payload[2])
                n = _as_int(payload[3])
                if None in (m, n):
                    raise RuntimeError("gelu stage has dynamic dims")
                # GELU stage: LUT sigmoid(1.702*x) then in-place mul with x.
                task = gen.generate_lut_combined_task(
                    GELU_LE_TABLE,
                    GELU_LO_TABLE,
                    shape=(m, n),
                    src_dma=PLACEHOLDER_INPUT,
                    dst_dma=PLACEHOLDER_OUTPUT,
                    lut_params=GELU_LUT_PARAMS,
                )
                tasks.append(_task_to_record(task, stage_index, sid))
                task = gen.generate_elementwise(
                    AbstractElementwiseTask(
                        op_name="gelu_mul",
                        op_type="Mul",
                        n_inputs=2,
                        shape=(m, n),
                        broadcast_b=False,
                    ),
                    TensorHandle("a", (m, n), "float16", m * n * 2, dma_addr=PLACEHOLDER_INPUT),
                    TensorHandle("b", (m, n), "float16", m * n * 2, dma_addr=PLACEHOLDER_OUTPUT),
                    TensorHandle("c", (m, n), "float16", m * n * 2, dma_addr=PLACEHOLDER_OUTPUT),
                )
                tasks.append(_task_to_record(task, stage_index, sid))
            elif sid == 3:
                m = _as_int(payload[2])
                n = _as_int(payload[3])
                if None in (m, n):
                    raise RuntimeError("relu stage has dynamic dims")
                task = gen.generate_elementwise(
                    AbstractElementwiseTask(
                        op_name="relu_noop",
                        op_type="Add",
                        n_inputs=2,
                        shape=(m, n),
                        broadcast_b=True,
                    ),
                    TensorHandle("a", (m, n), "float16", m * n * 2, dma_addr=PLACEHOLDER_INPUT),
                    TensorHandle("b", (n,), "float16", n * 2, dma_addr=PLACEHOLDER_WEIGHT),
                    TensorHandle("c", (m, n), "float16", m * n * 2, dma_addr=PLACEHOLDER_OUTPUT),
                )
                tasks.append(_task_to_record(task, stage_index, sid))
            elif sid == 4:
                n0 = _as_int(payload[2])
                ch = _as_int(payload[3])
                h = _as_int(payload[4])
                w = _as_int(payload[5])
                if None in (n0, ch, h, w):
                    raise RuntimeError("relu4d stage has dynamic dims")
                total = n0 * ch * h * w
                task = gen.generate_elementwise(
                    AbstractElementwiseTask(
                        op_name="relu4d_noop",
                        op_type="Add",
                        n_inputs=2,
                        shape=(total, 1),
                        broadcast_b=True,
                    ),
                    TensorHandle(
                        "a", (total, 1), "float16", total * 2, dma_addr=PLACEHOLDER_INPUT
                    ),
                    TensorHandle("b", (1,), "float16", 2, dma_addr=PLACEHOLDER_WEIGHT),
                    TensorHandle(
                        "c", (total, 1), "float16", total * 2, dma_addr=PLACEHOLDER_OUTPUT
                    ),
                )
                tasks.append(_task_to_record(task, stage_index, sid))
            elif sid == 5 or sid == 8:
                n0 = _as_int(payload[3])
                c = _as_int(payload[4])
                h = _as_int(payload[5])
                w = _as_int(payload[6])
                oc = _as_int(payload[7])
                kh = _as_int(payload[8])
                kw = _as_int(payload[9])
                oh = _as_int(payload[10])
                ow = _as_int(payload[11])
                stride_h = _as_int(payload[12])
                stride_w = _as_int(payload[13])
                pad_top = _as_int(payload[14])
                pad_left = _as_int(payload[15])
                pad_bottom = _as_int(payload[16])
                pad_right = _as_int(payload[17])
                if None in (
                    n0,
                    c,
                    h,
                    w,
                    oc,
                    kh,
                    kw,
                    oh,
                    ow,
                    stride_h,
                    stride_w,
                    pad_top,
                    pad_left,
                    pad_bottom,
                    pad_right,
                ):
                    raise RuntimeError("conv2d stage has dynamic dims")
                task = _build_conv2d_task_record(
                    gen,
                    n0=n0,
                    c=c,
                    h=h,
                    w=w,
                    oc=oc,
                    kh=kh,
                    kw=kw,
                    oh=oh,
                    ow=ow,
                    stride_h=stride_h,
                    stride_w=stride_w,
                    pad_top=pad_top,
                    pad_left=pad_left,
                    pad_bottom=pad_bottom,
                    pad_right=pad_right,
                    relu=(sid == 8),
                )
                tasks.append(_task_to_record(task, stage_index, sid))
            elif sid == 6:
                m = _as_int(payload[4])
                k = _as_int(payload[5])
                n = _as_int(payload[6])
                if None in (m, k, n):
                    raise RuntimeError("matmul+bias+relu stage has dynamic dims")
                raw = os.getenv("TVM_RKNPU_STAGE6_FUSED_PAYLOAD", "").lower()
                # Stage-6 semantic contract is fused matmul+bias+relu by default.
                use_fused_payload = raw not in ("0", "false", "no", "off")
                for t in _build_matmul_task_records(
                        gen,
                        "matmul_bias_relu",
                        m=m,
                        k=k,
                        n=n,
                        relu=use_fused_payload,
                        has_bias=use_fused_payload,
                ):
                    tasks.append(_task_to_record(t, stage_index, sid))
            elif sid == 11:
                m = _as_int(payload[4])
                k = _as_int(payload[5])
                n = _as_int(payload[6])
                if None in (m, k, n):
                    raise RuntimeError("matmul+bias stage has dynamic dims")
                for t in _build_matmul_task_records(
                        gen,
                        "matmul_bias",
                        m=m,
                        k=k,
                        n=n,
                        relu=False,
                        has_bias=True,
                ):
                    tasks.append(_task_to_record(t, stage_index, sid))
            elif sid == 7:
                m = _as_int(payload[3])
                n = _as_int(payload[4])
                bias_1d = _as_int(payload[5])
                if None in (m, n, bias_1d):
                    raise RuntimeError("add+relu stage has dynamic dims")
                b_shape = (n,) if bias_1d == 1 else (m, n)
                b_elems = b_shape[0] * (b_shape[1] if len(b_shape) == 2 else 1)
                task = gen.generate_elementwise(
                    AbstractElementwiseTask(
                        op_name="add_relu",
                        op_type="Add",
                        n_inputs=2,
                        shape=(m, n),
                        broadcast_b=False,
                    ),
                    TensorHandle("a", (m, n), "float16", m * n * 2, dma_addr=PLACEHOLDER_INPUT),
                    TensorHandle(
                        "b", b_shape, "float16", b_elems * 2, dma_addr=PLACEHOLDER_WEIGHT
                    ),
                    TensorHandle("c", (m, n), "float16", m * n * 2, dma_addr=PLACEHOLDER_OUTPUT),
                )
                tasks.append(_task_to_record(task, stage_index, sid))
            else:
                task = generate_ppu_task(0)
                tasks.append(_task_to_record(task, stage_index, sid))
        submit_tasks.append(tasks)

    blob = bytearray()
    blob += struct.pack("<II", _BRIDGE_CHAIN_BLOB_V4_MAGIC, len(submit_tasks))
    for stages, tasks in zip(submits, submit_tasks):
        blob += struct.pack("<II", len(stages), len(tasks))
        for t in tasks:
            regcmd_blob = t["regcmd_blob"]
            blob += struct.pack(
                "<IIIIIII",
                t["stage_index"],
                len(regcmd_blob) // 8,
                t["enable_mask"],
                t["int_mask"],
                t["int_clear"],
                t["regcfg_amount"],
                len(t["relocs"]),
            )
            for cmd_index, arg_index, flags in t["relocs"]:
                blob += struct.pack("<IHH", cmd_index, arg_index, flags)
            blob += regcmd_blob
    return bytes(blob)


def build_rknpu_schedule_report(mod: tvm.ir.IRModule) -> Dict:
    """Build a compile-time schedule report for chained submits.

    The report is intentionally compact but includes:
    - Objective fields for phase-2 regression checks.
    - Submit/stage/task structure and reloc totals.
    - Tiling signals for matmul-like stages.
    - Relax allocation visibility signals from the current module text.
    """
    stage_names = {
        1: "matmul",
        2: "add",
        9: "mul",
        10: "exp",
        12: "reciprocal",
        13: "gelu",
        3: "relu",
        4: "relu4d",
        5: "conv2d",
        6: "matmul_bias_relu",
        11: "matmul_bias",
        7: "add_relu",
        8: "conv2d_relu",
    }
    submits = _extract_schedule_submits(mod)
    submit_tasks: List[List[dict]] = []
    submit_stage_reports: List[List[Dict]] = []
    submit_stage_ids: List[List[int]] = []
    report_gen = RegCmdGenerator()
    for stages in submits:
        tasks: List[dict] = []
        stage_report: List[Dict] = []
        stage_ids: List[int] = []
        task_cursor = 0
        for stage_index, (sid, payload) in enumerate(stages):
            stage_ids.append(sid)
            stage_tasks: List[dict] = []
            if sid == 1:
                m = _as_int(payload[3])
                k = _as_int(payload[4])
                n = _as_int(payload[5])
                if None in (m, k, n):
                    continue
                for t in _build_matmul_task_records(
                        report_gen, "matmul", m=m, k=k, n=n, relu=False, has_bias=False
                ):
                    stage_tasks.append(_task_to_record(t, stage_index, sid))
            elif sid == 2:
                m = _as_int(payload[3])
                n = _as_int(payload[4])
                b_mode = _as_int(payload[5])
                if None in (m, n, b_mode):
                    continue
                b_shape = (n,) if b_mode == 1 else (m, 1) if b_mode == 2 else (m, n)
                b_elems = b_shape[0] * (b_shape[1] if len(b_shape) == 2 else 1)
                t = report_gen.generate_elementwise(
                    AbstractElementwiseTask(
                        op_name="add",
                        op_type="Add",
                        n_inputs=2,
                        shape=(m, n),
                        broadcast_b=False,
                    ),
                    TensorHandle("a", (m, n), "float16", m * n * 2, dma_addr=PLACEHOLDER_INPUT),
                    TensorHandle("b", b_shape, "float16", b_elems * 2, dma_addr=PLACEHOLDER_WEIGHT),
                    TensorHandle("c", (m, n), "float16", m * n * 2, dma_addr=PLACEHOLDER_OUTPUT),
                )
                stage_tasks.append(_task_to_record(t, stage_index, sid))
            elif sid == 9:
                m = _as_int(payload[3])
                n = _as_int(payload[4])
                b_mode = _as_int(payload[5])
                if None in (m, n, b_mode):
                    continue
                b_shape = (n,) if b_mode == 1 else (m, 1) if b_mode == 2 else (m, n)
                b_elems = b_shape[0] * (b_shape[1] if len(b_shape) == 2 else 1)
                t = report_gen.generate_elementwise(
                    AbstractElementwiseTask(
                        op_name="mul",
                        op_type="Mul",
                        n_inputs=2,
                        shape=(m, n),
                        broadcast_b=False,
                    ),
                    TensorHandle("a", (m, n), "float16", m * n * 2, dma_addr=PLACEHOLDER_INPUT),
                    TensorHandle("b", b_shape, "float16", b_elems * 2, dma_addr=PLACEHOLDER_WEIGHT),
                    TensorHandle("c", (m, n), "float16", m * n * 2, dma_addr=PLACEHOLDER_OUTPUT),
                )
                stage_tasks.append(_task_to_record(t, stage_index, sid))
            elif sid == 3:
                m = _as_int(payload[2])
                n = _as_int(payload[3])
                if None in (m, n):
                    continue
                t = report_gen.generate_elementwise(
                    AbstractElementwiseTask(
                        op_name="relu_noop",
                        op_type="Add",
                        n_inputs=2,
                        shape=(m, n),
                        broadcast_b=True,
                    ),
                    TensorHandle("a", (m, n), "float16", m * n * 2, dma_addr=PLACEHOLDER_INPUT),
                    TensorHandle("b", (n,), "float16", n * 2, dma_addr=PLACEHOLDER_WEIGHT),
                    TensorHandle("c", (m, n), "float16", m * n * 2, dma_addr=PLACEHOLDER_OUTPUT),
                )
                stage_tasks.append(_task_to_record(t, stage_index, sid))
            elif sid == 4:
                n0 = _as_int(payload[2])
                ch = _as_int(payload[3])
                h = _as_int(payload[4])
                w = _as_int(payload[5])
                if None in (n0, ch, h, w):
                    continue
                total = n0 * ch * h * w
                t = report_gen.generate_elementwise(
                    AbstractElementwiseTask(
                        op_name="relu4d_noop",
                        op_type="Add",
                        n_inputs=2,
                        shape=(total, 1),
                        broadcast_b=True,
                    ),
                    TensorHandle("a", (total, 1), "float16", total * 2, dma_addr=PLACEHOLDER_INPUT),
                    TensorHandle("b", (1,), "float16", 2, dma_addr=PLACEHOLDER_WEIGHT),
                    TensorHandle("c", (total, 1), "float16", total * 2, dma_addr=PLACEHOLDER_OUTPUT),
                )
                stage_tasks.append(_task_to_record(t, stage_index, sid))
            elif sid == 10:
                m = _as_int(payload[2])
                n = _as_int(payload[3])
                if None in (m, n):
                    continue
                t = report_gen.generate_lut_combined_task(
                    EXP_LE_TABLE,
                    EXP_LO_TABLE,
                    shape=(m, n),
                    src_dma=PLACEHOLDER_INPUT,
                    dst_dma=PLACEHOLDER_OUTPUT,
                    lut_params=EXP_LUT_PARAMS,
                )
                stage_tasks.append(_task_to_record(t, stage_index, sid))
            elif sid == 12:
                m = _as_int(payload[2])
                n = _as_int(payload[3])
                if None in (m, n):
                    continue
                recip_le, recip_lo, recip_params = build_reciprocal_tables(
                    _infer_reciprocal_max_domain(stages, stage_index, m, n)
                )
                t = report_gen.generate_lut_combined_task(
                    recip_le,
                    recip_lo,
                    shape=(m, n),
                    src_dma=PLACEHOLDER_INPUT,
                    dst_dma=PLACEHOLDER_OUTPUT,
                    lut_params=recip_params,
                )
                stage_tasks.append(_task_to_record(t, stage_index, sid))
            elif sid == 13:
                m = _as_int(payload[2])
                n = _as_int(payload[3])
                if None in (m, n):
                    continue
                t = report_gen.generate_lut_combined_task(
                    GELU_LE_TABLE,
                    GELU_LO_TABLE,
                    shape=(m, n),
                    src_dma=PLACEHOLDER_INPUT,
                    dst_dma=PLACEHOLDER_OUTPUT,
                    lut_params=GELU_LUT_PARAMS,
                )
                stage_tasks.append(_task_to_record(t, stage_index, sid))
                t = report_gen.generate_elementwise(
                    AbstractElementwiseTask(
                        op_name="gelu_mul",
                        op_type="Mul",
                        n_inputs=2,
                        shape=(m, n),
                        broadcast_b=False,
                    ),
                    TensorHandle("a", (m, n), "float16", m * n * 2, dma_addr=PLACEHOLDER_INPUT),
                    TensorHandle("b", (m, n), "float16", m * n * 2, dma_addr=PLACEHOLDER_OUTPUT),
                    TensorHandle("c", (m, n), "float16", m * n * 2, dma_addr=PLACEHOLDER_OUTPUT),
                )
                stage_tasks.append(_task_to_record(t, stage_index, sid))
            elif sid == 5 or sid == 8:
                n0 = _as_int(payload[3])
                c = _as_int(payload[4])
                h = _as_int(payload[5])
                w = _as_int(payload[6])
                oc = _as_int(payload[7])
                kh = _as_int(payload[8])
                kw = _as_int(payload[9])
                oh = _as_int(payload[10])
                ow = _as_int(payload[11])
                stride_h = _as_int(payload[12])
                stride_w = _as_int(payload[13])
                pad_top = _as_int(payload[14])
                pad_left = _as_int(payload[15])
                pad_bottom = _as_int(payload[16])
                pad_right = _as_int(payload[17])
                if None in (
                    n0,
                    c,
                    h,
                    w,
                    oc,
                    kh,
                    kw,
                    oh,
                    ow,
                    stride_h,
                    stride_w,
                    pad_top,
                    pad_left,
                    pad_bottom,
                    pad_right,
                ):
                    continue
                t = _build_conv2d_task_record(
                    report_gen,
                    n0=n0,
                    c=c,
                    h=h,
                    w=w,
                    oc=oc,
                    kh=kh,
                    kw=kw,
                    oh=oh,
                    ow=ow,
                    stride_h=stride_h,
                    stride_w=stride_w,
                    pad_top=pad_top,
                    pad_left=pad_left,
                    pad_bottom=pad_bottom,
                    pad_right=pad_right,
                    relu=(sid == 8),
                )
                stage_tasks.append(_task_to_record(t, stage_index, sid))
            elif sid == 6:
                m = _as_int(payload[4])
                k = _as_int(payload[5])
                n = _as_int(payload[6])
                if None in (m, k, n):
                    continue
                raw = os.getenv("TVM_RKNPU_STAGE6_FUSED_PAYLOAD", "").lower()
                use_fused_payload = raw not in ("0", "false", "no", "off")
                for t in _build_matmul_task_records(
                        report_gen,
                        "matmul_bias_relu",
                        m=m,
                        k=k,
                        n=n,
                        relu=use_fused_payload,
                        has_bias=use_fused_payload,
                ):
                    stage_tasks.append(_task_to_record(t, stage_index, sid))
            elif sid == 11:
                m = _as_int(payload[4])
                k = _as_int(payload[5])
                n = _as_int(payload[6])
                if None in (m, k, n):
                    continue
                for t in _build_matmul_task_records(
                        report_gen,
                        "matmul_bias",
                        m=m,
                        k=k,
                        n=n,
                        relu=False,
                        has_bias=True,
                ):
                    stage_tasks.append(_task_to_record(t, stage_index, sid))
            elif sid == 7:
                m = _as_int(payload[3])
                n = _as_int(payload[4])
                bias_1d = _as_int(payload[5])
                if None in (m, n, bias_1d):
                    continue
                b_shape = (n,) if bias_1d == 1 else (m, n)
                b_elems = b_shape[0] * (b_shape[1] if len(b_shape) == 2 else 1)
                t = report_gen.generate_elementwise(
                    AbstractElementwiseTask(
                        op_name="add_relu",
                        op_type="Add",
                        n_inputs=2,
                        shape=(m, n),
                        broadcast_b=False,
                    ),
                    TensorHandle("a", (m, n), "float16", m * n * 2, dma_addr=PLACEHOLDER_INPUT),
                    TensorHandle("b", b_shape, "float16", b_elems * 2, dma_addr=PLACEHOLDER_WEIGHT),
                    TensorHandle("c", (m, n), "float16", m * n * 2, dma_addr=PLACEHOLDER_OUTPUT),
                )
                stage_tasks.append(_task_to_record(t, stage_index, sid))
            else:
                stage_tasks.append({"stage_index": stage_index, "relocs": []})
            stage_signature = _stage_signature_for_payload(sid, payload)
            stage_task_summaries = _stage_task_summaries(stage_tasks, task_cursor)
            stage_partition_summary = _stage_partition_summary(stage_task_summaries)
            stage_blocks = _stage_block_list(stage_task_summaries)
            is_partitioned = bool(
                stage_partition_summary
                and (
                    len(stage_partition_summary.get("rows", [])) > 1
                    or len(stage_partition_summary.get("cols", [])) > 1
                )
            )
            tasks.extend(stage_tasks)
            stage_report.append(
                {
                    "stage_index": stage_index,
                    "stage_id": sid,
                    "stage_name": stage_names.get(sid, f"stage_{sid}"),
                    "task_count": len(stage_tasks),
                    "reloc_count": int(sum(len(t.get("relocs", [])) for t in stage_tasks)),
                    "is_tiled": len(stage_tasks) > 1,
                    "is_partitioned": is_partitioned,
                    "task_index_range": [int(task_cursor), int(task_cursor + len(stage_tasks))],
                    "signature": stage_signature,
                    "blocks": stage_blocks,
                    "partition_summary": stage_partition_summary,
                    "task_summaries": stage_task_summaries,
                }
            )
            task_cursor += len(stage_tasks)
        submit_tasks.append(tasks)
        submit_stage_reports.append(stage_report)
        submit_stage_ids.append(stage_ids)

    submit_stage_counts = [len(stages) for stages in submits]
    submit_task_counts = [len(tasks) for tasks in submit_tasks]
    submit_reloc_counts = [[len(t.get("relocs", [])) for t in tasks] for tasks in submit_tasks]
    total_tasks = int(sum(submit_task_counts))
    total_reloc_entries = int(sum(sum(x) for x in submit_reloc_counts))
    num_tiled_stages = int(
        sum(1 for submit in submit_stage_reports for stage in submit if stage["is_tiled"])
    )

    text = mod.script()
    alloc_storage_count = (
        text.count("R.memory.alloc_storage(")
        + text.count("R.builtin.alloc_storage(")
    )
    alloc_tensor_count = text.count("R.memory.alloc_tensor(") + text.count("R.builtin.alloc_tensor(")

    compat_blocked_count = 0
    compat_blocked_boundaries: List[Dict[str, object]] = []
    for _, maybe_prim in mod.functions_items():
        if not isinstance(maybe_prim, tir.PrimFunc):
            continue
        attrs = maybe_prim.attrs
        if attrs is None:
            continue
        raw_count = attrs.get("rknpu.pc_chain_compat_blocked_count")
        if isinstance(raw_count, tir.IntImm):
            compat_blocked_count += int(raw_count.value)
        raw_json = attrs.get("rknpu.pc_chain_compat_blocked_json")
        if isinstance(raw_json, tvm.runtime.String):
            try:
                parsed = json.loads(str(raw_json))
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict):
                            compat_blocked_boundaries.append(item)
            except Exception:  # pylint: disable=broad-exception-caught
                continue
    return {
        "objective": {
            "minimize_submit_count": True,
            "maximize_pc_chaining_per_submit": True,
            "avoid_redundant_boundary_transforms": True,
            "maximize_planner_buffer_reuse": True,
        },
        "num_submits": len(submits),
        "submit_stage_counts": submit_stage_counts,
        "submit_task_counts": submit_task_counts,
        "submit_stage_ids": submit_stage_ids,
        "submit_reloc_counts": submit_reloc_counts,
        "submit_stage_reports": submit_stage_reports,
        "num_tiled_stages": num_tiled_stages,
        "total_tasks": total_tasks,
        "total_reloc_entries": total_reloc_entries,
        "fused_groups": [
            {"submit_index": i, "stage_ids": stage_ids}
            for i, stage_ids in enumerate(submit_stage_ids)
        ],
        "chained_groups": [
            {
                "submit_index": i,
                "stage_count": submit_stage_counts[i],
                "task_count": submit_task_counts[i],
            }
            for i in range(len(submit_stage_counts))
        ],
        "planner_alloc_signals": {
            "alloc_storage_count": int(alloc_storage_count),
            "alloc_tensor_count": int(alloc_tensor_count),
        },
        "chain_compatibility": {
            "blocked_boundary_count": int(compat_blocked_count),
            "blocked_boundaries": compat_blocked_boundaries,
        },
    }


def attach_bridge_chain_blob_attr(mod: tvm.ir.IRModule) -> tvm.ir.IRModule:
    """Attach base64-encoded chain blob metadata for runtime bridge preload."""
    blob = build_bridge_chain_blob(mod)
    if not blob:
        return mod
    b64 = base64.b64encode(blob).decode("ascii")
    return mod.with_attr("rknpu.bridge_chain_blob_b64", tvm.runtime.String(b64))


def attach_schedule_report_attr(mod: tvm.ir.IRModule) -> tvm.ir.IRModule:
    """Attach JSON schedule report metadata for diagnostics/regression checks."""
    report = build_rknpu_schedule_report(mod)
    text = json.dumps(report, separators=(",", ":"))
    return mod.with_attr("rknpu.schedule_report_json", tvm.runtime.String(text))


def plan_rknpu_tir_memory(mod: tvm.ir.IRModule) -> tvm.ir.IRModule:
    """Run the Relax memory planning stages on an already call_tir-based module."""
    seq = tvm.transform.Sequential(
        [
            relax.transform.RewriteDataflowReshape(),
            relax.transform.ToNonDataflow(),
            relax.transform.RemovePurityChecking(),
            relax.transform.CallTIRRewrite(),
            relax.transform.StaticPlanBlockMemory(),
            DedicateRKNPUPlannedStorage(),
        ]
    )
    return seq(mod)


@tvm.transform.module_pass(opt_level=0, name="DedicateRKNPUPlannedStorage")
class DedicateRKNPUPlannedStorage:
    """Replace planned storage reuse with dedicated storage per alloc_tensor.

    The integrated encoder block is currently correctness-sensitive to Relax
    planner storage reuse under real submit. Keep planner-style alloc_storage /
    alloc_tensor IR so the runtime bridge still exercises the real path, but
    remove aliasing between planned tensors until lifetimes are validated.
    """

    def transform_module(self, mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext) -> tvm.ir.IRModule:
        return _DedicatePlannedStorageRewriter(mod).transform()


@mutator
class _DedicatePlannedStorageRewriter(PyExprMutator):
    def __init__(self, mod: tvm.ir.IRModule) -> None:
        super().__init__(mod)
        self.mod = mod
        self.memory_alloc_storage_op = tvm.ir.Op.get("relax.memory.alloc_storage")
        self.memory_alloc_tensor_op = tvm.ir.Op.get("relax.memory.alloc_tensor")

    def transform(self) -> tvm.ir.IRModule:
        for g_var, func in self.mod.functions_items():
            if isinstance(func, relax.Function):
                self.builder_.update_func(g_var, self.visit_expr(func))
        return self.builder_.get()

    def visit_call_(self, call: relax.Call) -> relax.Expr:  # pylint: disable=arguments-renamed
        call = super().visit_call_(call)
        if not isinstance(call, relax.Call) or call.op != self.memory_alloc_tensor_op:
            return call
        return self._rewrite_alloc_tensor(call)

    def _rewrite_alloc_tensor(self, call: relax.Call) -> relax.Expr:
        sinfo = call.struct_info
        if not isinstance(sinfo, TensorStructInfo):
            return call
        shape = _shape_as_ints(sinfo)
        if shape is None:
            return call
        dtype = sinfo.dtype
        item_bytes = tvm.runtime.DataType(dtype).bits // 8
        numel = 1
        for dim in shape:
            numel *= dim
        nbytes = numel * item_bytes

        scope = relax.StringImm("global")
        storage_dtype = relax.DataTypeImm(dtype)
        if isinstance(call.args[0], relax.Var):
            bound = self.lookup_binding(call.args[0])
            if isinstance(bound, relax.Call) and bound.op == self.memory_alloc_storage_op:
                if len(bound.args) >= 4:
                    storage_dtype = bound.args[3]
                if len(bound.args) >= 3:
                    scope = bound.args[2]

        new_storage = relax.Call(
            self.memory_alloc_storage_op,
            args=[
                relax.ShapeExpr([tir.IntImm("int64", nbytes)]),
                relax.PrimValue(tir.IntImm("int64", 0)),
                scope,
                storage_dtype,
            ],
            sinfo_args=[relax.ObjectStructInfo()],
        )
        return relax.Call(
            self.memory_alloc_tensor_op,
            args=[new_storage, call.args[1], call.args[2], call.args[3], call.args[4]],
            sinfo_args=call.sinfo_args,
        )
