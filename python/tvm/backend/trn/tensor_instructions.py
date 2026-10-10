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
"""Trainium tensor instructions with explicit NKI instruction identity."""

from tvm.ir import Expr, LambdaExpr, PrimType, Tuple, const
from tvm.tirx.tensor_instruction import Instruction, Operand, namespace


def _copy(view, sctx):
    from .tile_primitive.copy.default import copy_trn

    dst, src = view.args
    name = view.op.name.rsplit(".", 1)[-1]
    expected = (
        "load"
        if src.source.scope() == "global"
        else "store"
        if dst.source.scope() == "global"
        else "tensor_copy"
    )
    if name != expected:
        raise ValueError(f"{name} does not accept this memory direction; use {expected}")
    return copy_trn(view, sctx)


def _matmul(view, sctx):
    from .tile_primitive.gemm.default import matmul_trn

    return matmul_trn(view, sctx)


def _unary(view, sctx):
    from tvm.tirx.operator.tile_primitive.common import MapOpType

    from .tile_primitive.unary.default import unary_trn
    from .tile_primitive.unary.with_bias_scale import unary_with_bias_scale_trn

    if view.kind.endswith("_with_scale_bias"):
        opcode = view.call.attrs.opcode
        return unary_with_bias_scale_trn(view, MapOpType[opcode.upper()], sctx)
    return unary_trn(view, MapOpType.FILL if view.kind == "memset" else MapOpType.RECIPROCAL, sctx)


def _binary(view, sctx):
    from tvm.tirx.operator.tile_primitive.common import MapOpType

    from .tile_primitive.binary.default import binary_trn

    return binary_trn(view, MapOpType[view.call.attrs.opcode.upper()], sctx)


def _reduce(view, sctx):
    from tvm.tirx.operator.tile_primitive.common import ReduceOpType

    from .tile_primitive.reduction.utils import reduction_trn

    return reduction_trn(
        view, ReduceOpType[view.call.attrs.reduce_op.upper()], sctx, negate=view.call.attrs.negate
    )


def _chain(view, sctx):
    from .tile_primitive.compose_op.binary_chain import binary_chain_trn

    return binary_chain_trn(view, sctx)


def _binary_reduce(view, sctx):
    from .tile_primitive.compose_op.binary_reduce import binary_reduce_trn

    return binary_reduce_trn(view, sctx)


def _activation_reduce(view, sctx):
    from .tile_primitive.compose_op.unary_reduce import unary_reduce_trn

    return unary_reduce_trn(view, sctx)


def _select(view, sctx):
    from .tile_primitive.select.default import select_trn

    return select_trn(view, sctx)


def _axes(attrs):
    return Tuple([const(i) for i in attrs.axes])


def make_namespace():
    def spec(name, kind, operands, options=None, workspaces=(), project=None, lower=None):
        return Instruction(
            "tirx.trn.tile." + name,
            kind,
            tuple(operands) + tuple(Operand(w, "tensor", None) for w in workspaces),
            "TrnAttrs",
            options or {},
            workspaces,
            project,
            lower,
        )

    pair = [Operand("dst", "region"), Operand("src", "region")]
    result = [
        spec(
            name, "copy", pair, {"max_inst_size": 512} if name == "tensor_copy" else {}, lower=_copy
        )
        for name in ("load", "store", "tensor_copy")
    ]
    result += [
        spec(
            "matmul",
            "gemm",
            [
                Operand("D", "region"),
                Operand("A", "region"),
                Operand("B", "region"),
                Operand("C", "region", "@dst"),
                Operand("transpose_A", "scalar", False),
                Operand("transpose_B", "scalar", False),
                Operand("alpha", "scalar", 1.0),
                Operand("beta", "scalar", 0.0),
            ],
            workspaces=("acc_psum",),
            lower=_matmul,
        )
    ]
    for name in ("reciprocal", "memset"):
        result += [
            spec(
                name,
                name,
                [Operand("dst", "region"), Operand("src")],
                {"max_inst_size": 512},
                lower=_unary,
            )
        ]

    def activation_project(v, a):
        return a.opcode + "_with_scale_bias", [v["dst"], v["src"], v["scale"], v["bias"]], {}

    result += [
        spec(
            "activation",
            "exp_with_scale_bias",
            [*pair, Operand("scale", "scalar", 1.0), Operand("bias", "expr", 0.0)],
            {"opcode": "exp", "max_inst_size": 512},
            ("const_bias",),
            activation_project,
            _unary,
        )
    ]

    def binary_project(v, a):
        return (
            {"max": "maximum", "min": "minimum"}.get(a.opcode, a.opcode),
            [v["dst"], v["lhs"], v["rhs"]],
            {},
        )

    for name in ("tensortensor", "tensorscalar"):
        result += [
            spec(
                name,
                "add",
                [Operand("dst", "region"), Operand("lhs"), Operand("rhs")],
                {"opcode": "add", "max_inst_size": 512},
                project=binary_project,
                lower=_binary,
            )
        ]

    def reduce_project(v, a):
        return a.reduce_op, [v["dst"], v["src"], _axes(a), const(False)], {}

    result += [
        spec(
            "tensorreduce",
            "sum",
            pair,
            {"reduce_op": "sum", "axes": [-1], "negate": False, "max_inst_size": None},
            ("partial_reduce",),
            reduce_project,
            _reduce,
        )
    ]

    def chain_project(v, a):
        return (
            "binary_chain",
            [v["dst"], v["data"], v["operand0"], v["operand1"], a.op0, a.op1, a.reverse1],
            {},
        )

    for name in ("scalar_tensor_scalar", "scalar_tensor_tensor"):
        result += [
            spec(
                name,
                "binary_chain",
                [
                    Operand("dst", "region"),
                    Operand("data"),
                    Operand("operand0"),
                    Operand("operand1"),
                ],
                {"op0": "mul", "op1": "add", "reverse1": False, "max_inst_size": 512},
                project=chain_project,
                lower=_chain,
            )
        ]

    def br_project(v, a):
        return (
            "binary_reduce",
            [v["dst"], v["reduced"], v["lhs"], v["rhs"], a.opcode, a.reduce_op, _axes(a)],
            {},
        )

    result += [
        spec(
            "tensorscalar_reduce",
            "binary_reduce",
            [
                Operand("dst", "region"),
                Operand("reduced", "region"),
                Operand("lhs"),
                Operand("rhs"),
            ],
            {"opcode": "add", "reduce_op": "sum", "axes": [-1], "max_inst_size": None},
            ("partial_reduce",),
            br_project,
            _binary_reduce,
        )
    ]

    def ar_project(v, a):
        return (
            "unary_reduce_with_scale_bias",
            [
                v["dst"],
                v["reduced"],
                v["src"],
                a.opcode,
                a.reduce_op,
                v["scale"],
                v["bias"],
                _axes(a),
            ],
            {},
        )

    result += [
        spec(
            "activation_reduce",
            "unary_reduce_with_scale_bias",
            [
                Operand("dst", "region"),
                Operand("reduced", "region"),
                Operand("src", "region"),
                Operand("scale", "scalar", 1.0),
                Operand("bias", "expr", 0.0),
            ],
            {"opcode": "exp", "reduce_op": "sum", "axes": [-1], "max_inst_size": None},
            ("const_bias", "partial_reduce"),
            ar_project,
            _activation_reduce,
        )
    ]

    def select_project(v, a):
        return "select", [v["dst"], v["true_value"], v["false_value"], v["pred"]], {}

    select_spec = spec(
        "affine_select",
        "select",
        [Operand("dst", "region"), Operand("true_value"), Operand("false_value"), Operand("pred")],
        project=select_project,
        lower=_select,
    )
    result += [select_spec]
    ns = namespace("trn", result)
    original = ns.affine_select

    def affine_select(dst, true_value, false_value, pred, **kwargs):
        from tvm.tirx.tensor_instruction import _operand

        dst = _operand(dst, "region")
        if not isinstance(pred, Expr):
            pred = LambdaExpr([PrimType("int32")] * len(dst.region), pred)
        return original(dst, true_value, false_value, pred, **kwargs)

    ns.affine_select = affine_select
    return ns
