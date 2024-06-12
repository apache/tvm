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

"""Pattern table for CLML backend"""
import operator
from functools import reduce
import tvm
from tvm.relax import transform
from tvm.relax.dpl.pattern import (
    is_op,
    wildcard,
    GlobalVarPattern,
    TuplePattern,
)
from tvm import IRModule, relax, tir
from tvm.relax.backend.pattern_registry import register_patterns


def _shape_1d(shape):
    return reduce(operator.mul, shape, 1)


def _is_bias_like(shape, out_channel):
    return shape[-1] == out_channel and _shape_1d(shape) == out_channel


def _check_dequantize_matmul(ctx: relax.transform.PatternCheckContext) -> bool:
    root = ctx.annotated_expr["root"]
    wdq = ctx.annotated_expr["w_decoded"]

    if ctx.annotated_expr["lhs"].struct_info.dtype != "float16":
        return False
    if not isinstance(wdq, relax.Call):
        return False
    g_var = wdq.args[0]
    if not (isinstance(g_var, relax.GlobalVar) and "dequantize" in g_var.name_hint):
        return False

    if not (
        (len(root.struct_info.shape) == 3)
        and isinstance(root.struct_info.shape[0], tir.IntImm)
        and (root.struct_info.shape[0] == 1)
    ):
        return False

    if (
        (len(root.struct_info.shape) == 3)
        and not isinstance(root.struct_info.shape[-2], tir.IntImm)
        and not isinstance(root.struct_info.shape[-1], tir.IntImm)
    ):
        return False

    # if "bias" in ctx.annotated_expr:
    #    bias_shape = ctx.annotated_expr["bias"].struct_info.shape
    #    if not _is_bias_like(bias_shape, root.struct_info.shape[-1]):
    #        return False

    return True


def dequantize_matmul_patterns():
    """Returns a list of supported decode -> matmul patterns."""

    def _dequantize_matmul_pattern(name):
        scales = wildcard()
        x = wildcard()
        w_packed = wildcard()

        w_decoded = is_op("relax.call_tir")(
            GlobalVarPattern(),
            TuplePattern([w_packed, scales]),
        )
        matmul = is_op("relax.matmul")(x, w_decoded)

        annotations = {
            "root": matmul,
            "lhs": x,
            "w_encoded": w_packed,
            "w_decoded": w_decoded,
            "scales": scales,
        }

        if "bias" in name:
            annotations["bias"] = bias = wildcard()
            out = is_op("relax.add")(matmul, bias)
        else:
            out = matmul

        if "cast" in name:
            out = is_op("relax.astype")(out)

        return name, out, annotations, _check_dequantize_matmul

    return [
        _dequantize_matmul_pattern("adreno_accl.dequant_matmul_bias_cast"),
        _dequantize_matmul_pattern("adreno_accl.dequant_matmul_bias"),
        _dequantize_matmul_pattern("adreno_accl.dequant_matmul_cast"),
        _dequantize_matmul_pattern("adreno_accl.dequant_matmul"),
    ]


adreno_byoc_patterns = [
    *dequantize_matmul_patterns(),
]
register_patterns(adreno_byoc_patterns)


@tvm.transform.module_pass(opt_level=0, name="PartitionForAdrenoACCL")
class PartitionForAdrenoACCL:
    """A compiler pass that partition BYOC adreno Matmul."""

    def transform_module(
        self,
        mod: IRModule,
        _ctx: tvm.transform.PassContext,
    ) -> IRModule:
        """Apply required passed to transform"""

        mod = tvm.transform.Sequential(
            [
                transform.Normalize(),
                transform.FuseOpsByPattern(adreno_byoc_patterns, annotate_codegen=True),
                transform.RunCodegen(),
            ]
        )(mod)

        return mod
