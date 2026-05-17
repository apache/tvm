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

"""Minimal pattern table for the XNNPACK Relax backend."""

import tvm
from tvm.ir import IRModule
from tvm import relax
from tvm.relax.dpl.pattern import is_op, wildcard
from tvm.relax.transform import FuseOpsByPattern, PatternCheckContext

from .pattern_registry import get_patterns_with_prefix, register_patterns
from .utils import has_leaking_intermediate_variables


def _get_static_shape(expr: relax.Expr) -> list[int] | None:
    sinfo = expr.struct_info
    if not isinstance(sinfo, relax.TensorStructInfo):
        return None
    if sinfo.shape is None or not hasattr(sinfo.shape, "values"):
        return None

    shape = []
    for dim in sinfo.shape.values:
        if not isinstance(dim, tvm.tirx.expr.IntImm | int):
            return None
        dim = int(dim)
        if dim <= 0:
            return None
        shape.append(dim)
    return shape


def _check_relu(context: PatternCheckContext) -> bool:
    if has_leaking_intermediate_variables(context):
        return False

    input_expr = context.annotated_expr["input"]
    root_expr = context.annotated_expr["root"]

    if isinstance(input_expr, relax.Constant):
        return False

    if input_expr.struct_info.dtype != "float32" or root_expr.struct_info.dtype != "float32":
        return False

    input_shape = _get_static_shape(input_expr)
    output_shape = _get_static_shape(root_expr)
    if input_shape is None or output_shape is None:
        return False

    return input_shape == output_shape


_input = wildcard()
_relu = is_op("relax.nn.relu")(_input)

register_patterns(
    [
        (
            "xnnpack.relu",
            _relu,
            {"input": _input, "root": _relu},
            _check_relu,
        )
    ]
)


def partition_for_xnnpack(mod: IRModule) -> IRModule:
    """Partition the input module into XNNPACK-supported subgraphs.

    Phase 2 supports only static-shape float32 ``relax.nn.relu``.
    """

    patterns = get_patterns_with_prefix("xnnpack")
    return FuseOpsByPattern(patterns, bind_constants=False, annotate_codegen=True)(mod)
