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

"""Explicit instruction patterns, independent of adaptive CUDA tile lowering."""

from tvm.backend.cuda.ptx import instruction_name
from tvm.ir import StringImm
from tvm.tirx.tile.instruction import Instruction, Operand

from . import ld_global, ldmatrix
from .common import Operands

_PATTERNS = {
    "ld.global.v4.b32": ld_global,
    "ldmatrix.sync.aligned.m8n8.x2.shared.b16": ldmatrix,
}


def _lower(view, sctx):
    opcode, dst, src = view.args
    name = opcode.value
    if name not in _PATTERNS:
        raise ValueError(f"No explicit tile pattern for {name}; supported: {tuple(_PATTERNS)}")
    pattern = _PATTERNS[name]
    if sctx.scope_kind != pattern.SCOPE:
        raise ValueError(f"{name} requires scope={pattern.SCOPE!r}")
    operands = Operands(dst, src, pattern.SHAPE, pattern.SRC_SCOPE, sctx)
    pattern.check(operands, sctx)
    return operands.lower(name, pattern.emit)


_construct = Instruction(
    "tirx.cuda.tile.apply",
    "ptx_pattern",
    (Operand("instruction"), Operand("dst", "region"), Operand("src", "region")),
    ("thread", "warp"),
    lower=_lower,
).register()


def apply(instruction, dst, src, *, scope="thread", **kwargs):
    """Apply one explicitly selected PTX instruction over canonical atom tiles.

    Leading dimensions are repeat axes; trailing dimensions must match the
    instruction's documented operand pattern. No instruction selection, axis
    search, implicit data rearrangement, or scalar fallback is performed.

    The prototype supports ``ld.global.v4.b32`` at thread scope and
    ``ldmatrix.sync.aligned.m8n8.x2.shared.b16`` at warp scope. See
    ``docs/tirx/cuda_tile_patterns.rst`` for both complete contracts.
    """
    if isinstance(instruction, StringImm):
        instruction = instruction.value
    return _construct(instruction_name(instruction), dst, src, scope=scope, **kwargs)
