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

"""Explicit CUDA instruction patterns over ordinary IR Calls."""

from tvm.backend.cuda.ptx import instruction_forms, instruction_name
from tvm.ir import StringImm, Tuple
from tvm.tirx.tile.instruction import Instruction, Operand

from . import async_copy, ldmatrix, memory, mma, register, tcgen05_mma, tma, tmem, tmem_copy
from .ewise import ewise


def _lower(view, sctx):
    opcode, packed_args, packed_params = view.args
    args = list(packed_args.fields)
    params = {
        pair.fields[0].value: pair.fields[1]
        for pair in (packed_params.fields if packed_params is not None else ())
    }
    forms = instruction_forms(opcode.value)
    entry, modifiers = forms[0]
    if entry.ptx_name in ("ld", "st"):
        lower = memory.lower
    elif entry.name in ("ldmatrix", "stmatrix"):
        lower = ldmatrix.lower
    elif entry.ptx_name == "mma":
        lower = mma.lower
    elif entry.name.startswith(("cp_async_bulk_tensor_", "cp_reduce_async_bulk_tensor")):
        lower = tma.lower
    elif entry.name.startswith("tcgen05_mma"):
        lower = tcgen05_mma.lower
    elif entry.name == "tcgen05_cp":
        lower = tmem_copy.lower
    elif entry.name in ("tcgen05_ld", "tcgen05_st", "tcgen05_ld_split", "tcgen05_st_split"):
        lower = tmem.lower
    elif entry.name.startswith(("cp_async_ca", "cp_async_cg")) or entry.name == "cp_async_bulk_s2c":
        lower = async_copy.lower
    elif entry.ptx_name in register.MNEMONICS:
        return register.lower(opcode.value, forms, args, params, sctx)
    else:
        raise ValueError(f"No explicit tile pattern for {opcode.value}")
    return lower(opcode.value, entry, modifiers, args, params, sctx)


_construct = Instruction(
    "tirx.cuda.tile.apply",
    "ptx_pattern",
    (Operand("instruction"), Operand("operands"), Operand("parameters", default=None)),
    ("thread", "warp", "warpgroup", "cta", "cluster"),
    lower=_lower,
).register()


def apply(instruction, *operands, scope="thread", **parameters):
    """Repeat the explicitly named PTX instruction over its canonical operands.

    Operand axes are group, repeat, then instruction-specific atom coordinates.
    Group axes bind execution IDs; repeat axes generate unrolled loops. Static
    descriptor fields and runtime instruction operands are serialized as normal
    Call arguments. See ``docs/tirx/cuda_tile_patterns.rst`` for the contracts.
    """
    if isinstance(instruction, StringImm):
        instruction = instruction.value
    # The generic IR printer spells the registered three-argument Call. Accept
    # that spelling as well as the user-facing variadic operand syntax.
    if len(operands) == 2 and all(isinstance(x, Tuple | tuple | list) for x in operands):
        operands, encoded = operands
        pairs = encoded.fields if isinstance(encoded, Tuple) else encoded
        decoded = {}
        for pair in pairs:
            pair = pair.fields if isinstance(pair, Tuple) else pair
            key = pair[0].value if isinstance(pair[0], StringImm) else pair[0]
            decoded[key] = pair[1]
        decoded.update(parameters)
        parameters = decoded
        operands = tuple(operands.fields if isinstance(operands, Tuple) else operands)
    constructor_keys = {
        key: parameters.pop(key) for key in ("attrs", "ty", "ty_args", "loc") if key in parameters
    }
    packed = tuple((name, value) for name, value in sorted(parameters.items()))
    return _construct(
        instruction_name(instruction), operands, packed, scope=scope, **constructor_keys
    )
