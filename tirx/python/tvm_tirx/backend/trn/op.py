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
"""Trainium-owned NKI intrinsic Python wrappers."""

from __future__ import annotations

from tvm_tirx.ir import Call
from tvm_tirx.ir.location import UNKNOWN_LOC, Location


def nki_load(res, data, *, ty=None, loc: Location = UNKNOWN_LOC):
    return Call(ty, "tirx.nki.load", [res, data], loc=loc)


def nki_store(res, data, *, ty=None, loc: Location = UNKNOWN_LOC):
    return Call(ty, "tirx.nki.store", [res, data], loc=loc)


def nki_tensor_copy(res, data, *, ty=None, loc: Location = UNKNOWN_LOC):
    return Call(ty, "tirx.nki.tensor_copy", [res, data], loc=loc)


def nki_matmul(res, lhs, rhs, accum=True, *, ty=None, loc: Location = UNKNOWN_LOC):
    return Call(
        ty,
        "tirx.nki.matmul",
        [res, lhs, rhs, accum],
        loc=loc,
    )


def nki_activation(
    result, data, opcode, bias=0.0, scale=1.0, *, ty=None, loc: Location = UNKNOWN_LOC
):
    return Call(
        ty,
        "tirx.nki.activation",
        [result, data, opcode, bias, scale],
        loc=loc,
    )


def nki_reciprocal(result, data, *, ty=None, loc: Location = UNKNOWN_LOC):
    return Call(
        ty,
        "tirx.nki.reciprocal",
        [result, data],
        loc=loc,
    )


def nki_tensorreduce(result, data, opcode, negate, *axes, ty=None, loc: Location = UNKNOWN_LOC):
    return Call(
        ty,
        "tirx.nki.tensorreduce",
        [result, data, opcode, negate, *axes],
        loc=loc,
    )


def nki_tensortensor(result, operand0, operand1, opcode, *, ty=None, loc: Location = UNKNOWN_LOC):
    return Call(
        ty,
        "tirx.nki.tensortensor",
        [result, operand0, operand1, opcode],
        loc=loc,
    )


def nki_tensorscalar(
    result,
    operand0,
    operand1,
    opcode,
    reverse=False,
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
):
    return Call(
        ty,
        "tirx.nki.tensorscalar",
        [result, operand0, operand1, opcode, reverse],
        loc=loc,
    )


def nki_memset(result, value, *, ty=None, loc: Location = UNKNOWN_LOC):
    return Call(ty, "tirx.nki.memset", [result, value], loc=loc)


def nki_activation_reduce(
    reduce_res,
    act_res,
    data,
    opcode,
    reduce_opcode,
    bias=0.0,
    scale=1.0,
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
):
    return Call(
        ty,
        "tirx.nki.activation_reduce",
        [reduce_res, act_res, data, opcode, reduce_opcode, bias, scale],
        loc=loc,
    )


def nki_tensorscalar_reduce(
    reduce_res,
    tensorscalar_res,
    operand0,
    operand1,
    opcode,
    reduce_opcode,
    reverse=False,
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
):
    return Call(
        ty,
        "tirx.nki.tensorscalar_reduce",
        [reduce_res, tensorscalar_res, operand0, operand1, opcode, reduce_opcode, reverse],
        loc=loc,
    )


def nki_identity(result, size, *, ty=None, loc: Location = UNKNOWN_LOC):
    return Call(ty, "tirx.nki.identity", [result, size], loc=loc)


def nki_scalar_tensor_tensor(
    result,
    data,
    operand0,
    operand1,
    opcode0,
    opcode1,
    reverse0=False,
    reverse1=False,
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
):
    return Call(
        ty,
        "tirx.nki.scalar_tensor_tensor",
        [result, data, operand0, operand1, opcode0, opcode1, reverse0, reverse1],
        loc=loc,
    )


def nki_scalar_tensor_scalar(
    result,
    data,
    operand0,
    operand1,
    opcode0,
    opcode1,
    reverse0=False,
    reverse1=False,
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
):
    return Call(
        ty,
        "tirx.nki.scalar_tensor_scalar",
        [result, data, operand0, operand1, opcode0, opcode1, reverse0, reverse1],
        loc=loc,
    )


def nki_affine_select(
    result, pred, true_value, false_value, *, ty=None, loc: Location = UNKNOWN_LOC
):
    return Call(
        ty,
        "tirx.nki.affine_select",
        [result, pred, true_value, false_value],
        loc=loc,
    )


__all__ = [
    "nki_activation",
    "nki_activation_reduce",
    "nki_affine_select",
    "nki_identity",
    "nki_load",
    "nki_matmul",
    "nki_memset",
    "nki_reciprocal",
    "nki_scalar_tensor_scalar",
    "nki_scalar_tensor_tensor",
    "nki_store",
    "nki_tensor_copy",
    "nki_tensorreduce",
    "nki_tensorscalar",
    "nki_tensorscalar_reduce",
    "nki_tensortensor",
]
