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

from tvm.tirx.op import call_intrin


def nki_load(res, data):
    return call_intrin("", "tirx.nki_load", res, data)


def nki_store(res, data):
    return call_intrin("", "tirx.nki_store", res, data)


def nki_tensor_copy(res, data):
    return call_intrin("", "tirx.nki_tensor_copy", res, data)


def nki_matmul(res, lhs, rhs, accum=True):
    return call_intrin("", "tirx.nki_matmul", res, lhs, rhs, accum)


def nki_activation(result, data, opcode, bias=0.0, scale=1.0):
    return call_intrin("", "tirx.nki_activation", result, data, opcode, bias, scale)


def nki_reciprocal(result, data):
    return call_intrin("", "tirx.nki_reciprocal", result, data)


def nki_tensorreduce(result, data, opcode, negate, *axes):
    return call_intrin("", "tirx.nki_tensorreduce", result, data, opcode, negate, *axes)


def nki_tensortensor(result, operand0, operand1, opcode):
    return call_intrin("", "tirx.nki_tensortensor", result, operand0, operand1, opcode)


def nki_tensorscalar(result, operand0, operand1, opcode, reverse=False):
    return call_intrin("", "tirx.nki_tensorscalar", result, operand0, operand1, opcode, reverse)


def nki_memset(result, value):
    return call_intrin("", "tirx.nki_memset", result, value)


def nki_activation_reduce(reduce_res, act_res, data, opcode, reduce_opcode, bias=0.0, scale=1.0):
    return call_intrin(
        "",
        "tirx.nki_activation_reduce",
        reduce_res,
        act_res,
        data,
        opcode,
        reduce_opcode,
        bias,
        scale,
    )


def nki_tensorscalar_reduce(
    reduce_res, tensorscalar_res, operand0, operand1, opcode, reduce_opcode, reverse=False
):
    return call_intrin(
        "",
        "tirx.nki_tensorscalar_reduce",
        reduce_res,
        tensorscalar_res,
        operand0,
        operand1,
        opcode,
        reduce_opcode,
        reverse,
    )


def nki_identity(result, size):
    return call_intrin("", "tirx.nki_identity", result, size)


def nki_scalar_tensor_tensor(
    result, data, operand0, operand1, opcode0, opcode1, reverse0=False, reverse1=False
):
    return call_intrin(
        "",
        "tirx.nki_scalar_tensor_tensor",
        result,
        data,
        operand0,
        operand1,
        opcode0,
        opcode1,
        reverse0,
        reverse1,
    )


def nki_scalar_tensor_scalar(
    result, data, operand0, operand1, opcode0, opcode1, reverse0=False, reverse1=False
):
    return call_intrin(
        "",
        "tirx.nki_scalar_tensor_scalar",
        result,
        data,
        operand0,
        operand1,
        opcode0,
        opcode1,
        reverse0,
        reverse1,
    )


def nki_affine_select(result, pred, true_value, false_value):
    return call_intrin("", "tirx.nki_affine_select", result, pred, true_value, false_value)


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
