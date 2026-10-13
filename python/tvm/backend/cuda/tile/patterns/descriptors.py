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

"""Mechanical matrix-descriptor encoding and its documented address walk.

Descriptor fields are explicit byte strides, never recovered by reorganizing
an operand layout. A pattern evaluates the caller's view and checks it against
this address walk. PTX ISA 9.7.18.3.1 and 9.7.18.10.2.
"""

from tvm.backend.cuda.cpp.descriptors import encode_smem_descriptor_base_uint64
from tvm.script import tirx as T

from .common import scalar


def swizzle_bytes(offset, swizzle):
    return T.bitwise_xor(offset, T.bitwise_and(offset // 8, (2**swizzle - 1) * 16))


def matrix_descriptor(ctx, smem, ldo, sdo, swizzle, atom_origin):
    ldo, sdo, swizzle = int(scalar(ldo)), int(scalar(sdo)), int(scalar(swizzle))
    if swizzle not in range(4) or any(x < 0 or x % 16 or x >= 2**18 for x in (ldo, sdo)):
        raise ValueError("matrix descriptor requires swizzle 0..3 and 16-byte strides below 2^18")
    smem.aligned(atom_origin, 16)
    if swizzle:
        period = 128 << swizzle
        if int(smem.ty.data_alignment) % period:
            raise ValueError(
                "swizzled matrix descriptor requires a base aligned to its full swizzle period"
            )
        ctx.require(
            smem.memory(atom_origin) * smem.bits // 8 % period < (16 << swizzle),
            "matrix descriptor origin must be in the first row of a swizzle atom",
        )
    encoded = encode_smem_descriptor_base_uint64(ldo // 16, sdo // 16, swizzle)
    descriptor = T.uint64(encoded & 0xFFFFFFFF) | (T.uint64(encoded >> 32) << T.uint64(32))
    address = T.cast(
        T.cuda.cvta_generic_to_shared(smem.pointer(atom_origin)) // T.uint32(16), "uint64"
    )
    return T.bitwise_or(descriptor, T.bitwise_and(address, T.uint64(0x3FFF)))


def check_matrix_rows(ctx, smem, rows, words, ldo, sdo, swizzle):
    """Check b32 rows in groups of eight; 256b rows comprise two 16B units."""
    base = smem.memory((0, 0)) * 4
    for row in range(rows):
        for word in range(words):
            # Swizzled K-major words are contiguous; hardware ignores LDO.
            column = word * 4 if swizzle else word // 4 * ldo + word % 4 * 4
            offset = row // 8 * sdo + row % 8 * (16 << swizzle) + column
            expected = swizzle_bytes(base + offset, swizzle) if swizzle else base + offset
            ctx.require(
                smem.memory((row, word)) * 4 == expected,
                "shared view does not match the explicit matrix descriptor address walk",
            )
