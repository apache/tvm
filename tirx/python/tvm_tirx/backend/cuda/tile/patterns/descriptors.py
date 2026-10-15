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

Canonical coordinates determine the descriptor's byte strides. Only the finite
hardware swizzle modes are checked; no coordinate permutation or instruction
selection is performed. PTX ISA 9.7.18.3.1 and 9.7.18.4.1.
"""

import tvm_ffi

from tvm.backend.cuda.cpp.descriptors import encode_smem_descriptor_base_uint64
from tvm.ir import Var
from tvm.script import tirx as T
from tvm.tirx import IntImm
from tvm.tirx.layout import ComposeLayout


def swizzle_bytes(offset, swizzle):
    if swizzle == 4:
        # 128B span, 32B atomicity: XOR address bits [8:7] into [6:5].
        return T.bitwise_xor(offset, T.bitwise_and(offset // 4, 96))
    return T.bitwise_xor(offset, T.bitwise_and(offset // 8, (2**swizzle - 1) * 16))


def _encode_matrix_descriptor(ctx, smem, ldo, sdo, swizzle, atom_origin):
    if swizzle not in range(5) or any(x < 0 or x % 16 or x >= 2**18 for x in (ldo, sdo)):
        raise ValueError("matrix descriptor requires swizzle 0..4 and 16-byte strides below 2^18")
    smem.aligned(atom_origin, 16)
    if swizzle:
        period = 512 if swizzle == 4 else 128 << swizzle
        if int(smem.ty.data_alignment) % period:
            raise ValueError(
                "swizzled matrix descriptor requires a base aligned to its full swizzle period"
            )
    encoded = encode_smem_descriptor_base_uint64(ldo // 16, sdo // 16, swizzle)
    descriptor = T.uint64(encoded & 0xFFFFFFFF) | (T.uint64(encoded >> 32) << T.uint64(32))
    address = T.cuda.cvta_generic_to_shared(smem.pointer(atom_origin))
    if swizzle:
        # The descriptor starts at an unswizzled byte address. The native
        # instruction applies the permutation, including the origin's phase.
        # XOR swizzling is its own inverse; the backing base is period-aligned.
        physical = smem.memory(atom_origin) * smem.bits // 8
        address = (
            address
            - T.cast(physical, "uint32")
            + T.cast(swizzle_bytes(physical, swizzle), "uint32")
        )
    address = T.cast(address // T.uint32(16), "uint64")
    return T.bitwise_or(descriptor, T.bitwise_and(address, T.uint64(0x3FFF)))


def _constant_stride(ctx, difference):
    """Extract a candidate at one valid origin; the full walk is checked later."""
    value = ctx.analyzer.simplify(difference)
    if not isinstance(value, IntImm):
        origins = {var: bounds.start for var, bounds in ctx.static_ranges.items()}
        origins.update(dict.fromkeys([*ctx.group, *(ctx.repeat or ())], 0))
        value = tvm_ffi.structural_map(
            ctx.index_coordinates(value),
            (Var, lambda var: IntImm(var.ty.dtype, origins[var]) if var in origins else var),
            order="post",
        )
        value = ctx.analyzer.simplify(value)
    if not isinstance(value, IntImm):
        raise ValueError("descriptor strides must be compile-time constants")
    bits = int(value)
    if bits < 0 or bits % 128 or bits >= 8 * 2**18:
        raise ValueError("descriptor strides must be 16-byte multiples below 2^18")
    return bits // 8


def _layout_swizzle(smem):
    layout = smem.layout
    if isinstance(layout, ComposeLayout) and layout.swizzle_inner:
        atomic_bits = (1 << int(layout.per_element)) * smem.bits
        if atomic_bits == 128 and int(layout.atom_len) == 3:
            return int(layout.swizzle_len)
        if atomic_bits == 256 and int(layout.atom_len) == 2 and int(layout.swizzle_len) == 2:
            return 4
    return 0


def matrix_descriptor(ctx, smem, atom_shape, *, major="k", copy=False):
    """Extract, validate and encode a shared matrix's canonical address walk.

    cp coordinates are (row, b32_word); MMA coordinates are (MN, K).
    The major order fixes which coordinate enumerates hardware rows. Row-group
    and column-chunk boundaries determine the two strides. An unobserved LDO
    is 16B, an unobserved SDO is zero. Every element, including its region
    origin and group/repeat coordinates, must match the resulting descriptor.
    """
    modes = [4] if not copy and major == "mn" and smem.bits == 32 else list(range(4))
    preferred = _layout_swizzle(smem)
    if preferred in modes:
        modes.remove(preferred)
        modes.insert(0, preferred)
    rows, columns = atom_shape if major == "k" else atom_shape[::-1]

    def physical(row, column):
        coord = (row, column) if major == "k" else (column, row)
        return smem.memory(coord) * smem.bits

    failures = []
    for mode in modes:
        row_bits = (128 if mode == 4 else 16 << mode) * 8
        group_rows = 4 if mode == 4 else 8

        def permute(bits):
            return swizzle_bytes(bits // 8, mode) * 8 + bits % 8 if mode else bits

        base = permute(physical(0, 0))
        swapped = major == "mn" and mode == 0
        ldo, sdo = 16, 0
        try:
            if rows > group_rows:
                stride = _constant_stride(ctx, permute(physical(group_rows, 0)) - base)
                if swapped:
                    ldo = stride
                else:
                    sdo = stride
            chunk = (row_bits + smem.bits - 1) // smem.bits
            if chunk < columns and not (copy and mode):
                stride = _constant_stride(
                    ctx, permute(physical(0, chunk)) - base - (chunk * smem.bits % row_bits)
                )
                if swapped:
                    sdo = stride
                else:
                    ldo = stride
            row_stride, column_stride = (ldo, sdo) if swapped else (sdo, ldo)
            for row in range(rows):
                for column in range(columns):
                    column_bits = column * smem.bits
                    if not (copy and mode):
                        column_bits = (
                            column_bits // row_bits * column_stride * 8 + column_bits % row_bits
                        )
                    offset = row // group_rows * row_stride * 8 + row % group_rows * row_bits
                    ctx.require(
                        physical(row, column) == permute(base + offset + column_bits),
                        f"{smem.name}: descriptor address walk at row={row}, column={column}",
                    )
            return _encode_matrix_descriptor(ctx, smem, ldo, sdo, mode, (0, 0))
        except ValueError as error:
            failures.append(f"swizzle {mode}: {error}")
    raise ValueError(f"{smem.name}: no matching shared matrix descriptor; " + "; ".join(failures))
