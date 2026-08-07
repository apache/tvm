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
# pylint: disable=redefined-builtin, invalid-name, too-many-arguments, too-many-locals, too-many-positional-arguments
"""PTX WGMMA companions that are not instructions (mma_async itself is ptx).

``encode_matrix_descriptor`` is a pure-C bitfield fill and ``noop_barrier``
is an empty asm with one inout register operand -- neither maps to a PTX
instruction, so they stay device_intrinsic registrations.
"""

from ._schema import device_intrinsic
from .registry import CODEGEN_REGISTRY, register_codegen
from .types import PTXDataType

# =============================================================================
# wgmma noop_barrier / descriptor encode helpers (wait_group is ptx).
# =============================================================================


# =============================================================================
# wgmma_encode_matrix_descriptor — pure-C bitfield struct fill (no asm).
# =============================================================================
device_intrinsic(
    "cuda_wgmma_encode_matrix_descriptor",
    helper_name="ptx_wgmma_encode_matrix_descriptor",
    c_signature="(uint64_t* desc, void* addr, int ldo, int sdo, int swizzle)",
    body=(
        "  GmmaDescriptor _desc{};  // value-init: reading uncovered pad bits is UB\n"
        "\n"
        "  switch (swizzle) {\n"
        "    case 0: _desc.bitfield.layout_type_ = uint8_t(0); break; // No swizzle\n"
        "    case 1: _desc.bitfield.layout_type_ = uint8_t(3); break; // 32B swizzle\n"
        "    case 2: _desc.bitfield.layout_type_ = uint8_t(2); break; // 64B swizzle\n"
        "    case 3: _desc.bitfield.layout_type_ = uint8_t(1); break; // 128B swizzle\n"
        "  }\n"
        "\n"
        "  uint32_t start_address = __cvta_generic_to_shared(addr);\n"
        "  _desc.bitfield.start_address_ = static_cast<uint16_t>(start_address >> 4);\n"
        "\n"
        "  constexpr uint8_t base_offset = 0;\n"
        "  _desc.bitfield.base_offset_ = base_offset;\n"
        "\n"
        "  _desc.bitfield.stride_byte_offset_  = static_cast<uint32_t>(sdo);\n"
        "  _desc.bitfield.leading_byte_offset_ = static_cast<uint32_t>(ldo);\n"
        "\n"
        "  *desc = (uint64_t)_desc;"
    ),
    extra_deps=("gmma_descriptor",),
)


# =============================================================================
# wgmma_noop_barrier — empty asm with one inout register operand. Two
# device_intrinsic calls, one per supported dtype; dispatcher picks the form
# based on the operand's runtime dtype.
# =============================================================================
device_intrinsic(
    "cuda_wgmma_noop_barrier_uint32",
    helper_name="ptx_wgmma_fence_uint32_t",
    c_signature="(uint32_t reg)",
    body='    asm volatile("" : "+r"(reg) :: "memory");',
)
device_intrinsic(
    "cuda_wgmma_noop_barrier_float32",
    helper_name="ptx_wgmma_fence_float",
    c_signature="(float reg)",
    body='    asm volatile("" : "+f"(reg) :: "memory");',
)


@register_codegen("cuda_wgmma_noop_barrier")
def codegen_cuda_wgmma_noop_barrier(reg):
    dtype = str(reg.dtype)
    dtype_enum = PTXDataType.from_string(dtype)
    if dtype_enum == PTXDataType.UINT32:
        op_name = "tirx.cuda_wgmma_noop_barrier_uint32"
    elif dtype_enum == PTXDataType.FLOAT32:
        op_name = "tirx.cuda_wgmma_noop_barrier_float32"
    else:
        raise ValueError(f"Only support uint32/float32 for wgmma_fence, but got {dtype}.")
    result = CODEGEN_REGISTRY[op_name]([reg])
    return result[0] if isinstance(result, tuple) else result
