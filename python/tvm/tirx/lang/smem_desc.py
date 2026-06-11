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

"""SMEM matrix descriptor helper for tcgen05 / wgmma."""

from tvm.script import tirx as T
from tvm.tirx.operator.tile_primitive.cuda.common import smem_desc_add_16B_offset


@T.meta_class
class SmemDescriptor:
    """Encoded once via :meth:`init`, reused via :meth:`add_16B_offset`."""

    def __init__(self):
        self._buf = T.alloc_local([1], "uint64")

    @property
    def desc(self):
        return self._buf[0]

    @T.inline
    def init(self, smem_ptr, ldo, sdo, swizzle):
        T.ptx.tcgen05.encode_matrix_descriptor(
            T.address_of(self._buf[0]), smem_ptr, ldo, sdo, swizzle
        )

    def add_16B_offset(self, offset):
        return smem_desc_add_16B_offset(self._buf[0], offset)

    def make_lo_uniform(self):
        """Broadcast the lower 32 bits to all warp lanes via ``__shfl_sync``."""
        func_name = "smem_desc_make_lo_uniform"
        source_code = f"""
__forceinline__ __device__ void {func_name}(uint64_t* desc) {{
    SmemDescriptor* d = reinterpret_cast<SmemDescriptor*>(desc);
    d->lo = __shfl_sync(0xffffffff, d->lo, 0);
}}
"""
        return T.cuda.func_call(
            func_name, T.address_of(self._buf[0]), source_code=source_code, return_type="void"
        )
