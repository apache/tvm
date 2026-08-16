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

from tvm.backend.cuda.tile_primitive.common import smem_desc_add_16B_offset
from tvm.script import tirx as T


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
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(self._buf[0]), smem_ptr, ldo, sdo, swizzle
        )

    def add_16B_offset(self, offset):
        return smem_desc_add_16B_offset(self._buf[0], offset)

    def make_lo_uniform(self):
        """Broadcast the lower 32 bits to all warp lanes via ``__shfl_sync``."""
        desc_lo = T.alloc_local([1], "uint32")
        desc_hi = T.alloc_local([1], "uint32")
        T.ptx.mov.b64(desc_lo[0], desc_hi[0], self._buf[0])
        T.ptx.shfl_sync.idx.b32(
            desc_lo[0],
            desc_lo[0],
            T.uint32(0),
            T.uint32(0x1F),
            T.uint32(0xFFFFFFFF),
        )
        T.ptx.mov.b64(self._buf[0], desc_lo[0], desc_hi[0])
