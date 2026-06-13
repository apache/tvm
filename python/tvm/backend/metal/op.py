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
"""Metal-owned TIR intrinsic builders."""

from __future__ import annotations

from tvm.tirx.op import call_intrin


def make_filled_simdgroup_matrix(d, index, value, col=8, row=8):
    """Create a filled SIMDGroup matrix."""

    return call_intrin("handle", "tirx.make_filled_simdgroup_matrix", d, index, value, col, row)


def simdgroup_load(d, index, ptr, stride, col=8, row=8, transpose_matrix=False):
    """Load data from device or threadgroup memory to simdgroup."""

    return call_intrin(
        "handle",
        "tirx.simdgroup_load",
        d,
        index,
        ptr,
        stride,
        col,
        row,
        transpose_matrix,
    )


def simdgroup_store(d, index, ptr, stride, col=8, row=8, transpose_matrix=False):
    """Store data from simdgroup to device or threadgroup memory."""

    return call_intrin(
        "handle",
        "tirx.simdgroup_store",
        d,
        index,
        ptr,
        stride,
        col,
        row,
        transpose_matrix,
    )


def simdgroup_multiply_accumulate(d, index_d, a, index_a, b, index_b, c, index_c):
    """Multiply and accumulate two matrices in simdgroup."""

    return call_intrin(
        "handle",
        "tirx.simdgroup_multiply_accumulate",
        d,
        index_d,
        a,
        index_a,
        b,
        index_b,
        c,
        index_c,
    )


__all__ = [
    "make_filled_simdgroup_matrix",
    "simdgroup_load",
    "simdgroup_multiply_accumulate",
    "simdgroup_store",
]
