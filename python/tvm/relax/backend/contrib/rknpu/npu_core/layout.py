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

"""NPU data layout index functions for the RK3588 tiled memory format.

Weight layout: tiles of ``(kernels_per_group x 32)``
    FP16: 16 kernels x 32 channels
    INT8: 32 kernels x 32 channels

All index functions use **1-based** indices matching the original rk3588-npu C code.
They accept both scalar ints and numpy arrays for vectorised scatter/gather.
"""


def weight_index_fp16(C, k, c):
    """Return the flat element index for kernel *k*, channel *c* (FP16).

    Tiles of 16 kernels x 32 channels.  *k* and *c* are **1-based**.
    *C* is the total (aligned) channel count.
    """
    kpg = (k - 1) // 16
    cpg = (c - 1) // 32
    base = (cpg * 32) * 16 + kpg * 16 * C
    return base + (c - 1) % 32 + ((k - 1) % 16) * 32


def weight_index_int8(C, k, c):
    """Return the flat element index for kernel *k*, channel *c* (INT8).

    Tiles of 32 kernels x 32 channels.  *k* and *c* are **1-based**.
    *C* is the total (aligned) channel count.
    """
    kpg = (k - 1) // 32
    cpg = (c - 1) // 32
    base = (cpg * 32) * 32 + kpg * 32 * C
    return base + (c - 1) % 32 + ((k - 1) % 32) * 32
