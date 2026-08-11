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
# pylint: disable=unused-import
"""Hand-written CUDA C++ device helpers.

The CUDA backend emits device intrinsics two ways. A single PTX instruction is
a row in the :mod:`tvm.backend.cuda.ptx` table, rendered by a generic engine.
Everything else -- anything needing a hand-written ``__device__`` function body
-- is registered here, grouped by why it needs one:

- ``builtins`` — a CUDA builtin or library call wrapped as an op, no asm.
- ``asm`` — a body that must be hand-written asm: spin-wait loops, barrier
  pairs, empty compiler-barrier asm, special-register reads.
- ``descriptors`` — wgmma / tcgen05 descriptor bitfield encoding, plus the
  tcgen05 MMA dtype-kind and shape validation those encoders enforce.
- ``instrument`` — profiler timers, IKET events, ``printf`` / ``trap``.
- ``nvshmem`` — NVSHMEM RMA, signal, and collective bindings.

Importing this package is what registers those codegens; the shared registry,
``device_intrinsic`` schema, and header generator live in
:mod:`tvm.backend.cuda.codegen`.
"""

# Import op modules to register their codegen functions.
from . import asm, builtins, descriptors, instrument, nvshmem

__all__ = []
