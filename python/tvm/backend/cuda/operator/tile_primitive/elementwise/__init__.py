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

"""CUDA elementwise dispatch.

Split by storage scope to mirror ``cuda/copy/``:

  reg.py  — operands all in ``local`` (registers)   → induced partition
  smem.py — operands all in ``shared*``             → synthesized partition

Each op in ``ops.ALL_OPS`` is registered under both variants. Per-op packed
PTX/CUDA intrinsics live in ``vec_emit/`` (``binary_f32x2`` / ``cast_vec2``
/ ``fma_f32x2``) and are attached to the relevant ``OpSpec.vec_impls``.
"""

from .register import *

# Suppress submodule-attribute leakage. Without an explicit ``__all__`` here,
# ``from tvm.backend.cuda.operator.tile_primitive.elementwise import *`` (run by
# tile_primitive/__init__.py) re-exports the implicit submodule attributes
# (``ops``, ``reg``, ``smem``, ``vec_emit``) — and ``ops`` in particular
# shadows the top-level ``tile_primitive/ops.py`` (BinaryReduce / UnaryReduce
# / ...) when downstream code does ``from tile_primitive import ops``.
__all__: list[str] = []
