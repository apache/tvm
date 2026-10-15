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
"""Shared machinery behind every CUDA device-intrinsic codegen.

Both dialects build on this layer: :mod:`tvm.backend.cuda.ptx` renders PTX
instructions from a table, and :mod:`tvm.backend.cuda.cpp` registers
hand-written CUDA C++ device helpers.

- ``registry`` — the op-name → codegen map C++ queries during codegen.
- ``schema`` — :func:`device_intrinsic`, the declarative helper registration.
- ``header`` — CUDA header generator and its helper-tag table.
- ``types`` — PTX dtype enum mirroring ``src/backend/cuda/codegen/ptx.cc``.
- ``utils`` — small parsing / validation helpers.

Importing ``registry`` and ``header`` is load-bearing: their module-level
``register_global_func`` decorators publish ``tirx.intrinsics.cuda.get_codegen``
and ``tirx.intrinsics.cuda.header_generator``, which ``codegen_cuda.cc`` looks
up when it emits a kernel. Nothing else imports them for their side effects, so
dropping them here fails at kernel-build time, not at import time.
"""

from . import header, registry, schema, types, utils
from .header import TAGS, header_generator
from .registry import CODEGEN_REGISTRY, get_codegen, register_codegen
from .schema import device_intrinsic
from .types import PTXDataType

__all__ = [
    "CODEGEN_REGISTRY",
    "TAGS",
    "PTXDataType",
    "device_intrinsic",
    "get_codegen",
    "header_generator",
    "register_codegen",
]
