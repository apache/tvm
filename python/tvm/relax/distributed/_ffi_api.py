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
"""FFI APIs for tvm.relax.distributed"""
# tvm-ffi-stubgen(begin): import-section
# fmt: off
# isort: off
from __future__ import annotations
from tvm_ffi import init_ffi_api as _FFI_INIT_FUNC
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from collections.abc import Sequence
    from ir import IntImm, Range, Span
    from relax import DTensorStructInfo, TensorStructInfo
    from relax.distributed import DeviceMesh, Placement, PlacementSpec
    from tvm_ffi import Shape
# isort: on
# fmt: on
# tvm-ffi-stubgen(end)


# tvm-ffi-stubgen(begin): global/relax.distributed
# fmt: off
_FFI_INIT_FUNC("relax.distributed", __name__)
if TYPE_CHECKING:
    def DTensorStructInfo(_0: TensorStructInfo, _1: DeviceMesh, _2: Placement, _3: Span, /) -> DTensorStructInfo: ...
    def DeviceMesh(_0: Shape, _1: Sequence[IntImm], _2: Range | None, /) -> DeviceMesh: ...
    def Placement(_0: Sequence[PlacementSpec], /) -> Placement: ...
    def PlacementFromText(_0: str, /) -> Placement: ...
    def Replica() -> PlacementSpec: ...
    def Sharding(_0: int, /) -> PlacementSpec: ...
# fmt: on
# tvm-ffi-stubgen(end)
