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
"""FFI APIs for tvm.te"""
# tvm-ffi-stubgen(begin): import-section
# fmt: off
# isort: off
from __future__ import annotations
from tvm_ffi import init_ffi_api as _FFI_INIT_FUNC
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from ir import PrimExpr
    from te import ComputeOp, ExternOp, Operation, ScanOp, Tensor
    from tir import Buffer, IterVar, Stmt
    from tvm_ffi import dtype
    from typing import Any
# isort: on
# fmt: on
# tvm-ffi-stubgen(end)



# tvm-ffi-stubgen(begin): global/te
# fmt: off
_FFI_INIT_FUNC("te", __name__)
if TYPE_CHECKING:
    def ComputeOp(_0: str, _1: str, _2: Mapping[str, Any] | None, _3: Sequence[IterVar], _4: Sequence[PrimExpr], /) -> ComputeOp: ...
    def CreatePrimFunc(*args: Any) -> Any: ...
    def ExternOp(_0: str, _1: str, _2: Mapping[str, Any] | None, _3: Sequence[Tensor], _4: Sequence[Buffer], _5: Sequence[Buffer], _6: Stmt, /) -> ExternOp: ...
    def OpGetOutput(_0: Operation, _1: int, /) -> Tensor: ...
    def OpInputTensors(_0: Operation, /) -> Sequence[Tensor]: ...
    def OpNumOutputs(_0: Operation, /) -> int: ...
    def Placeholder(_0: PrimExpr | Sequence[PrimExpr], _1: dtype, _2: str, /) -> Tensor: ...
    def ScanOp(_0: str, _1: str, _2: Mapping[str, Any] | None, _3: IterVar, _4: Sequence[Tensor], _5: Sequence[Tensor], _6: Sequence[Tensor], _7: Sequence[Tensor], /) -> ScanOp: ...
    def Tensor(_0: Sequence[PrimExpr], _1: dtype, _2: Operation, _3: int, /) -> Tensor: ...
    def TensorEqual(_0: Tensor, _1: Tensor, /) -> bool: ...
    def TensorHash(_0: Tensor, /) -> int: ...
# fmt: on
# tvm-ffi-stubgen(end)
