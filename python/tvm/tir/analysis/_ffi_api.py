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
"""FFI APIs for tvm.tir.analysis"""
# tvm-ffi-stubgen(begin): import-section
# fmt: off
# isort: off
from __future__ import annotations
from tvm_ffi import init_ffi_api as _FFI_INIT_FUNC
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from ir import IRModule, IntImm, PrimExpr
    from tir import Block, Buffer, BufferRegion, PrimFunc, Stmt, Var
    from transform import Pass
    from tvm_ffi import Object
    from typing import Any
# isort: on
# fmt: on
# tvm-ffi-stubgen(end)



# tvm-ffi-stubgen(begin): global/tir.analysis
# fmt: off
_FFI_INIT_FUNC("tir.analysis", __name__)
if TYPE_CHECKING:
    def EstimateTIRFlops(_0: Object, /) -> float: ...
    def GetBlockAccessRegion(_0: Block, _1: Mapping[Var, Buffer], /) -> Sequence[Sequence[BufferRegion]]: ...
    def GetBlockReadWriteRegion(_0: Block, _1: Mapping[Var, Buffer], /) -> Sequence[Sequence[BufferRegion]]: ...
    def OOBChecker() -> Pass: ...
    def UndefinedVars(*args: Any) -> Any: ...
    def VerifyWellFormed(_0: Object, _1: bool, /) -> bool: ...
    def _identify_memcpy(_0: Stmt, /) -> Sequence[Object]: ...
    def calculate_allocated_bytes(_0: Object, /) -> Mapping[str, Mapping[str, IntImm]]: ...
    def detect_buffer_access_lca(_0: PrimFunc, /) -> Mapping[Buffer, Stmt | None]: ...
    def expr_deep_equal(_0: PrimExpr, _1: PrimExpr, /) -> bool: ...
    def find_anchor_block(_0: IRModule, /) -> Block | None: ...
    def get_vtcm_compaction_passes() -> Sequence[Pass]: ...
    def is_pure_function(_0: PrimFunc, _1: bool, /) -> bool: ...
    def verify_gpu_code(_0: PrimFunc, _1: Mapping[str, PrimExpr], /) -> bool: ...
    def verify_memory(_0: PrimFunc, /) -> bool: ...
    def verify_ssa(_0: PrimFunc, /) -> bool: ...
# fmt: on
# tvm-ffi-stubgen(end)
