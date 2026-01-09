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
"""FFI for TVM diagnostics."""
# tvm-ffi-stubgen(begin): import-section
# fmt: off
# isort: off
from __future__ import annotations
from tvm_ffi import init_ffi_api as _FFI_INIT_FUNC
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ir import IRModule, Span
    from typing import Callable
# isort: on
# fmt: on
# tvm-ffi-stubgen(end)



# tvm-ffi-stubgen(begin): global/diagnostics
# fmt: off
_FFI_INIT_FUNC("diagnostics", __name__)
if TYPE_CHECKING:
    def ClearRenderer() -> None: ...
    def Default(_0: IRModule, /) -> DiagnosticContext: ...
    def DefaultRenderer() -> DiagnosticRenderer: ...
    def Diagnostic(_0: int, _1: Span, _2: str, /) -> Diagnostic: ...
    def DiagnosticContext(_0: IRModule, _1: DiagnosticRenderer, /) -> DiagnosticContext: ...
    def DiagnosticContextRender(_0: DiagnosticContext, /) -> None: ...
    def DiagnosticRenderer(_0: Callable[[DiagnosticContext], None], /) -> DiagnosticRenderer: ...
    def DiagnosticRendererRender(_0: DiagnosticRenderer, _1: DiagnosticContext, /) -> None: ...
    def Emit(_0: DiagnosticContext, _1: Diagnostic, /) -> None: ...
    def GetRenderer() -> DiagnosticRenderer: ...
# fmt: on
# tvm-ffi-stubgen(end)
