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
"""FFI APIs"""
# tvm-ffi-stubgen(begin): import-section
# fmt: off
# isort: off
from __future__ import annotations
from tvm_ffi import init_ffi_api as _FFI_INIT_FUNC
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from ir import BaseFunc, GlobalInfo, GlobalVar, VDevice
    from script.ir_builder import IRModuleFrame
    from tvm_ffi import Object
    from typing import Any
# isort: on
# fmt: on
# tvm-ffi-stubgen(end)


# tvm-ffi-stubgen(begin): global/script.ir_builder.ir
# fmt: off
_FFI_INIT_FUNC("script.ir_builder.ir", __name__)
if TYPE_CHECKING:
    def DeclFunction(_0: str, _1: BaseFunc, /) -> GlobalVar: ...
    def DefFunction(_0: str, _1: BaseFunc, /) -> None: ...
    def IRModule() -> IRModuleFrame: ...
    def LookupVDevice(_0: str, _1: int, /) -> VDevice: ...
    def ModuleAttrs(_0: Mapping[str, Any], _1: bool, /) -> None: ...
    def ModuleGetAttr(_0: str, /) -> Object | None: ...
    def ModuleGlobalInfos(_0: Mapping[str, Sequence[GlobalInfo]], /) -> None: ...
    def ModuleSetAttr(_0: str, _1: Object | None, _2: bool, /) -> None: ...
# fmt: on
# tvm-ffi-stubgen(end)
