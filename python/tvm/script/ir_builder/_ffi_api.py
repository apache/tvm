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
"""FFI APIs for tvm.script.ir_builder"""
# tvm-ffi-stubgen(begin): import-section
# fmt: off
# isort: off
from __future__ import annotations
from tvm_ffi import init_ffi_api as _FFI_INIT_FUNC
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from script.ir_builder import IRBuilder, IRBuilderFrame
    from tvm_ffi import Object
    from typing import Callable
# isort: on
# fmt: on
# tvm-ffi-stubgen(end)


# tvm-ffi-stubgen(begin): global/script.ir_builder
# fmt: off
_FFI_INIT_FUNC("script.ir_builder", __name__)
if TYPE_CHECKING:
    def IRBuilder() -> IRBuilder: ...
    def IRBuilderCurrent() -> IRBuilder: ...
    def IRBuilderEnter(_0: IRBuilder, /) -> None: ...
    def IRBuilderExit(_0: IRBuilder, /) -> None: ...
    def IRBuilderFrameAddCallback(_0: IRBuilderFrame, _1: Callable[[], None], /) -> None: ...
    def IRBuilderFrameEnter(_0: IRBuilderFrame, /) -> None: ...
    def IRBuilderFrameExit(_0: IRBuilderFrame, /) -> None: ...
    def IRBuilderGet(_0: IRBuilder, /) -> Object: ...
    def IRBuilderIsInScope() -> bool: ...
    def IRBuilderName(_0: str, _1: Object, /) -> Object: ...
# fmt: on
# tvm-ffi-stubgen(end)
