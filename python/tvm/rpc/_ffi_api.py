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
"""FFI APIs for tvm.rpc"""
# tvm-ffi-stubgen(begin): import-section
# fmt: off
# isort: off
from __future__ import annotations
from tvm_ffi import init_ffi_api as _FFI_INIT_FUNC
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tvm_ffi import Module
    from typing import Any, Callable
# isort: on
# fmt: on
# tvm-ffi-stubgen(end)



# tvm-ffi-stubgen(begin): global/rpc
# fmt: off
_FFI_INIT_FUNC("rpc", __name__)
if TYPE_CHECKING:
    def Connect(*args: Any) -> Any: ...
    def CreateEventDrivenServer(_0: Callable[..., Any], _1: str, _2: str, /) -> Callable[..., Any]: ...
    def CreatePipeClient(*args: Any) -> Any: ...
    def ImportRemoteModule(_0: Module, _1: Module, /) -> None: ...
    def LoadRemoteModule(_0: Module, _1: str, /) -> Module: ...
    def LocalSession() -> Module: ...
    def ReturnException(_0: int, _1: str, /) -> None: ...
    def ServerLoop(*args: Any) -> Any: ...
    def SessTableIndex(*args: Any) -> Any: ...
# fmt: on
# tvm-ffi-stubgen(end)
