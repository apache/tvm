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
"""FFI APIs for tvm.target"""
# tvm-ffi-stubgen(begin): import-section
# fmt: off
# isort: off
from __future__ import annotations
from tvm_ffi import init_ffi_api as _FFI_INIT_FUNC
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from ir import IRModule
    from target import Target, TargetKind, VirtualDevice
    from tvm_ffi import Device, Module
    from typing import Any
# isort: on
# fmt: on
# tvm-ffi-stubgen(end)



# tvm-ffi-stubgen(begin): global/target
# fmt: off
_FFI_INIT_FUNC("target", __name__)
if TYPE_CHECKING:
    def Build(_0: IRModule, _1: Target, /) -> Module: ...
    def ListTargetKindOptions(_0: TargetKind, /) -> Mapping[str, str]: ...
    def ListTargetKindOptionsFromName(_0: str, /) -> Mapping[str, str]: ...
    def ListTargetKinds() -> Sequence[str]: ...
    def Target(*args: Any) -> Any: ...
    def TargetCurrent(_0: bool, /) -> Target: ...
    def TargetEnterScope(_0: Target, /) -> None: ...
    def TargetExitScope(_0: Target, /) -> None: ...
    def TargetExport(_0: Target, /) -> Mapping[str, Any]: ...
    def TargetGetDeviceType(_0: Target, /) -> int: ...
    def TargetGetFeature(_0: Target, _1: str, /) -> Any: ...
    def TargetKindGetAttr(_0: TargetKind, _1: str, /) -> Any: ...
    def TargetTagAddTag(_0: str, _1: Mapping[str, Any], _2: bool, /) -> Target: ...
    def TargetTagListTags() -> Mapping[str, Target]: ...
    def VirtualDevice_ForDeviceTargetAndMemoryScope(_0: Device, _1: Target, _2: str, /) -> VirtualDevice: ...
    def WithHost(_0: Target, _1: Target, /) -> Target: ...
# fmt: on
# tvm-ffi-stubgen(end)
