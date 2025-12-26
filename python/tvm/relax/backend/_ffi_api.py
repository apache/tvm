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
"""FFI API for Relax backend."""
# tvm-ffi-stubgen(begin): import-section
# fmt: off
# isort: off
from __future__ import annotations
from tvm_ffi import init_ffi_api as _FFI_INIT_FUNC
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from collections.abc import Sequence
    from ir import IRModule
    from meta_schedule import ExtractedTask
    from relax.transform import FusionPattern
    from target import Target
# isort: on
# fmt: on
# tvm-ffi-stubgen(end)


# tvm-ffi-stubgen(begin): global/relax.backend
# fmt: off
_FFI_INIT_FUNC("relax.backend", __name__)
if TYPE_CHECKING:
    def GetPattern(_0: str, /) -> FusionPattern | None: ...
    def GetPatternsWithPrefix(_0: str, /) -> Sequence[FusionPattern]: ...
    def MetaScheduleExtractTask(_0: IRModule, _1: Target, _2: str, /) -> Sequence[ExtractedTask]: ...
    def RegisterPatterns(_0: Sequence[FusionPattern], /) -> None: ...
    def RemovePatterns(_0: Sequence[str], /) -> None: ...
# fmt: on
# tvm-ffi-stubgen(end)
