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
"""FFI for profiling"""
# tvm-ffi-stubgen(begin): import-section
# fmt: off
# isort: off
from __future__ import annotations
from tvm_ffi import init_ffi_api as _FFI_INIT_FUNC
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from runtime.profiling import DeviceWrapper, MetricCollector, Report
    from tvm_ffi import Device, Module, Object
    from typing import Any, Callable
# isort: on
# fmt: on
# tvm-ffi-stubgen(end)


# tvm-ffi-stubgen(begin): global/runtime.profiling
# fmt: off
_FFI_INIT_FUNC("runtime.profiling", __name__)
if TYPE_CHECKING:
    def AsCSV(_0: Report, /) -> str: ...
    def AsJSON(_0: Report, /) -> str: ...
    def AsTable(_0: Report, _1: bool, _2: bool, _3: bool, /) -> str: ...
    def Count(_0: int, /) -> Object: ...
    def DeviceWrapper(_0: Device, /) -> DeviceWrapper: ...
    def Duration(_0: float, /) -> Object: ...
    def FromJSON(_0: str, /) -> Report: ...
    def Percent(_0: float, /) -> Object: ...
    def ProfileFunction(_0: Module, _1: str, _2: int, _3: int, _4: int, _5: Sequence[MetricCollector], /) -> Callable[..., Any]: ...
    def Ratio(_0: float, /) -> Object: ...
    def Report(_0: Sequence[Mapping[str, Any]], _1: Mapping[str, Mapping[str, Any]], _2: Mapping[str, Any], /) -> Report: ...
# fmt: on
# tvm-ffi-stubgen(end)
