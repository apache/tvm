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
"""Constructor APIs"""
# tvm-ffi-stubgen(begin): import-section
# fmt: off
# isort: off
from __future__ import annotations
from tvm_ffi import init_ffi_api as _FFI_INIT_FUNC
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from collections.abc import Sequence
    from ir import FloatImm, RelaxExpr
    from tvm_ffi import dtype
# isort: on
# fmt: on
# tvm-ffi-stubgen(end)


# tvm-ffi-stubgen(begin): global/relax.op.image
# fmt: off
_FFI_INIT_FUNC("relax.op.image", __name__)
if TYPE_CHECKING:
    def grid_sample(_0: RelaxExpr, _1: RelaxExpr, _2: str, _3: str, _4: str, _5: bool, /) -> RelaxExpr: ...
    def resize2d(_0: RelaxExpr, _1: RelaxExpr, _2: Sequence[FloatImm], _3: str, _4: str, _5: str, _6: str, _7: float, _8: int, _9: float, _10: dtype | None, /) -> RelaxExpr: ...
# fmt: on
# tvm-ffi-stubgen(end)
