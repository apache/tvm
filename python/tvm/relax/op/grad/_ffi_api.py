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
"""FFI APIs for tvm.relax.op.grad"""
# tvm-ffi-stubgen(begin): import-section
# fmt: off
# isort: off
from __future__ import annotations
from tvm_ffi import init_ffi_api as _FFI_INIT_FUNC
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from collections.abc import Sequence
    from ir import IntImm, RelaxExpr
# isort: on
# fmt: on
# tvm-ffi-stubgen(end)


# tvm-ffi-stubgen(begin): global/relax.op.grad
# fmt: off
_FFI_INIT_FUNC("relax.op.grad", __name__)
if TYPE_CHECKING:
    def avg_pool2d_backward(_0: RelaxExpr, _1: RelaxExpr, _2: Sequence[IntImm], _3: Sequence[IntImm], _4: Sequence[IntImm], _5: Sequence[IntImm], _6: bool, _7: bool, _8: str, _9: str | None, /) -> RelaxExpr: ...
    def end_checkpoint(_0: RelaxExpr, /) -> RelaxExpr: ...
    def max_pool2d_backward(_0: RelaxExpr, _1: RelaxExpr, _2: Sequence[IntImm], _3: Sequence[IntImm], _4: Sequence[IntImm], _5: Sequence[IntImm], _6: bool, _7: bool, _8: str, _9: str | None, /) -> RelaxExpr: ...
    def nll_loss_backward(_0: RelaxExpr, _1: RelaxExpr, _2: RelaxExpr, _3: RelaxExpr | None, _4: str, _5: int, /) -> RelaxExpr: ...
    def no_grad(_0: RelaxExpr, /) -> RelaxExpr: ...
    def start_checkpoint(_0: RelaxExpr, /) -> RelaxExpr: ...
    def take_backward(_0: RelaxExpr, _1: RelaxExpr, _2: RelaxExpr, _3: int | None, /) -> RelaxExpr: ...
# fmt: on
# tvm-ffi-stubgen(end)
