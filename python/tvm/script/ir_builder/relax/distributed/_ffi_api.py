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
"""FFI APIs for tvm.script.ir_builder.relax.distributed"""
# tvm-ffi-stubgen(begin): import-section
# fmt: off
# isort: off
from __future__ import annotations
from tvm_ffi import init_ffi_api as _FFI_INIT_FUNC
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from collections.abc import Sequence
    from ir import RelaxExpr
    from relax import DTensorStructInfo
    from relax.expr import Tuple
# isort: on
# fmt: on
# tvm-ffi-stubgen(end)


# tvm-ffi-stubgen(begin): global/script.ir_builder.relax.distributed
# fmt: off
_FFI_INIT_FUNC("script.ir_builder.relax.distributed", __name__)
if TYPE_CHECKING:
    def call_tir_dist(_0: RelaxExpr, _1: Tuple, _2: Sequence[DTensorStructInfo], _3: RelaxExpr | None, /) -> RelaxExpr: ...
# fmt: on
# tvm-ffi-stubgen(end)
