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
"""FFI APIs for tvm.relax.op.builtin"""
# tvm-ffi-stubgen(begin): import-section
# fmt: off
# isort: off
from __future__ import annotations
from tvm_ffi import init_ffi_api as _FFI_INIT_FUNC
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ir import RelaxExpr
    from relax.expr import DataTypeImm, PrimValue, StringImm
# isort: on
# fmt: on
# tvm-ffi-stubgen(end)


# tvm-ffi-stubgen(begin): global/relax.op.builtin
# fmt: off
_FFI_INIT_FUNC("relax.op.builtin", __name__)
if TYPE_CHECKING:
    def alloc_tensor(_0: RelaxExpr, _1: DataTypeImm, _2: PrimValue, _3: StringImm, /) -> RelaxExpr: ...
    def stop_lift_params(_0: RelaxExpr, /) -> RelaxExpr: ...
# fmt: on
# tvm-ffi-stubgen(end)
