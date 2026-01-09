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
"""FFI APIs for tvm.script.ir_builder.relax"""
# tvm-ffi-stubgen(begin): import-section
# fmt: off
# isort: off
from __future__ import annotations
from tvm_ffi import init_ffi_api as _FFI_INIT_FUNC
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from ir import IntImm, RelaxExpr, StructInfo
    from relax.expr import Var, VarBinding
    from script.ir_builder.relax import BlockFrame, ElseFrame, FunctionFrame, IfFrame, SeqExprFrame, ThenFrame
    from typing import Any
# isort: on
# fmt: on
# tvm-ffi-stubgen(end)


# tvm-ffi-stubgen(begin): global/script.ir_builder.relax
# fmt: off
_FFI_INIT_FUNC("script.ir_builder.relax", __name__)
if TYPE_CHECKING:
    def Arg(_0: str, _1: StructInfo, /) -> Var: ...
    def BindingBlock() -> BlockFrame: ...
    def Dataflow() -> BlockFrame: ...
    def DataflowBlockOutput(_0: Sequence[Var], /) -> None: ...
    def Else() -> ElseFrame: ...
    def Emit(_0: RelaxExpr, _1: StructInfo | None, /) -> Var: ...
    def EmitMatchCast(_0: RelaxExpr, _1: StructInfo, /) -> Var: ...
    def EmitVarBinding(_0: VarBinding, /) -> Var: ...
    def FuncAttrs(_0: Mapping[str, Any], /) -> None: ...
    def FuncName(_0: str, /) -> None: ...
    def FuncRetStructInfo(_0: StructInfo, /) -> None: ...
    def FuncRetValue(_0: RelaxExpr, /) -> None: ...
    def Function(_0: IntImm, _1: IntImm, /) -> FunctionFrame: ...
    def If(_0: RelaxExpr, /) -> IfFrame: ...
    def SeqExpr() -> SeqExprFrame: ...
    def Then() -> ThenFrame: ...
# fmt: on
# tvm-ffi-stubgen(end)
