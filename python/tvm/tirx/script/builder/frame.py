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
"""IRBuilder for TIR"""

from tvm_ffi import register_object as _register_object

from tvm.script.ir_builder.base import IRBuilderFrame
from tvm.tirx import Var


@_register_object("script.ir_builder.tirx.TIRFrame")
class TIRFrame(IRBuilderFrame): ...


@_register_object("script.ir_builder.tirx.PrimFuncFrame")
class PrimFuncFrame(TIRFrame): ...


@_register_object("script.ir_builder.tirx.SSBlockFrame")
class SBlockFrame(TIRFrame): ...


@_register_object("script.ir_builder.tirx.SBlockInitFrame")
class BlockInitFrame(TIRFrame): ...


@_register_object("script.ir_builder.tirx.ForFrame")
class ForFrame(TIRFrame):
    def __enter__(self) -> Var | list[Var]:  # type: ignore[override]
        super().__enter__()
        return self.vars if len(self.vars) > 1 else self.vars[0]


@_register_object("script.ir_builder.tirx.AssertFrame")
class AssertFrame(TIRFrame): ...


@_register_object("script.ir_builder.tirx.AttrFrame")
class AttrFrame(TIRFrame): ...


@_register_object("script.ir_builder.tirx.WhileFrame")
class WhileFrame(TIRFrame): ...


@_register_object("script.ir_builder.tirx.IfFrame")
class IfFrame(TIRFrame): ...


@_register_object("script.ir_builder.tirx.ThenFrame")
class ThenFrame(TIRFrame): ...


@_register_object("script.ir_builder.tirx.ElseFrame")
class ElseFrame(TIRFrame): ...


@_register_object("script.ir_builder.tirx.LaunchThreadFrame")
class LaunchThreadFrame(TIRFrame):
    def __enter__(self) -> Var:
        super().__enter__()
        return self.iter_var.var
