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
from typing import List, Union

from tvm._ffi import register_object as _register_object
from tvm.tir import Buffer, Var

from ..base import IRBuilderFrame


@_register_object("script.ir_builder.tir.TIRFrame")
class TIRFrame(IRBuilderFrame):
    ...


@_register_object("script.ir_builder.tir.PrimFuncFrame")
class PrimFuncFrame(TIRFrame):
    ...


@_register_object("script.ir_builder.tir.BlockFrame")
class BlockFrame(TIRFrame):
    ...


@_register_object("script.ir_builder.tir.BlockInitFrame")
class BlockInitFrame(TIRFrame):
    ...


@_register_object("script.ir_builder.tir.ForFrame")
class ForFrame(TIRFrame):
    def __enter__(self) -> Union[Var, List[Var]]:  # type: ignore[override]
        super().__enter__()
        return self.vars if len(self.vars) > 1 else self.vars[0]


@_register_object("script.ir_builder.tir.AssertFrame")
class AssertFrame(TIRFrame):
    ...


@_register_object("script.ir_builder.tir.LetFrame")
class LetFrame(TIRFrame):
    ...


@_register_object("script.ir_builder.tir.RealizeFrame")
class RealizeFrame(TIRFrame):
    ...


@_register_object("script.ir_builder.tir.AllocateFrame")
class AllocateFrame(TIRFrame):
    def __enter__(self) -> Buffer:
        super().__enter__()
        return self.buffer_var


@_register_object("script.ir_builder.tir.AllocateConstFrame")
class AllocateConstFrame(TIRFrame):
    def __enter__(self) -> Buffer:
        super().__enter__()
        return self.buffer_var


@_register_object("script.ir_builder.tir.AttrFrame")
class AttrFrame(TIRFrame):
    ...


@_register_object("script.ir_builder.tir.WhileFrame")
class WhileFrame(TIRFrame):
    ...


@_register_object("script.ir_builder.tir.IfFrame")
class IfFrame(TIRFrame):
    ...


@_register_object("script.ir_builder.tir.ThenFrame")
class ThenFrame(TIRFrame):
    ...


@_register_object("script.ir_builder.tir.ElseFrame")
class ElseFrame(TIRFrame):
    ...


@_register_object("script.ir_builder.tir.DeclBufferFrame")
class DeclBufferFrame(TIRFrame):
    def __enter__(self) -> Buffer:
        super().__enter__()
        return self.buffer


@_register_object("script.ir_builder.tir.LaunchThreadFrame")
class LaunchThreadFrame(TIRFrame):
    ...
