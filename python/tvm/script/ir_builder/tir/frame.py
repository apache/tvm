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
from tvm.tir import Var

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


@_register_object("script.ir_builder.tir.ForFrame")
class ForFrame(TIRFrame):
    def __enter__(self) -> Union[Var, List[Var]]:
        super().__enter__()
        return self.vars if len(self.vars) > 1 else self.vars[0]
