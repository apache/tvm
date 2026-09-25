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

from collections.abc import Sequence

from tvm_ffi import Array
from tvm_ffi import register_object as _register_object

from tvm.script.ir_builder.base import IRBuilderFrame
from tvm.tirx import Buffer, Var

from . import _ffi_api


@_register_object("script.ir_builder.tirx.TIRFrame")
class TIRFrame(IRBuilderFrame): ...


@_register_object("script.ir_builder.tirx.PrimFuncFrame")
class PrimFuncFrame(TIRFrame):
    """Native function frame retaining signature and finalized results."""

    def default_buffer_layout(self, shape, scope):
        """Select the default layout for buffers constructed in this function."""
        from tvm.tirx.layout import S, TileLayout

        return None if scope in ("trn.sbuf", "trn.psum") else TileLayout(S[tuple(shape)])

    @property
    def params(self):
        """The native declared parameters, shared with the resumed body."""
        return self.args


@_register_object("script.ir_builder.tirx.ForFrame")
class ForFrame(TIRFrame):
    def set_names(self, names: str | Sequence[str] | None) -> None:
        """Configure final variable names before entry, preserving native identities.

        A starred target expands to the remaining dimensions. Omitted names
        preserve defaults supplied by native construction; no names are replayed
        during entry.
        """
        _ffi_api.ForFrameSetNames(self, names)

    def __enter__(self) -> Var | Array[Var]:  # type: ignore[override]
        """Enter with one variable directly, or the native sequence for multiple loops."""
        super().__enter__()
        return self.vars[0] if len(self.vars) == 1 else self.vars


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


@_register_object("script.ir_builder.tirx.DeclBufferFrame")
class DeclBufferFrame(TIRFrame):
    def __enter__(self) -> Buffer:
        super().__enter__()
        return self.buffer


@_register_object("script.ir_builder.tirx.LaunchThreadFrame")
class LaunchThreadFrame(TIRFrame):
    def __enter__(self) -> Var:
        super().__enter__()
        return self.iter_var.var


@_register_object("script.ir_builder.tirx.HintFrame")
class HintFrame(TIRFrame): ...
