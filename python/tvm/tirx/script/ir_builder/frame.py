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

from tvm.script.ir_builder.frame import AssertFrame as AssertFrame
from tvm.script.ir_builder.frame import ElseFrame as ElseFrame
from tvm.script.ir_builder.frame import ForFrame as ForFrame
from tvm.script.ir_builder.frame import IfFrame as IfFrame
from tvm.script.ir_builder.frame import RegionFrame as RegionFrame
from tvm.script.ir_builder.frame import StmtFrame
from tvm.script.ir_builder.frame import ThenFrame as ThenFrame
from tvm.script.ir_builder.frame import WhileFrame as WhileFrame


@_register_object("script.ir_builder.tirx.TIRFrame")
class TIRFrame(StmtFrame): ...


@_register_object("script.ir_builder.tirx.FunctionFrame")
class FunctionFrame(TIRFrame):
    """Native function frame retaining signature and finalized results."""

    def default_buffer_layout(self, shape, scope):
        """Select the default layout for buffers constructed in this function."""
        from tvm.tirx.layout import S, TileLayout

        return None if scope in ("trn.sbuf", "trn.psum") else TileLayout(S[tuple(shape)])

    @property
    def params(self):
        """The native declared parameters, shared with the resumed body."""
        return self.args
