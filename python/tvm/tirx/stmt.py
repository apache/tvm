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
"""TIRx-specific statement nodes and buffer-region construction."""

import tvm_ffi

from tvm import ir as _ir
from tvm.ir import Range, Span, TensorRegion, Var

from . import _ffi_api
from .exec_scope import ScopeIdDef


def BufferRegion(buffer: Var, region: list[Range]) -> TensorRegion:
    """Construct a buffer-backed tensor region with TIRX subscript semantics.

    Parameters
    ----------
    buffer : Var
        The source buffer.

    region : List[Range]
        The ranges, with one entry for each buffer dimension.
    """
    return _ffi_api.BufferRegion(buffer, region)


@tvm_ffi.register_object("tirx.ScopeIdDefStmt")
class ScopeIdDefStmt(_ir.Stmt):
    """ScopeIdDefStmt node.

    Leaf statement that introduces scope-identifier vars
    (``wg_id = Tx.warpgroup_id([N])``, ``warp_id = Tx.warp_id_in_wg([4])``,
    ``lane_id = Tx.lane_id([32])``, …) at the kernel-body top level. The
    underlying ``ScopeIdDef`` carries the def vars, their extents, and
    the parent/child scope binding.

    Note: the C++ field is named ``def`` (a Python keyword). Access it
    via ``getattr(stmt, "def")`` or ``stmt.__getattribute__("def")`` —
    the type-annotation alias here is purely for documentation.

    Parameters
    ----------
    def_ : ScopeIdDef
        The scope-id definition (def vars, extents, scope binding).

    span : Optional[Span]
        The location of this statement in the source code.
    """

    span: Span | None

    def __init__(self, def_: ScopeIdDef, span: Span | None = None) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.ScopeIdDefStmt,  # type: ignore
            def_,
            span,
        )  # type: ignore
