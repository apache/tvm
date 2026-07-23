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
# pylint: disable=no-member, super-init-not-called

"""Definition of execution scope."""

from tvm_ffi import register_object

from tvm.runtime import Object

from . import _ffi_api
from .expr import Expr, Var


@register_object("tirx.ScopeIdDef")
class ScopeIdDef(Object):
    """Definition of scope identifiers with their extents and parent-child relationships.

    The constructor accepts ``parent`` and ``cur`` as scope-name strings; they
    are converted by the FFI into the closed ``ScopeBinding`` enum and stored
    on the ``scope`` field (an ``int`` value of that enum).

    ``extents=None`` defers the extent: the value is inferred from sibling
    ScopeIdDef relationships at LowerTIRx entry via the verifier's closure.
    Deferred form requires ``def_ids`` to contain exactly one Var.
    """

    def_ids: list[Var]
    extents: list[Expr] | None
    scope: int

    def __init__(
        self,
        def_ids: list[Var],
        extents: list[Expr] | None,
        parent: str,
        cur: str,
        preferred_extents: list[Expr] | None = None,
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.ScopeIdDef, def_ids, extents, parent, cur, preferred_extents
        )


_SCOPE_KIND_TO_NAME = {
    2: "cluster",
    3: "cta",
    4: "warpgroup",
    5: "warp",
    6: "thread",
}


# Mirror of ``enum class ScopeBinding`` in tvm/tirx/exec_scope.h. Maps the
# ``int`` value of ``ScopeIdDef.scope`` back to the ``(parent, cur)`` pair
# that ``ScopeIdDef.__init__`` accepts — needed when Python code wants to
# rebuild a ``ScopeIdDef`` from an existing one (e.g. a StmtMutator
# walking and rewriting extents).
_SCOPE_BINDING_TO_PARENT_CUR = {
    0: ("kernel", "cluster"),
    1: ("kernel", "cta"),
    2: ("cluster", "cta"),
    3: ("cta", "warpgroup"),
    4: ("cta", "warp"),
    5: ("warpgroup", "warp"),
    6: ("warp", "thread"),
    7: ("cta", "thread"),
    8: ("warpgroup", "thread"),
    9: ("cluster", "cta_pair"),
}


@register_object("tirx.ExecScope")
class ExecScope(Object):
    """An execution scope, identified by one of {cluster, cta, warpgroup, warp,
    thread}. The ctor FATALs on any other name."""

    kind: int
    scope_id_def: list[ScopeIdDef]

    def __init__(self, name: str):
        self.__init_handle_by_constructor__(_ffi_api.ExecScope, name)

    @property
    def name(self) -> str:
        """Human-readable name of this scope (derived from ``kind``)."""
        return _SCOPE_KIND_TO_NAME[self.kind]
