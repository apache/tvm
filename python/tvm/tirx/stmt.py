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

from typing import TYPE_CHECKING, Any, ClassVar

import tvm_ffi

from tvm import ir as _ir
from tvm.ir import Expr, Op, Range, Span, TensorRegion, Var

from . import _ffi_api
from .exec_scope import ExecScope, ScopeIdDef

if TYPE_CHECKING:
    from .tile_primitive import DispatchContext


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


@tvm_ffi.register_object("tirx.TileOpCall")
class TileOpCall(_ir.Stmt):
    """TileOpCall node.

    Parameters
    ----------
    op : Op
        The operator.

    args : List[Expr]
        The arguments.

    workspace : Map[str, Var]
        The workspace.

    config : Map[str, Expr]
        The scheduler/config dictionary. Omit unused keys; explicit None values
        are invalid. max_inst_size=-1 means unbounded, while omission retains
        the backend default. A present gather4 contains exactly four coordinates.

    dispatch : Optional[str]
        The explicit variant name to dispatch to.

    scope : ExecScope
        The cooperation scope of this call. Defaults to ``thread`` (an unscoped call).
    """

    args: list[Expr]
    workspace: dict[str, Var]
    config: dict[str, Expr]
    dispatch: str | None
    scope: ExecScope
    _registry: ClassVar[dict[Op, type["TileOpCall"]]] = {}

    def __init__(
        self,
        *args: list[Expr],
        op: Op | None = None,
        workspace: dict[str, Var] | None = None,
        config: dict[str, Any] | None = None,
        dispatch: str | None = None,
        scope: ExecScope | None = None,
    ) -> None:
        if workspace is None:
            workspace = {}
        if config is None:
            config = {}
        if scope is None:
            scope = ExecScope("thread")
        if op is None:
            assert self.__class__ != TileOpCall, (
                "Directly instantiating TileOpCall needs to specify the op"
            )
            op = self.__class__.op
        self.__init_handle_by_constructor__(
            _ffi_api.TileOpCall,
            op,
            args,
            workspace,
            config,
            dispatch,
            scope,  # pylint: disable=no-member
        )

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "op"):
            cls._registry[cls.op] = cls

    @classmethod
    def downcast(cls, instance: "TileOpCall") -> "TileOpCall":
        subclass = cls._registry.get(instance.op)
        if subclass is None:
            return instance  # Unknown op: return as-is
        new_instance = subclass.__new__(subclass)
        new_instance.__init_handle_by_constructor__(
            _ffi_api.TileOpCallCopyHandle,
            instance,  # pylint: disable=no-member
        )
        return new_instance

    def replace(self, **changes: Any) -> "TileOpCall":
        """Return a copy of this call with selected fields replaced.

        Every field that is not overridden in ``changes`` is preserved from
        ``self`` (including ``scope``), so rebuilds never silently drop fields.
        The returned node is downcast to the registered subclass for ``op``.

        Parameters
        ----------
        **changes : Any
            Field overrides; any of ``op``, ``args``, ``workspace``, ``config``,
            ``dispatch``, ``scope``.

        Returns
        -------
        new_call : TileOpCall
            A new call with the requested fields replaced.
        """
        unknown = set(changes) - {"op", "args", "workspace", "config", "dispatch", "scope"}
        if unknown:
            raise TypeError(f"Unknown field(s) for TileOpCall.replace: {sorted(unknown)}")
        new_call = TileOpCall(
            *changes.get("args", self.args),
            op=changes.get("op", self.op),
            workspace=changes.get("workspace", self.workspace),
            config=changes.get("config", self.config),
            dispatch=changes.get("dispatch", self.dispatch),
            scope=changes.get("scope", self.scope),
        )
        return TileOpCall.downcast(new_call)

    def with_workspace(self, workspace: dict[str, Var]) -> "TileOpCall":
        """Return a copy with ``workspace`` replaced, preserving all other fields."""
        return self.replace(workspace=workspace)

    @property
    def srcs(self) -> list[Expr]:
        raise NotImplementedError("Subclass must implement this method")

    @property
    def dsts(self) -> list[Expr]:
        raise NotImplementedError("Subclass must implement this method")

    def get_private_buffers(
        self, buffer_dict: dict[Any, tuple[Var, _ir.Stmt | None]], sctx: "DispatchContext"
    ) -> dict[str, Any]:
        """
        Create private (intermediate) buffers needed in this operator.

        Parameters
        ----------
        buffer_dict: Dict[Any, Tuple[Var, Optional[tvm.ir.Stmt]]]
            A dictionary containing private buffers (and their init stmts) in other operators.
            Key can be anything to reference the buffer.
            This is used to reuse private buffers in other operators (like identity tensor etc.).
            If the buffer is not found in the buffer_dict, it will be created and added to
            the buffer_dict.
            If the buffer is found in the buffer_dict but smaller than required, it will be
            enlarged and updated.

        sctx: DispatchContext
            The dispatch context.
            This is used to get the target and reuse op dispatch implementations.

        Returns:
        -------
        private_buffer_refs: Dict[str, Any]
            The references to private buffers created in this operator.
            Key will be the name to add into workspace.
            private buffer can be accessed by buffer_dict[private_buffer_refs[name]]
        """
        if sctx.target.kind.name == "trn":
            return self.get_private_buffers_trn(buffer_dict, sctx)
        elif sctx.target.kind.name == "cuda":
            return self.get_private_buffers_cuda(buffer_dict, sctx)
        else:
            raise ValueError(f"Unsupported target: {sctx.target.kind.name}")

    def get_private_buffers_trn(
        self, buffer_dict: dict[Any, tuple[Var, _ir.Stmt | None]], sctx: "DispatchContext"
    ) -> dict[str, Any]:
        return {}

    def get_private_buffers_cuda(
        self, buffer_dict: dict[Any, tuple[Var, _ir.Stmt | None]], sctx: "DispatchContext"
    ) -> dict[str, Any]:
        return {}

    def validate(self) -> None:
        pass
