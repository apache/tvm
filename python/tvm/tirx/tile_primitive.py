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
"""TIRx tile primitive IR nodes: LambdaExpr, DispatchContext, TilePrimitiveCall.

Mirrors the C++ header ``include/tvm/tirx/tile_primitive.h``.
"""
# pylint: disable=no-member

import inspect
from collections.abc import Callable
from typing import Any, ClassVar

import tvm_ffi
from tvm_ffi import register_object

from tvm.ir import Op, PrimExpr, Range
from tvm.runtime import Object, Scriptable
from tvm.target import Target

from . import _ffi_api
from .buffer import Buffer
from .exec_scope import ExecScope
from .expr import FloatImm, IterVar, Var
from .stmt import Stmt


@register_object("tirx.LambdaExpr")
class LambdaExpr(Object):
    """A reified Python lambda: bound variables and a body over them.

    Used by tile primitive ops that take a per-element expression over the
    destination axes (e.g. ``tirx.tile.select``).
    """

    vars: list[Var]
    pred: PrimExpr

    def __init__(self, f_pred: Callable[..., PrimExpr]):
        vars = [Var(name, "int32") for name in inspect.signature(f_pred).parameters]
        pred = f_pred(*vars)
        self.__init_handle_by_constructor__(_ffi_api.LambdaExpr, vars, pred)

    def apply(self, indices: list[PrimExpr]) -> PrimExpr:
        """Substitute the bound variables with the given indices, returning the body."""
        return _ffi_api.LambdaExprApply(self, indices)


@register_object("tirx.DispatchContext")
class DispatchContext(Object, Scriptable):
    """DispatchContext node.

    Parameters
    ----------
    target : Target
        The target of the dispatch context.

    exec_scope : ExecScope
        The execution scope of the dispatch context.

    launch_params : Dict[str, PrimExpr]
        The launch parameters of the dispatch context.

    var_range_map : Dict[Var, Range]
        A map from loop variables to their ranges.

    callbacks : Dict[str, Object]
        The callbacks of the dispatch context.

    shared_state : Dict[str, Object]
        Shared state persisting across dispatch calls within a single lowering pass.
    """

    target: Target
    exec_scope: ExecScope
    launch_params: dict[str, IterVar]
    var_range_map: dict[Var, Range]
    alloc_only: bool
    callbacks: dict[str, Object]
    shared_state: dict[str, Object]
    inter: dict[str, list]
    intra: dict[str, list]
    scope_kind: str

    kPrivateAlloc = "private_alloc"
    kDeviceInitStmt = "device_init_stmt"
    kHostInitStmt = "host_init_stmt"
    kPostBufferDefStmt = "post_buffer_def_stmt"

    def __init__(
        self,
        target: Target,
        exec_scope: ExecScope,
        launch_params: dict[str, IterVar],
        var_range_map: dict[Var, Range],
        alloc_only: bool = False,
        callbacks: dict[str, Object] = {},
        shared_state: dict[str, Object] = {},
        inter: dict[str, list] | None = None,
        intra: dict[str, list] | None = None,
        scope_kind: str = "",
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.DispatchContext,  # pylint: disable=no-member
            target,
            exec_scope,
            launch_params,
            var_range_map,
            alloc_only,
            callbacks,
            shared_state,
            inter or {},
            intra or {},
            scope_kind,
        )

    def add_alloc_buffer(self, buffer: Buffer) -> None:
        """Add an allocated buffer to the dispatch context.
           Can be called only if alloc_only is True.
           The buffer will be added to the workspace of operator (the key in the workspace is the buffer name).

        Parameters
        ----------
        buffer : Buffer
            The buffer to be added.
        """  # noqa: E501
        _ffi_api.DispatchContextAddAllocBuffer(self, buffer)  # pylint: disable=no-member

    def add_init_stmt(self, stmt: Stmt, host: bool = False) -> None:
        """Add an initialization statement to the dispatch context.
           Device initialization statements is only allowed if alloc_only is True.
           Host initialization statements will be ignored if alloc_only is True.
           The statements will be added to the beginning of the kernel.

        Parameters
        ----------
        stmt : Stmt
            The initialization statement to be added.
        host : bool
            Whether the statement is a host statement.
            If True, the statement will be added to the host code (before the kernel).
            If False, the statement will be added to the kernel body (at the beginning of the kernel).
        """  # noqa: E501
        _ffi_api.DispatchContextAddInitStmt(self, stmt, host)  # pylint: disable=no-member

    def add_post_buffer_def_stmt(self, buffer: Buffer, stmt: Stmt) -> None:
        """Add a statement to be inserted after a buffer's definition (DeclBuffer/AllocBuffer).

        Parameters
        ----------
        buffer : Buffer
            The buffer whose definition scope the statement should appear in.
        stmt : Stmt
            The statement to be inserted.
        """
        _ffi_api.DispatchContextAddPostBufferDefStmt(self, buffer, stmt)  # pylint: disable=no-member

    def cache_get(self, key: str) -> Object | None:
        """Look up a cached value by key.

        Parameters
        ----------
        key : str
            Cache key (built by the caller from construction parameters).

        Returns
        -------
        Optional[Object]
            The cached value, or None on miss.
        """
        return _ffi_api.DispatchContextSharedStateGet(self, key)

    def cache_set(self, key: str, value: Object) -> None:
        """Store a value in the cross-dispatch cache.

        Parameters
        ----------
        key : str
            Cache key (built by the caller from construction parameters).
        value : Object
            The object to cache (e.g. a Buffer or Var).
        """
        _ffi_api.DispatchContextSharedStateSet(self, key, value)

    def is_cuda(self) -> bool:
        """Check if the target is CUDA."""
        return self.target.kind.name == "cuda"

    def is_trn(self) -> bool:
        """Check if the target is Trainium."""
        return self.target.kind.name == "trn"

    def is_target(self, name: str) -> bool:
        """Check if the target kind matches ``name``."""
        return self.target.kind.name == name

    # -- scope predicates ----------------------------------------------------
    #
    # Each ``is_<scope>`` returns True iff the op site is at that scope kind.
    # Backed by ``self.scope_kind``, which 1-1 maps to a canonical intra
    # TileLayout shape:
    #   thread     -> {}
    #   warp       -> {laneid}
    #   warpgroup  -> {laneid, wid_in_wg}
    #   cta        -> {laneid, warpid}
    #   cluster    -> {laneid, warpid, cta_id}
    #
    # Prefer these predicates over raw ``self.scope_kind == "..."`` comparisons
    # so dispatchers that later need stricter intra/inter shape checks can
    # tighten the predicate body without touching every call site.

    @property
    def is_thread(self) -> bool:
        return self.scope_kind == "thread"

    @property
    def is_warp(self) -> bool:
        return self.scope_kind == "warp"

    @property
    def is_warpgroup(self) -> bool:
        return self.scope_kind == "warpgroup"

    @property
    def is_cta(self) -> bool:
        return self.scope_kind == "cta"

    @property
    def is_cluster(self) -> bool:
        return self.scope_kind == "cluster"


def normalize_const_arg(arg) -> PrimExpr:
    if isinstance(arg, float):
        return FloatImm("float32", arg)
    return arg


@tvm_ffi.register_object("tirx.TilePrimitiveCall")
class TilePrimitiveCall(Stmt):
    """TilePrimitiveCall node.

    Parameters
    ----------
    op : Op
        The operator.

    args : List[PrimExpr]
        The arguments.

    workspace : Map[str, Buffer]
        The workspace.

    config : Map[str, ObjectRef]
        The scheduler/config dictionary.

    dispatch : Optional[str]
        The explicit variant name to dispatch to.

    scope : ExecScope
        The cooperation scope of this call. Defaults to ``thread`` (an unscoped call).
    """

    args: list[PrimExpr]
    workspace: dict[str, Buffer]
    config: dict[str, Any]
    dispatch: str | None
    scope: ExecScope
    _registry: ClassVar[dict[Op, type["TilePrimitiveCall"]]] = {}

    def __init__(
        self,
        *args: list[PrimExpr],
        op: Op | None = None,
        workspace: dict[str, Buffer] | None = None,
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
            assert self.__class__ != TilePrimitiveCall, (
                "Directly instantiating TilePrimitiveCall needs to specify the op"
            )
            op = self.__class__.op
        args = list(map(normalize_const_arg, args))
        self.__init_handle_by_constructor__(
            _ffi_api.TilePrimitiveCall,
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
    def downcast(cls, instance: "TilePrimitiveCall") -> "TilePrimitiveCall":
        subclass = cls._registry.get(instance.op)
        if subclass is None:
            return instance  # Unknown op: return as-is
        new_instance = subclass.__new__(subclass)
        new_instance.__init_handle_by_constructor__(
            _ffi_api.TilePrimitiveCallCopyHandle,
            instance,  # pylint: disable=no-member
        )
        return new_instance

    def replace(self, **changes: Any) -> "TilePrimitiveCall":
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
        new_call : TilePrimitiveCall
            A new call with the requested fields replaced.
        """
        unknown = set(changes) - {"op", "args", "workspace", "config", "dispatch", "scope"}
        if unknown:
            raise TypeError(f"Unknown field(s) for TilePrimitiveCall.replace: {sorted(unknown)}")
        new_call = TilePrimitiveCall(
            *changes.get("args", self.args),
            op=changes.get("op", self.op),
            workspace=changes.get("workspace", self.workspace),
            config=changes.get("config", self.config),
            dispatch=changes.get("dispatch", self.dispatch),
            scope=changes.get("scope", self.scope),
        )
        return TilePrimitiveCall.downcast(new_call)

    def with_workspace(self, workspace: dict[str, Buffer]) -> "TilePrimitiveCall":
        """Return a copy with ``workspace`` replaced, preserving all other fields."""
        return self.replace(workspace=workspace)

    @property
    def srcs(self) -> list[PrimExpr]:
        raise NotImplementedError("Subclass must implement this method")

    @property
    def dsts(self) -> list[PrimExpr]:
        raise NotImplementedError("Subclass must implement this method")

    def get_private_buffers(
        self, buffer_dict: dict[Any, tuple[Buffer, Stmt | None]], sctx: "DispatchContext"
    ) -> dict[str, Any]:
        """
        Create private (intermediate) buffers needed in this operator.

        Parameters
        ----------
        buffer_dict: Dict[Any, Tuple[Buffer, Optional[Stmt]]]
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
        self, buffer_dict: dict[Any, tuple[Buffer, Stmt | None]], sctx: "DispatchContext"
    ) -> dict[str, Any]:
        return {}

    def get_private_buffers_cuda(
        self, buffer_dict: dict[Any, tuple[Buffer, Stmt | None]], sctx: "DispatchContext"
    ) -> dict[str, Any]:
        return {}

    def validate(self) -> None:
        pass
