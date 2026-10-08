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
"""Statement AST Node in TVM.

Each statement node have subfields that can be visited from python side.

.. code-block:: python

    x = tvm.tirx.Var("n", "int32")
    buffer = tvm.tirx.decl_tensor((16,), "float32")
    st = tvm.tirx.stmt.TensorStore(buffer, 1, (x,))
    assert isinstance(st, tvm.tirx.stmt.TensorStore)
    assert(st.buffer == buffer)
"""

from collections.abc import Mapping, Sequence
from enum import IntEnum
from typing import Any

import tvm_ffi

from tvm.ir import DictAttrs, Expr, Op, Range, Span, StringImm, TensorRegion, Type, Var, make_node
from tvm.runtime import Object, Scriptable

from . import _ffi_api
from .exec_scope import ScopeIdDef


@tvm_ffi.register_object("tirx.Stmt")
class Stmt(Object, Scriptable):
    """Base class of all the statements."""


@tvm_ffi.register_object("tirx.Bind")
class Bind(Stmt):
    """Bind node.

    Bind a variable to a value in the enclosing scope.
    Bind has no body field.
    The bound variable is visible in all subsequent statements
    within the same enclosing scope (SeqStmt, ForNode.body, etc.).

    Parameters
    ----------
    var : Var
        The variable in the binding.

    value : Expr
        The value to be bound.

    span : Optional[Span]
        The location of the stmt in the source code.
    """

    var: Var
    value: Expr
    span: Span | None

    def __init__(self, var: Var, value: Expr, span: Span | None = None) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.Bind,
            var,
            value,
            span,  # type: ignore
        )


@tvm_ffi.register_object("tirx.AssertStmt")
class AssertStmt(Stmt):
    """AssertStmt node.

    Parameters
    ----------
    kind : StringImm
        The error kind, e.g. "RuntimeError", "TypeError", "ValueError".

    condition : Expr
        The assert condition.

    message_parts : list[StringImm]
        Error message fragments, concatenated at runtime when assertion fails.

    span : Span | None
        The location of the stmt in the source code.
    """

    kind: StringImm
    condition: Expr
    message_parts: list
    span: Span | None

    def __init__(
        self,
        kind: StringImm,
        condition: Expr,
        message_parts: list | None = None,
        span: Span | None = None,
    ) -> None:
        if message_parts is None:
            message_parts = []
        self.__init_handle_by_constructor__(
            _ffi_api.AssertStmt,
            kind,
            condition,
            message_parts,
            span,  # type: ignore
        )


class ForKind(IntEnum):
    """The kind of the for loop.

    note
    ----
    ForKind can change the control flow semantics
    of the loop and need to be considered in all TIR passes.
    """

    DEFAULT = 0  # Ordinary loop with sequential iteration semantics.
    PARALLEL = 1  # Parallel execution, optionally with explicit thread placement.
    VECTORIZED = 2
    UNROLLED = 3


@tvm_ffi.register_object("tirx.For")
class For(Stmt):
    """For node.

    Parameters
    ----------
    loop_var : Var
        The loop variable.

    min : Expr
        The beginning value.

    extent : Expr
        The length of the loop.

    kind : ForKind
        The type of the for.

    body : Stmt | Sequence[Stmt]
        The body statement.

    annotations: Optional[Mapping[str, Object]]
        Additional loop annotations. For parallel loops, the optional
        ``thread_binding`` entry is a string naming the bound thread.
        Parallel loops without this entry use CPU parallel execution.

    step : Expr
        The loop step. Defaults to None, which represents one.

    span : Optional[Span]
        The location of the stmt in the source code.
    """

    loop_var: Var
    min: Expr
    extent: Expr
    kind: ForKind
    body: "SeqStmt"
    annotations: Mapping[str, Object]
    step: Expr | None
    span: Span | None

    def __init__(
        self,
        loop_var: Var,
        min: Expr,  # pylint: disable=redefined-builtin
        extent: Expr,
        kind: ForKind,
        body: Stmt | Sequence[Stmt],
        annotations: Mapping[str, Object] | None = None,
        step: Expr | None = None,
        span: Span | None = None,
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.For,  # type: ignore
            loop_var,
            min,
            extent,
            kind,
            body,
            annotations,
            step,
            span,
        )


@tvm_ffi.register_object("tirx.While")
class While(Stmt):
    """While node.

    Parameters
    ----------
    condition : Expr
        The termination condition.

    body : Stmt | Sequence[Stmt]
        The body statement.

    span : Optional[Span]
        The location of the stmt in the source code.
    """

    condition: Expr
    body: "SeqStmt"
    span: Span | None

    def __init__(
        self, condition: Expr, body: Stmt | Sequence[Stmt], span: Span | None = None
    ) -> None:
        self.__init_handle_by_constructor__(_ffi_api.While, condition, body, span)  # type: ignore


@tvm_ffi.register_object("tirx.TensorStore")
class TensorStore(Stmt):
    """Store a value into a tensor variable.

    Parameters
    ----------
    buffer : Var
        The buffer.

    value : Expr
        The value we to be stored.

    indices : List[Expr]
        The indices location to be stored.

    span : Optional[Span]
        The location of the stmt in the source code.
    """

    buffer: Var
    value: Expr
    indices: list[Expr]
    span: Span | None

    def __init__(
        self,
        buffer: Var,
        value: Expr,
        indices: list[Expr],
        span: Span | None = None,
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.TensorStore,
            buffer,
            value,
            indices,
            span,  # type: ignore
        )


@tvm_ffi.register_object("tirx.RegionStmt")
class RegionStmt(Stmt):
    """An operation with enclosing-scope operands and one lexical body.

    ``body_params`` define variables visible only within ``body``. Their count
    and types must match the operation's required ``FRegionGetBodyParams`` hook.
    Zero-parameter regions register an empty-return hook; operations without a
    hook do not support region construction. Explicit parameter
    identities and their references in ``body`` are preserved. ``result_vars``
    define variables after the region in the enclosing sequence. Attributes are
    evaluated outside the body-parameter scope. Direct construction and JSON
    serialization support result variables; structured script syntax currently
    supports only result-free regions.
    """

    op: Op
    args: list[Expr]
    body_params: list[Var]
    attrs: DictAttrs
    body: "SeqStmt"
    result_vars: list[Var]
    span: Span | None

    def __init__(
        self,
        op: Op | str,
        args: Sequence[Expr],
        body_params: Sequence[Var],
        attrs: DictAttrs | Mapping[str, Any] | None,
        body: Stmt | Sequence[Stmt],
        result_vars: Sequence[Var] | None = None,
        span: Span | None = None,
    ) -> None:
        if isinstance(op, str):
            op = Op.get(op)
        if attrs is None or isinstance(attrs, Mapping):
            attrs = make_node("ir.DictAttrs", **(attrs or {}))
        self.__init_handle_by_constructor__(
            _ffi_api.RegionStmt,
            op,
            args,
            body_params,
            attrs,
            body,
            [] if result_vars is None else result_vars,
            span,
        )


@tvm_ffi.register_object("tirx.SeqStmt")
class SeqStmt(Stmt):
    """Sequence of statements.

    Parameters
    ----------
    seq : Stmt | Sequence[Stmt]
        The statements, flattened into one sequence. Empty and singleton sequences are valid.

    span : Optional[Span]
        The location of the stmt in the source code.
    """

    seq: list[Stmt]
    span: Span | None

    def __init__(self, seq: Stmt | Sequence[Stmt], span: Span | None = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.SeqStmt, seq, span)  # type: ignore

    def __getitem__(self, i: int):
        return self.seq[i]

    def __len__(self):
        return len(self.seq)


@tvm_ffi.register_object("tirx.IfThenElse")
class IfThenElse(Stmt):
    """IfThenElse node.

    Parameters
    ----------
    condition : Expr
        The expression

    then_case : Stmt | Sequence[Stmt]
        The statement to execute if condition is true.

    else_case : Stmt | Sequence[Stmt] | None
        The statement to execute if condition is false.

    span : Optional[Span]
        The location of the stmt in the source code.
    """

    condition: Expr
    then_case: SeqStmt
    else_case: SeqStmt | None

    def __init__(
        self,
        condition: Expr,
        then_case: Stmt | Sequence[Stmt],
        else_case: Stmt | Sequence[Stmt] | None,
        span: Span | None = None,
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.IfThenElse,
            condition,
            then_case,
            else_case,
            span,  # type: ignore
        )


@tvm_ffi.register_object("tirx.Evaluate")
class Evaluate(Stmt):
    """Evaluate node.

    Parameters
    ----------
    value : Expr
        The expression to be evaluated.

    span : Optional[Span]
        The location of the stmt in the source code.
    """

    value: Expr
    span: Span | None

    def __init__(self, value: Expr, span: Span | None = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.Evaluate, value, span)  # type: ignore


@tvm_ffi.register_object("tirx.BufferRegionType")
class BufferRegionType(Type):
    """The TIRX subscript type of a buffer-backed :class:`tvm.ir.TensorRegion`."""

    def __init__(self) -> None:
        self.__init_handle_by_constructor__(_ffi_api.BufferRegionType)  # type: ignore


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
class ScopeIdDefStmt(Stmt):
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


@tvm_ffi.register_object("tirx.Break")
class Break(Stmt):
    """Break node.

    Parameters
    ----------
    """

    def __init__(self, span: Span | None = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.Break, span)  # type: ignore


@tvm_ffi.register_object("tirx.Return")
class Return(Stmt):
    """Return node.

    Parameters
    ----------
    value : Expr
        The value to return.

    span : Optional[Span]
        The location of this statement in the source code.
    """

    value: Expr
    span: Span | None

    def __init__(self, value: Expr, span: Span | None = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.Return, value, span)  # type: ignore


@tvm_ffi.register_object("tirx.Continue")
class Continue(Stmt):
    """Continue node.

    Parameters
    ----------
    """

    def __init__(self, span: Span | None = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.Continue, span)  # type: ignore


def stmt_seq(*args: Expr | Stmt) -> SeqStmt:
    """Make sequence of statements

    Parameters
    ----------
    *args : Union[Expr, Stmt]
        List of statements to be combined as sequence.

    Returns
    -------
    stmt : SeqStmt
        The combined sequence.
    """
    ret = []
    for value in args:
        if not isinstance(value, Stmt):
            value = Evaluate(value)
        ret.append(value)
    return SeqStmt(ret)


def stmt_list(stmt: Stmt) -> list[Stmt]:
    """Make list of stmt from blocks.

    Parameters
    ----------
    stmt : Stmt
        The input statement.

    Returns
    -------
    stmt_list : List[Stmt]
        The unpacked list of statements
    """
    if isinstance(stmt, SeqStmt):
        return list(stmt.seq)
    return [stmt]


# Source-compatibility re-export: TilePrimitiveCall lives in tile_primitive.py
# after the tile-primitive module merge. Imported last to avoid a cycle with
# tile_primitive's own ``from .stmt import Stmt``.
from .tile_primitive import TilePrimitiveCall  # noqa: E402,F401  isort: skip
