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
    st = tvm.tirx.stmt.BufferStore(buffer, 1, (x,))
    assert isinstance(st, tvm.tirx.stmt.BufferStore)
    assert(st.buffer == buffer)
"""

from collections.abc import Mapping
from enum import IntEnum
from typing import Any

import tvm_ffi

from tvm.ir import Expr, Range, Span, StringImm, TensorRegion, Type
from tvm.runtime import Object, Scriptable

from . import _ffi_api
from .buffer import Buffer
from .exec_scope import ScopeIdDef
from .expr import IterVar, Var


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

    SERIAL = 0
    PARALLEL = 1
    VECTORIZED = 2
    UNROLLED = 3
    THREAD_BINDING = 4  # pylint: disable=invalid-name


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

    body : Stmt
        The body statement.

    thread_binding: Optional[tirx.IterVar]
        The thread this loop binds to. Only valid
        if kind is ThreadBinding

    step : Expr
        The loop step. Default to none which
        represent one.

    annotations: Optional[Mapping[str, Object]]
        Additional annotation hints.

    span : Optional[Span]
        The location of the stmt in the source code.
    """

    loop_var: Var
    min: Expr
    extent: Expr
    kind: ForKind
    body: Stmt
    thread_binding: IterVar | None
    annotations: Mapping[str, Object]
    step: Expr | None
    span: Span | None

    def __init__(
        self,
        loop_var: Var,
        min: Expr,  # pylint: disable=redefined-builtin
        extent: Expr,
        kind: ForKind,
        body: Stmt,
        thread_binding: IterVar | None = None,
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
            thread_binding,
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

    body : Stmt
        The body statement.

    span : Optional[Span]
        The location of the stmt in the source code.
    """

    condition: Expr
    body: Stmt
    span: Span | None

    def __init__(self, condition: Expr, body: Stmt, span: Span | None = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.While, condition, body, span)  # type: ignore


@tvm_ffi.register_object("tirx.BufferStore")
class BufferStore(Stmt):
    """Buffer store node.

    Parameters
    ----------
    buffer : Buffer
        The buffer.

    value : Expr
        The value we to be stored.

    indices : List[Expr]
        The indices location to be stored.

    span : Optional[Span]
        The location of the stmt in the source code.
    """

    buffer: Buffer
    value: Expr
    indices: list[Expr]
    span: Span | None

    def __init__(
        self,
        buffer: Buffer,
        value: Expr,
        indices: list[Expr],
        span: Span | None = None,
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.BufferStore,
            buffer,
            value,
            indices,
            span,  # type: ignore
        )


@tvm_ffi.register_object("tirx.AttrStmt")
class AttrStmt(Stmt):
    """AttrStmt node.

    Parameters
    ----------
    node : Any
        The node to annotate the attribute

    attr_key : str
        Attribute type key.

    value : Expr
        The value of the attribute

    body : Stmt
        The body statement.

    span : Optional[Span]
        The location of the stmt in the source code.
    """

    node: Any
    attr_key: str
    value: Expr
    body: Stmt
    span: Span | None

    def __init__(
        self, node: Any, attr_key: str, value: Expr, body: Stmt, span: Span | None = None
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.AttrStmt,
            node,
            attr_key,
            value,
            body,
            span,  # type: ignore
        )


@tvm_ffi.register_object("tirx.SeqStmt")
class SeqStmt(Stmt):
    """Sequence of statements.

    Parameters
    ----------
    seq : List[Stmt]
        The statements

    span : Optional[Span]
        The location of the stmt in the source code.
    """

    seq: list[Stmt]
    span: Span | None

    def __init__(self, seq: list[Stmt], span: Span | None = None) -> None:
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

    then_case : Stmt
        The statement to execute if condition is true.

    else_case : Optional[Stmt]
        The statement to execute if condition is false.

    span : Optional[Span]
        The location of the stmt in the source code.
    """

    condition: Expr
    then_case: Stmt
    else_case: Stmt | None

    def __init__(
        self, condition: Expr, then_case: Stmt, else_case: Stmt | None, span: Span | None = None
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


def BufferRegion(buffer: Buffer, region: list[Range]) -> TensorRegion:
    """Construct a buffer-backed tensor region with TIRX subscript semantics.

    Parameters
    ----------
    buffer : Buffer
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
    stmt : Stmt
        The combined statement.
    """
    ret = []
    for value in args:
        if not isinstance(value, Stmt):
            value = Evaluate(value)
        ret.append(value)
    if len(ret) == 1:
        return ret[0]
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
        res = []
        for x in stmt:
            res += stmt_list(x)
        return res
    return [stmt]


# Source-compatibility re-export: TilePrimitiveCall lives in tile_primitive.py
# after the tile-primitive module merge. Imported last to avoid a cycle with
# tile_primitive's own ``from .stmt import Stmt``.
from .tile_primitive import TilePrimitiveCall  # noqa: E402,F401  isort: skip
