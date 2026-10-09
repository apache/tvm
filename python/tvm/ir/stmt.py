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
"""Shared statement nodes and sequence helpers."""

from collections.abc import Mapping, Sequence
from enum import IntEnum
from typing import Any

import tvm_ffi

from tvm.ir import DictAttrs, Expr, Op, Scriptable, Span, StringImm, Var, make_node
from tvm.runtime import Object

from . import _ffi_api


@tvm_ffi.register_object("ir.Stmt")
class Stmt(Object, Scriptable):
    """Base class of all the statements."""


@tvm_ffi.register_object("ir.Bind")
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


@tvm_ffi.register_object("ir.AssertStmt")
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


@tvm_ffi.register_object("ir.For")
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
        Additional loop annotations interpreted by the consuming dialect.

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


@tvm_ffi.register_object("ir.While")
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


@tvm_ffi.register_object("ir.TensorStore")
class TensorStore(Stmt):
    """Store a primitive value into a tensor destination.

    Parameters
    ----------
    dest : Expr
        The destination expression with a supported concrete store type.

    indices : list[Expr]
        The indices location to be stored.

    value : Expr
        The primitive value to be stored.

    span : Optional[Span]
        The location of the stmt in the source code.
    """

    dest: Expr
    indices: list[Expr]
    value: Expr
    span: Span | None

    def __init__(
        self,
        dest: Expr,
        indices: list[Expr],
        value: Expr,
        span: Span | None = None,
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.TensorStore,
            dest,
            indices,
            value,
            span,
        )


@tvm_ffi.register_object("ir.RegionStmt")
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


@tvm_ffi.register_object("ir.SeqStmt")
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


@tvm_ffi.register_object("ir.If")
class If(Stmt):
    """If node.

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
            _ffi_api.If,
            condition,
            then_case,
            else_case,
            span,  # type: ignore
        )


@tvm_ffi.register_object("ir.Evaluate")
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


@tvm_ffi.register_object("ir.Break")
class Break(Stmt):
    """Break node.

    Parameters
    ----------
    """

    def __init__(self, span: Span | None = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.Break, span)  # type: ignore


@tvm_ffi.register_object("ir.Return")
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


@tvm_ffi.register_object("ir.Continue")
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
