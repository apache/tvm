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
    buffer = tvm.tirx.decl_buffer((16,), "float32")
    st = tvm.tirx.stmt.BufferStore(buffer, 1, (x,))
    assert isinstance(st, tvm.tirx.stmt.BufferStore)
    assert(st.buffer == buffer)
"""

from collections.abc import Mapping
from enum import IntEnum

import tvm_ffi

from tvm.ir import PrimExpr, Range, Span
from tvm.runtime import Object, Scriptable, const
from tvm.tirx import IntImm

from . import _ffi_api
from .buffer import Buffer
from .exec_scope import ScopeIdDef
from .expr import IterVar, StringImm, Var


@tvm_ffi.register_object("tirx.Stmt")
class Stmt(Object, Scriptable):
    """Base class of all the statements."""


def _normalize_legacy_stmt(stmt: Stmt | None) -> Stmt | None:
    """Expand legacy body-carrying leaf stmt wrappers into SeqStmt form.

    Legacy python compatibility may attach a `body` attribute to leaf statements
    (Bind/DeclBuffer/AllocBuffer). This helper converts such wrappers to the new
    leaf + SeqStmt representation when embedding inside another statement node.
    """

    if stmt is None:
        return None

    prefix: list[Stmt] = []
    cur = stmt
    while True:
        if isinstance(cur, DeclBuffer) and hasattr(cur, "body"):
            prefix.append(DeclBuffer(cur.buffer, cur.span))
            cur = cur.body
            continue
        if isinstance(cur, AllocBuffer) and hasattr(cur, "body"):
            prefix.append(AllocBuffer(cur.buffer, cur.annotations, cur.span))
            cur = cur.body
            continue
        break

    if not prefix:
        return stmt

    normalized_tail = _normalize_legacy_stmt(cur)
    if normalized_tail is not None:
        prefix.append(normalized_tail)
    if len(prefix) == 1:
        return prefix[0]
    return SeqStmt(prefix)


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

    value : PrimExpr
        The value to be bound.

    span : Optional[Span]
        The location of the stmt in the source code.
    """

    var: Var
    value: PrimExpr
    span: Span | None

    def __init__(self, var: Var, value: PrimExpr, span: Span | None = None) -> None:
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

    condition : PrimExpr
        The assert condition.

    message_parts : list[StringImm]
        Error message fragments, concatenated at runtime when assertion fails.

    span : Span | None
        The location of the stmt in the source code.
    """

    kind: StringImm
    condition: PrimExpr
    message_parts: list
    span: Span | None

    def __init__(
        self,
        kind: StringImm,
        condition: PrimExpr,
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

    min : PrimExpr
        The beginning value.

    extent : PrimExpr
        The length of the loop.

    kind : ForKind
        The type of the for.

    body : Stmt
        The body statement.

    thread_binding: Optional[tirx.IterVar]
        The thread this loop binds to. Only valid
        if kind is ThreadBinding

    step : PrimExpr
        The loop step. Default to none which
        represent one.

    annotations: Optional[Mapping[str, Object]]
        Additional annotation hints.

    span : Optional[Span]
        The location of the stmt in the source code.
    """

    loop_var: Var
    min: PrimExpr
    extent: PrimExpr
    kind: ForKind
    body: Stmt
    thread_binding: IterVar | None
    annotations: Mapping[str, Object]
    step: PrimExpr | None
    span: Span | None

    def __init__(
        self,
        loop_var: Var,
        min: PrimExpr,  # pylint: disable=redefined-builtin
        extent: PrimExpr,
        kind: ForKind,
        body: Stmt,
        thread_binding: IterVar | None = None,
        annotations: Mapping[str, Object] | None = None,
        step: PrimExpr | None = None,
        span: Span | None = None,
    ) -> None:
        body = _normalize_legacy_stmt(body)
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
    condition : PrimExpr
        The termination condition.

    body : Stmt
        The body statement.

    span : Optional[Span]
        The location of the stmt in the source code.
    """

    condition: PrimExpr
    body: Stmt
    span: Span | None

    def __init__(self, condition: PrimExpr, body: Stmt, span: Span | None = None) -> None:
        body = _normalize_legacy_stmt(body)
        self.__init_handle_by_constructor__(_ffi_api.While, condition, body, span)  # type: ignore


@tvm_ffi.register_object("tirx.BufferStore")
class BufferStore(Stmt):
    """Buffer store node.

    Parameters
    ----------
    buffer : Buffer
        The buffer.

    value : PrimExpr
        The value we to be stored.

    indices : List[PrimExpr]
        The indices location to be stored.

    predicate : Optional[PrimExpr]
        A vector mask of boolean values indicating which lanes of a vector are to be
        stored. The number lanes of the mask must be equal to the number of lanes in
        value.

    span : Optional[Span]
        The location of the stmt in the source code.
    """

    buffer: Buffer
    value: PrimExpr
    indices: list[PrimExpr]
    predicate: PrimExpr | None
    span: Span | None

    def __init__(
        self,
        buffer: Buffer,
        value: PrimExpr,
        indices: list[PrimExpr],
        predicate: PrimExpr | None = None,
        span: Span | None = None,
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.BufferStore,
            buffer,
            value,
            indices,
            predicate,
            span,  # type: ignore
        )


@tvm_ffi.register_object("tirx.AllocBuffer")
class AllocBuffer(Stmt):
    """AllocBuffer node.

    Allocates a buffer and declares it in scope.

    Parameters
    ----------
    buffer: Buffer
        The buffer being allocated and declared.

    annotations: Optional[dict]
        Additional annotations about the allocation.

    span: Optional[Span]
        The location of this AllocBuffer in the source code.
    """

    buffer: Buffer
    span: Span | None

    def __init__(self, buffer: Buffer, *args, **kwargs) -> None:
        body: Stmt | None = None
        annotations: dict | None = None
        span: Span | None = None

        idx = 0
        argc = len(args)

        # Legacy form: AllocBuffer(buffer, body[, annotations][, span])
        if idx < argc and isinstance(args[idx], Stmt):
            body = args[idx]
            idx += 1

        if idx < argc:
            arg = args[idx]
            if isinstance(arg, Mapping):
                annotations = dict(arg)
                idx += 1
            elif arg is None:
                annotations = None
                idx += 1
            elif isinstance(arg, Span):
                span = arg
                idx += 1
            else:
                raise TypeError(
                    "AllocBuffer expects (buffer[, annotations][, span]) or "
                    "legacy (buffer, body[, annotations][, span])"
                )

        if idx < argc:
            arg = args[idx]
            if arg is None or isinstance(arg, Span):
                span = arg
                idx += 1
            else:
                raise TypeError("AllocBuffer span must be a Span or None")

        if idx != argc:
            raise TypeError(
                "AllocBuffer expects (buffer[, annotations][, span]) or "
                "legacy (buffer, body[, annotations][, span])"
            )

        if kwargs:
            invalid_keys = set(kwargs.keys()) - {"body", "annotations", "span"}
            if invalid_keys:
                raise TypeError(f"Unexpected keyword arguments for AllocBuffer: {invalid_keys}")
            if "body" in kwargs:
                kw_body = kwargs["body"]
                if kw_body is not None and not isinstance(kw_body, Stmt):
                    raise TypeError("AllocBuffer body must be a Stmt or None")
                if body is not None and kw_body is not None and body is not kw_body:
                    raise TypeError("AllocBuffer body specified by both args and kwargs")
                body = kw_body if kw_body is not None else body
            if "annotations" in kwargs:
                kw_ann = kwargs["annotations"]
                if kw_ann is not None and not isinstance(kw_ann, Mapping):
                    raise TypeError("AllocBuffer annotations must be Mapping or None")
                if annotations is not None and kw_ann is not None and annotations != dict(kw_ann):
                    raise TypeError("AllocBuffer annotations specified by both args and kwargs")
                annotations = dict(kw_ann) if kw_ann is not None else annotations
            if "span" in kwargs:
                kw_span = kwargs["span"]
                if kw_span is not None and not isinstance(kw_span, Span):
                    raise TypeError("AllocBuffer span must be a Span or None")
                if span is not None and kw_span is not None and span is not kw_span:
                    raise TypeError("AllocBuffer span specified by both args and kwargs")
                span = kw_span if kw_span is not None else span

        self.__init_handle_by_constructor__(_ffi_api.AllocBuffer, buffer, annotations, span)
        # Legacy compatibility. Body is carried on python side only.
        if body is not None:
            self.body = body


@tvm_ffi.register_object("tirx.DeclBuffer")
class DeclBuffer(Stmt):
    """DeclBuffer node.

    Parameters
    ----------
    buffer: Buffer
        The buffer being declared.

    span: Optional[Span]
        The location of this DeclBuffer in the source code.
    """

    buffer: Buffer
    span: Span | None

    def __init__(self, buffer: Buffer, *args, **kwargs) -> None:
        body: Stmt | None = None
        span: Span | None = None

        if len(args) == 1:
            arg0 = args[0]
            if isinstance(arg0, Stmt):
                body = arg0
            elif arg0 is None or isinstance(arg0, Span):
                span = arg0
            else:
                raise TypeError(
                    "DeclBuffer expects (buffer[, span]) or legacy (buffer, body[, span])"
                )
        elif len(args) == 2:
            body, span = args
            if body is not None and not isinstance(body, Stmt):
                raise TypeError("Legacy DeclBuffer body must be a Stmt or None")
            if span is not None and not isinstance(span, Span):
                raise TypeError("DeclBuffer span must be a Span or None")
        elif len(args) > 2:
            raise TypeError("DeclBuffer expects (buffer[, span]) or legacy (buffer, body[, span])")

        if kwargs:
            invalid_keys = set(kwargs.keys()) - {"body", "span"}
            if invalid_keys:
                raise TypeError(f"Unexpected keyword arguments for DeclBuffer: {invalid_keys}")
            if "body" in kwargs:
                kw_body = kwargs["body"]
                if kw_body is not None and not isinstance(kw_body, Stmt):
                    raise TypeError("DeclBuffer body must be a Stmt or None")
                if body is not None and kw_body is not None and body is not kw_body:
                    raise TypeError("DeclBuffer body specified by both args and kwargs")
                body = kw_body if kw_body is not None else body
            if "span" in kwargs:
                kw_span = kwargs["span"]
                if kw_span is not None and not isinstance(kw_span, Span):
                    raise TypeError("DeclBuffer span must be a Span or None")
                if span is not None and kw_span is not None and span is not kw_span:
                    raise TypeError("DeclBuffer span specified by both args and kwargs")
                span = kw_span if kw_span is not None else span

        self.__init_handle_by_constructor__(_ffi_api.DeclBuffer, buffer, span)
        # Legacy compatibility. Body is carried on python side only.
        if body is not None:
            self.body = body


@tvm_ffi.register_object("tirx.AttrStmt")
class AttrStmt(Stmt):
    """AttrStmt node.

    Parameters
    ----------
    node : Object
        The node to annotate the attribute

    attr_key : str
        Attribute type key.

    value : PrimExpr
        The value of the attribute

    body : Stmt
        The body statement.

    span : Optional[Span]
        The location of the stmt in the source code.
    """

    node: Object
    attr_key: str
    value: PrimExpr
    body: Stmt
    span: Span | None

    def __init__(
        self, node: Object, attr_key: str, value: PrimExpr, body: Stmt, span: Span | None = None
    ) -> None:
        body = _normalize_legacy_stmt(body)
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
        seq = [_normalize_legacy_stmt(s) for s in seq]
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
    condition : PrimExpr
        The expression

    then_case : Stmt
        The statement to execute if condition is true.

    else_case : Optional[Stmt]
        The statement to execute if condition is false.

    span : Optional[Span]
        The location of the stmt in the source code.
    """

    condition: PrimExpr
    then_case: Stmt
    else_case: Stmt | None

    def __init__(
        self, condition: PrimExpr, then_case: Stmt, else_case: Stmt | None, span: Span | None = None
    ) -> None:
        then_case = _normalize_legacy_stmt(then_case)
        else_case = _normalize_legacy_stmt(else_case)
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
    value : PrimExpr
        The expression to be evaluated.

    span : Optional[Span]
        The location of the stmt in the source code.
    """

    value: PrimExpr
    span: Span | None

    def __init__(self, value: PrimExpr, span: Span | None = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.Evaluate, value, span)  # type: ignore


@tvm_ffi.register_object("tirx.BufferRegion")
class BufferRegion(Object, Scriptable):
    """BufferRegion node.

    Parameters
    ----------
    buffer : Buffer
        The buffer of the buffer region

    region : List[Range]
        The region array of the buffer region
    """

    buffer: Buffer
    region: list[Range]

    def __init__(self, buffer: Buffer, region: list[Range]) -> None:
        self.__init_handle_by_constructor__(_ffi_api.BufferRegion, buffer, region)  # type: ignore

    def __getitem__(self, indices):
        from ..arith import Analyzer

        if not isinstance(indices, tuple | list):
            indices = [indices]

        has_step = any(
            isinstance(i, slice) and (i.step is not None and i.step != 1) for i in indices
        )
        if has_step:
            raise ValueError("BufferRegion slicing does not support steps")

        analyzer = Analyzer()
        new_region = []
        for i, index in enumerate(indices):
            old_range = self.region[i]
            if isinstance(index, slice):
                start = 0 if index.start is None else index.start
                stop = old_range.extent if index.stop is None else index.stop
                new_min = old_range.min + start
                new_extent = analyzer.simplify(stop - start)
                new_region.append(Range.from_min_extent(new_min, new_extent))
            else:
                new_min = old_range.min + index
                new_region.append(
                    Range.from_min_extent(
                        new_min, IntImm(index.ty, 1) if isinstance(index, PrimExpr) else 1
                    )
                )
        # Fill remaining dimensions with their original ranges
        for i in range(len(indices), len(self.region)):
            new_region.append(self.region[i])
        return BufferRegion(self.buffer, new_region)


@tvm_ffi.register_object("tirx.MatchBufferRegion")
class MatchBufferRegion(Object, Scriptable):
    """MatchBufferRegion node.

    Parameters
    ----------
    buffer : Buffer
        The target buffer

    source : BufferRegion
        The region of source buffer
    """

    buffer: Buffer
    source: BufferRegion

    def __init__(self, buffer: Buffer, source: BufferRegion) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.MatchBufferRegion,
            buffer,
            source,  # type: ignore
        )


@tvm_ffi.register_object("tirx.SBlock")
class SBlock(Stmt):
    """SBlock node.

    Parameters
    ----------
    iter_vars : List[IterVar]
        The block Variable.

    reads : List[BufferRegion]
        The read buffer regions of the block.

    writes: List[BufferRegion]
        The write buffer regions of the block.

    name_hint: str
        the name_hint of the block.

    body: Stmt
        The body of the block.

    init: Optional[Stmt]
        The init block of the reduction block

    alloc_buffers: Optional[list[Buffer]]
        The buffer allocations

    match_buffers: Optional[List[MatchBufferRegion]]
        The subregion buffer match

    annotations: Optional[Mapping[str, Object]]
        Additional annotation hints.

    span : Optional[Span]
        The location of this block in the source code.
    """

    iter_vars: list[IterVar]
    reads: list[BufferRegion]
    writes: list[BufferRegion]
    name_hint: str
    body: Stmt
    init: Stmt | None
    alloc_buffers: list[Buffer]
    match_buffers: list[MatchBufferRegion]
    annotations: Mapping[str, Object]
    span: Span | None

    def __init__(
        self,
        iter_vars: list[IterVar],
        reads: list[BufferRegion],
        writes: list[BufferRegion],
        name_hint: str,
        body: Stmt,
        init: Stmt | None = None,
        alloc_buffers: list[Buffer] | None = None,
        match_buffers: list[MatchBufferRegion] | None = None,
        annotations: Mapping[str, Object] | None = None,
        span: Span | None = None,
    ) -> None:
        if alloc_buffers is None:
            alloc_buffers = []
        if match_buffers is None:
            match_buffers = []
        if annotations is None:
            annotations = {}
        body = _normalize_legacy_stmt(body)
        init = _normalize_legacy_stmt(init)
        self.__init_handle_by_constructor__(
            _ffi_api.SBlock,  # type: ignore
            iter_vars,
            reads,
            writes,
            name_hint,
            body,
            init,
            alloc_buffers,
            match_buffers,
            annotations,
            span,
        )  # type: ignore


@tvm_ffi.register_object("tirx.SBlockRealize")
class SBlockRealize(Stmt):
    """SBlockRealize node.

    Parameters
    ----------
    iter_values : List[PrimExpr]
        The binding values of the block var.

    predicate : Union[PrimExpr, bool]
        The predicate of the block.

    block : SBlock
        The block to realize

    span : Optional[Span]
        The location of this block_realize in the source code.
    """

    iter_values: list[PrimExpr]
    predicate: PrimExpr
    block: SBlock
    span: Span | None

    def __init__(
        self,
        iter_values: list[PrimExpr],
        predicate: PrimExpr | bool,
        block: SBlock,
        span: Span | None = None,
    ) -> None:
        if isinstance(predicate, bool):
            predicate = const(predicate, "bool")
        self.__init_handle_by_constructor__(
            _ffi_api.SBlockRealize,  # type: ignore
            iter_values,
            predicate,
            block,
            span,
        )  # type: ignore


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


@tvm_ffi.register_object("tirx.Continue")
class Continue(Stmt):
    """Continue node.

    Parameters
    ----------
    """

    def __init__(self, span: Span | None = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.Continue, span)  # type: ignore


def stmt_seq(*args: PrimExpr | Stmt) -> SeqStmt:
    """Make sequence of statements

    Parameters
    ----------
    *args : Union[PrimExpr, Stmt]
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
