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

    x = tvm.tir.Var("n", "int32")
    buffer = tvm.tir.decl_buffer((16,), "float32")
    st = tvm.tir.stmt.BufferStore(buffer, 1, (x,))
    assert isinstance(st, tvm.tir.stmt.BufferStore)
    assert(st.buffer == buffer)
"""
from enum import IntEnum
from typing import List, Mapping, Optional, Union

import tvm._ffi
from tvm.ir import PrimExpr, Range, Span
from tvm.runtime import Object, Scriptable, const, NDArray

from . import _ffi_api
from .buffer import Buffer, DataProducer
from .expr import Var, IterVar


class Stmt(Object, Scriptable):
    """Base class of all the statements."""


@tvm._ffi.register_object("tir.LetStmt")
class LetStmt(Stmt):
    """LetStmt node.

    Parameters
    ----------
    var : Var
        The variable in the binding.

    value : PrimExpr
        The value in to be bound.

    body : Stmt
        The body statement.

    span : Optional[Span]
        The location of the stmt in the source code.
    """

    var: Var
    value: PrimExpr
    body: Stmt
    span: Optional[Span]

    def __init__(self, var: Var, value: PrimExpr, body: Stmt, span: Optional[Span] = None) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.LetStmt, var, value, body, span  # type: ignore
        )


@tvm._ffi.register_object("tir.AssertStmt")
class AssertStmt(Stmt):
    """AssertStmt node.

    Parameters
    ----------
    condition : PrimExpr
        The assert condition.

    message : PrimExpr
        The error message.

    body : tvm.tir.Stmt
        The body statement.

    span : Optional[Span]
        The location of the stmt in the source code.
    """

    condition: PrimExpr
    message: PrimExpr
    body: Stmt
    span: Optional[Span]

    def __init__(
        self, condition: PrimExpr, message: PrimExpr, body: Stmt, span: Optional[Span] = None
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.AssertStmt, condition, message, body, span  # type: ignore
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


@tvm._ffi.register_object("tir.For")
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

    thread_binding: Optional[tir.IterVar]
        The thread this loop binds to. Only valid
        if kind is ThreadBinding

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
    thread_binding: Optional[IterVar]
    annotations: Mapping[str, Object]
    span: Optional[Span]

    def __init__(
        self,
        loop_var: Var,
        min: PrimExpr,  # pylint: disable=redefined-builtin
        extent: PrimExpr,
        kind: ForKind,
        body: Stmt,
        thread_binding: Optional[IterVar] = None,
        annotations: Optional[Mapping[str, Object]] = None,
        span: Optional[Span] = None,
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
            span,
        )


@tvm._ffi.register_object("tir.While")
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
    span: Optional[Span]

    def __init__(self, condition: PrimExpr, body: Stmt, span: Optional[Span] = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.While, condition, body, span)  # type: ignore


@tvm._ffi.register_object("tir.BufferStore")
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
    indices: List[PrimExpr]
    predicate: Optional[PrimExpr]
    span: Optional[Span]

    def __init__(
        self,
        buffer: Buffer,
        value: PrimExpr,
        indices: List[PrimExpr],
        predicate: Optional[PrimExpr] = None,
        span: Optional[Span] = None,
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.BufferStore, buffer, value, indices, predicate, span  # type: ignore
        )


@tvm._ffi.register_object("tir.BufferRealize")
class BufferRealize(Stmt):
    """Buffer realize node.

    Parameters
    ----------
    buffer : Buffer
        The buffer.

    bounds : List[Range]
        The value we to be stored.

    condition : PrimExpr
        The realize condition.

    body : Stmt
        The body of the statement.

    span : Optional[Span]
        The location of the stmt in the source code.
    """

    buffer: Buffer
    bounds: List[Range]
    condition: PrimExpr
    body: Stmt
    span: Optional[Span]

    def __init__(
        self,
        buffer: Buffer,
        bounds: List[Range],
        condition: PrimExpr,
        body: Stmt,
        span: Optional[Span] = None,
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.BufferRealize, buffer, bounds, condition, body, span  # type: ignore
        )


@tvm._ffi.register_object("tir.ProducerStore")
class ProducerStore(Stmt):
    """ProducerStore node.

    Parameters
    ----------
    producer : DataProducer
        The data producer.

    value : PrimExpr
        The value to be stored.

    indices : list of Expr
        The index arguments of the store.

    span : Optional[Span]
        The location of the stmt in the source code.
    """

    producer: DataProducer
    value: PrimExpr
    indices: List[PrimExpr]
    span: Optional[Span]

    def __init__(
        self,
        producer: DataProducer,
        value: PrimExpr,
        indices: List[PrimExpr],
        span: Optional[Span] = None,
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.ProducerStore, producer, value, indices, span  # type: ignore
        )


@tvm._ffi.register_object("tir.Allocate")
class Allocate(Stmt):
    """Allocate node.

    Parameters
    ----------
    buffer_var : Var
        The buffer variable.

    dtype : str
        The data type of the buffer.

    extents : list of Expr
        The extents of the allocate

    condition : PrimExpr
        The condition.

    body : Stmt
        The body statement.

    annotations: Optional[Mapping[str, Object]]
        Additional annotation hints

    span : Optional[Span]
        The location of the stmt in the source code.
    """

    buffer_var: Var
    dtype: str
    extents: List[PrimExpr]
    condition: PrimExpr
    body: Stmt
    annotations: Mapping[str, Object]
    span: Optional[Span]

    def __init__(
        self,
        buffer_var: Var,
        dtype: str,
        extents: List[PrimExpr],
        condition: PrimExpr,
        body: Stmt,
        annotations: Optional[Mapping[str, Object]] = None,
        span: Optional[Span] = None,
    ) -> None:
        if annotations is None:
            annotations = dict()
        self.__init_handle_by_constructor__(
            _ffi_api.Allocate,  # type: ignore
            buffer_var,
            dtype,
            extents,
            condition,
            body,
            annotations,
            span,
        )


@tvm._ffi.register_object("tir.AllocateConst")
class AllocateConst(Stmt):
    """Allocate constant node.

    Parameters
    ----------
    buffer_var : Var
        The buffer variable.

    dtype : str
        The data type of the buffer.

    extents : list of Expr
        The extents of the allocate

    data_or_idx : Union[NDArray, int]
        If an NDArray, this is the const data associated with the
        constant.  If an integer, this is the index into the
        "constants" attribute of the `IRModule` that contains the
        `AllocateConst`.

    body : Stmt
        The body statement.

    annotations : Optional[Mapping[str, Object]]
        Additional annotations about the allocation.

    span : Optional[Span]
        The location of the stmt in the source code.
    """

    buffer_var: Var
    dtype: str
    extents: List[PrimExpr]
    data: Optional[NDArray]
    irmod_storage_idx: Optional[int]
    body: Stmt
    annotations: Mapping[str, Object]
    span: Optional[Span]

    def __init__(
        self,
        buffer_var: Var,
        dtype: str,
        extents: List[PrimExpr],
        data_or_idx: Union[NDArray, int],
        body: Stmt,
        annotations: Optional[Mapping[str, Object]] = None,
        span: Optional[Span] = None,
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.AllocateConst,  # type: ignore
            buffer_var,
            dtype,
            extents,
            data_or_idx,
            body,
            annotations,
            span,
        )


@tvm._ffi.register_object("tir.DeclBuffer")
class DeclBuffer(Stmt):
    """DeclBuffer node.

    Parameters
    ----------
    buffer: Buffer
        The buffer being declared.

    body: Stmt
        The body statement to be executed.

    span: Optional[Span]
        The location of this DeclBuffer in the source code.
    """

    buffer: Buffer
    body: Stmt
    span: Optional[Span]

    def __init__(self, buffer: Buffer, body: Stmt, span: Optional[Span] = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.DeclBuffer, buffer, body, span)


@tvm._ffi.register_object("tir.AttrStmt")
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
    span: Optional[Span]

    def __init__(
        self, node: Object, attr_key: str, value: PrimExpr, body: Stmt, span: Optional[Span] = None
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.AttrStmt, node, attr_key, value, body, span  # type: ignore
        )


@tvm._ffi.register_object("tir.ProducerRealize")
class ProducerRealize(Stmt):
    """ProducerRealize node.

    Parameters
    ----------
    producer : DataProducer
        The data producer.

    bounds : List[Range]
        The bound of realize

    condition : PrimExpr
        The realize condition.

    body : Stmt
        The realize body

    storage_scope : str
        The storage scope associated with this realization

    span : Optional[Span]
        The location of the stmt in the source code.
    """

    producer: DataProducer
    bounds: List[Range]
    condition: PrimExpr
    body: Stmt
    storage_scope: str
    span: Optional[Span]

    def __init__(
        self,
        producer: DataProducer,
        bounds: List[Range],
        condition: PrimExpr,
        body: Stmt,
        storage_scope: str = "",
        span: Optional[Span] = None,
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.ProducerRealize,  # type: ignore
            producer,
            bounds,
            condition,
            body,
            storage_scope,
            span,
        )


@tvm._ffi.register_object("tir.SeqStmt")
class SeqStmt(Stmt):
    """Sequence of statements.

    Parameters
    ----------
    seq : List[Stmt]
        The statements

    span : Optional[Span]
        The location of the stmt in the source code.
    """

    seq: List[Stmt]
    span: Optional[Span]

    def __init__(self, seq: List[Stmt], span: Optional[Span] = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.SeqStmt, seq, span)  # type: ignore

    def __getitem__(self, i: int):
        return self.seq[i]

    def __len__(self):
        return len(self.seq)


@tvm._ffi.register_object("tir.IfThenElse")
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
    else_case: Optional[Stmt]

    def __init__(
        self,
        condition: PrimExpr,
        then_case: Stmt,
        else_case: Optional[Stmt],
        span: Optional[Span] = None,
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.IfThenElse, condition, then_case, else_case, span  # type: ignore
        )


@tvm._ffi.register_object("tir.Evaluate")
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
    span: Optional[Span]

    def __init__(self, value: PrimExpr, span: Optional[Span] = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.Evaluate, value, span)  # type: ignore


@tvm._ffi.register_object("tir.Prefetch")
class Prefetch(Stmt):
    """Prefetch node.

    Parameters
    ----------
    buffer : Buffer
        The buffer to be prefetched.

    bounds : List[Range]
        The bounds to be prefetched.

    span : Optional[Span]
        The location of the stmt in the source code.
    """

    buffer: Buffer
    bounds: List[Range]
    span: Optional[Span]

    def __init__(self, buffer: Buffer, bounds: List[Range], span: Optional[Span] = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.Prefetch, buffer, bounds, span)  # type: ignore


@tvm._ffi.register_object("tir.BufferRegion")
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
    region: List[Range]

    def __init__(self, buffer: Buffer, region: List[Range]) -> None:
        self.__init_handle_by_constructor__(_ffi_api.BufferRegion, buffer, region)  # type: ignore


@tvm._ffi.register_object("tir.MatchBufferRegion")
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
            _ffi_api.MatchBufferRegion, buffer, source  # type: ignore
        )


@tvm._ffi.register_object("tir.Block")
class Block(Stmt):
    """Block node.

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

    iter_vars: List[IterVar]
    reads: List[BufferRegion]
    writes: List[BufferRegion]
    name_hint: str
    body: Stmt
    init: Optional[Stmt]
    alloc_buffers: List[Buffer]
    match_buffers: List[MatchBufferRegion]
    annotations: Mapping[str, Object]
    span: Optional[Span]

    def __init__(
        self,
        iter_vars: List[IterVar],
        reads: List[BufferRegion],
        writes: List[BufferRegion],
        name_hint: str,
        body: Stmt,
        init: Optional[Stmt] = None,
        alloc_buffers: Optional[List[Buffer]] = None,
        match_buffers: Optional[List[MatchBufferRegion]] = None,
        annotations: Optional[Mapping[str, Object]] = None,
        span: Optional[Span] = None,
    ) -> None:
        if alloc_buffers is None:
            alloc_buffers = []
        if match_buffers is None:
            match_buffers = []
        if annotations is None:
            annotations = {}
        self.__init_handle_by_constructor__(
            _ffi_api.Block,  # type: ignore
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


@tvm._ffi.register_object("tir.BlockRealize")
class BlockRealize(Stmt):
    """BlockRealize node.

    Parameters
    ----------
    iter_values : List[PrimExpr]
        The binding values of the block var.

    predicate : Union[PrimExpr, bool]
        The predicate of the block.

    block : Block
        The block to realize

    span : Optional[Span]
        The location of this block_realize in the source code.
    """

    iter_values: List[PrimExpr]
    predicate: PrimExpr
    block: Block
    span: Optional[Span]

    def __init__(
        self,
        iter_values: List[PrimExpr],
        predicate: Union[PrimExpr, bool],
        block: Block,
        span: Optional[Span] = None,
    ) -> None:
        if isinstance(predicate, bool):
            predicate = const(predicate, "bool")
        self.__init_handle_by_constructor__(
            _ffi_api.BlockRealize,  # type: ignore
            iter_values,
            predicate,
            block,
            span,
        )  # type: ignore


def stmt_seq(*args: Union[PrimExpr, Stmt]) -> SeqStmt:
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


def stmt_list(stmt: Stmt) -> List[Stmt]:
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
