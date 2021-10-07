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
    a = tvm.tir.Var("array", "handle")
    st = tvm.tir.stmt.Store(a, x + 1, 1)
    assert isinstance(st, tvm.tir.stmt.Store)
    assert(st.buffer_var == a)
"""
from enum import IntEnum
from typing import List, Mapping, Optional, Union

import tvm._ffi
from tvm.ir import PrimExpr, Range, Span
from tvm.runtime import Object, const

from . import _ffi_api
from .buffer import Buffer
from .expr import IterVar


class Stmt(Object):
    """Base class of all the statements."""


@tvm._ffi.register_object("tir.LetStmt")
class LetStmt(Stmt):
    """LetStmt node.

    Parameters
    ----------
    var : Var
        The variable in the binding.

    value : PrimExpr
        The value in to be binded.

    body : Stmt
        The body statement.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, var, value, body, span=None):
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
        The location of this itervar in the source code.
    """

    def __init__(self, condition, message, body, span=None):
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
    THREAD_BINDING = 4


@tvm._ffi.register_object("tir.For")
class For(Stmt):
    """For node.

    Parameters
    ----------
    loop_var : Var
        The loop variable.

    min_val : PrimExpr
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

    annotations: tvm.ir.Map
        Additional annotation hints.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(
        self,
        loop_var,
        min_val,
        extent,
        kind,
        body,
        thread_binding=None,
        annotations=None,
        span=None,
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.For,  # type: ignore
            loop_var,
            min_val,
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
        The location of this itervar in the source code.
    """

    def __init__(self, condition, body, span=None):
        self.__init_handle_by_constructor__(
            _ffi_api.While,  # type: ignore
            condition,
            body,
            span,
        )


@tvm._ffi.register_object("tir.Store")
class Store(Stmt):
    """Store node.

    Parameters
    ----------
    buffer_var : Var
        The buffer Variable.

    value : PrimExpr
        The value we want to store.

    index : PrimExpr
        The index in the store expression.

    predicate : PrimExpr
        The store predicate.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, buffer_var, value, index, predicate=None, span=None):
        if predicate is None:
            predicate = _ffi_api.const_true(value.dtype, span)  # type: ignore
        self.__init_handle_by_constructor__(
            _ffi_api.Store, buffer_var, value, index, predicate, span  # type: ignore
        )


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

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, buffer, value, indices, span=None):
        self.__init_handle_by_constructor__(
            _ffi_api.BufferStore, buffer, value, indices, span  # type: ignore
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
        The location of this itervar in the source code.
    """

    def __init__(self, buffer, bounds, condition, body, span=None):
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
        The location of this itervar in the source code.
    """

    def __init__(self, producer, value, indices, span=None):
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
        The location of this itervar in the source code.
    """

    def __init__(self, buffer_var, dtype, extents, condition, body, annotations=None, span=None):
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


@tvm._ffi.register_object("tir.AttrStmt")
class AttrStmt(Stmt):
    """AttrStmt node.

    Parameters
    ----------
    node : Node
        The node to annotate the attribute

    attr_key : str
        Attribute type key.

    value : PrimExpr
        The value of the attribute

    body : Stmt
        The body statement.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, node, attr_key, value, body, span=None):
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

    bounds : list of range
        The bound of realize

    condition : PrimExpr
        The realize condition.

    body : Stmt
        The realize body

    storage_scope : str
        The storage scope associated with this realization

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, producer, bounds, condition, body, storage_scope="", span=None):
        self.__init_handle_by_constructor__(
            _ffi_api.ProducerRealize,
            producer,
            bounds,
            condition,
            body,
            storage_scope,
            span,  # type: ignore
        )


@tvm._ffi.register_object("tir.SeqStmt")
class SeqStmt(Stmt):
    """Sequence of statements.

    Parameters
    ----------
    seq : List[Stmt]
        The statements

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, seq, span=None):
        self.__init_handle_by_constructor__(_ffi_api.SeqStmt, seq, span)  # type: ignore

    def __getitem__(self, i):
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

    else_case : Stmt
        The statement to execute if condition is false.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, condition, then_case, else_case, span=None):
        self.__init_handle_by_constructor__(
            _ffi_api.IfThenElse, condition, then_case, else_case, span  # type: ignore
        )


@tvm._ffi.register_object("tir.Evaluate")
class Evaluate(Stmt):
    """Evaluate node.

    Parameters
    ----------
    value : PrimExpr
        The expression to be evalued.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, value, span=None):
        self.__init_handle_by_constructor__(_ffi_api.Evaluate, value, span)  # type: ignore


@tvm._ffi.register_object("tir.Prefetch")
class Prefetch(Stmt):
    """Prefetch node.

    Parameters
    ----------
    buffer : Buffer
        The buffer to be prefetched.

    bounds : list of Range
        The bounds to be prefetched.

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, buffer, bounds, span=None):
        self.__init_handle_by_constructor__(_ffi_api.Prefetch, buffer, bounds, span)  # type: ignore


@tvm._ffi.register_object("tir.BufferRegion")
class BufferRegion(Object):
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

    def __init__(self, buffer: Buffer, region: List[Range]):
        self.__init_handle_by_constructor__(_ffi_api.BufferRegion, buffer, region)  # type: ignore


@tvm._ffi.register_object("tir.MatchBufferRegion")
class MatchBufferRegion(Object):
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

    def __init__(self, buffer: Buffer, source: BufferRegion):
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
    alloc_buffers: Optional[List[Buffer]]
    match_buffers: Optional[List[MatchBufferRegion]]
    annotations: Optional[Mapping[str, Object]]
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
    ):
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
    ):
        if isinstance(predicate, bool):
            predicate = const(predicate, "bool")
        self.__init_handle_by_constructor__(
            _ffi_api.BlockRealize,  # type: ignore
            iter_values,
            predicate,
            block,
            span,
        )  # type: ignore


def stmt_seq(*args):
    """Make sequence of statements

    Parameters
    ----------
    args : list of Expr or Var
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


def stmt_list(stmt):
    """Make list of stmt from blocks.

    Parameters
    ----------
    stmt : A block statement

    Returns
    -------
    stmt_list : list of Stmt
         The unpacked list of statements
    """
    if isinstance(stmt, SeqStmt):
        res = []
        for x in stmt:
            res += stmt_list(x)
        return res
    return [stmt]
