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
"""Schedulable TensorIR statement nodes."""

from collections.abc import Mapping

import tvm_ffi

from tvm.ir import Expr, Span, TensorRegion
from tvm.runtime import Object, Scriptable, const
from tvm.tirx.buffer import Buffer
from tvm.tirx.expr import IterVar
from tvm.tirx.stmt import Stmt, _normalize_legacy_stmt

from . import _ffi_api


@tvm_ffi.register_object("s_tir.MatchBufferRegion")
class MatchBufferRegion(Object, Scriptable):
    """MatchBufferRegion node.

    Parameters
    ----------
    buffer : Buffer
        The target buffer

    source : TensorRegion
        The region of source buffer
    """

    buffer: Buffer
    source: TensorRegion

    def __init__(self, buffer: Buffer, source: TensorRegion) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.MatchBufferRegion,
            buffer,
            source,  # type: ignore
        )


@tvm_ffi.register_object("s_tir.SBlock")
class SBlock(Stmt):
    """SBlock node.

    Parameters
    ----------
    iter_vars : List[IterVar]
        The block Variable.

    reads : List[TensorRegion]
        The read buffer regions of the block.

    writes: List[TensorRegion]
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
    reads: list[TensorRegion]
    writes: list[TensorRegion]
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
        reads: list[TensorRegion],
        writes: list[TensorRegion],
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


@tvm_ffi.register_object("s_tir.SBlockRealize")
class SBlockRealize(Stmt):
    """SBlockRealize node.

    Parameters
    ----------
    iter_values : List[Expr]
        The binding values of the block var.

    predicate : Union[Expr, bool]
        The predicate of the block.

    block : SBlock
        The block to realize

    span : Optional[Span]
        The location of this block_realize in the source code.
    """

    iter_values: list[Expr]
    predicate: Expr
    block: SBlock
    span: Span | None

    def __init__(
        self,
        iter_values: list[Expr],
        predicate: Expr | bool,
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
