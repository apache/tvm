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
"""TVM Script Parser Scope Handler Classes"""
# pylint: disable=redefined-builtin, unused-argument, invalid-name, relative-beyond-top-level
from typing import Tuple, Any, Callable, Optional, List, Union, Mapping

import synr
import numpy as np
import tvm.tir
from tvm.runtime import Object, String, convert
from tvm.ir import Span, Range
from tvm.tir import Stmt, PrimExpr, IterVar, Var, Buffer, BufferRegion, ForKind

from .node import BufferSlice

from ..context_maintainer import ContextMaintainer
from ..registry import register
from ..utils import (
    get_param_list,
    tvm_span_from_synr,
    call_with_error_reporting,
)


class ScopeHandler:
    """Base class for all scope handlers"""

    def __init__(self, func: Callable):
        self.func: Callable = func
        self.body: Optional[Stmt] = None
        self.node: Optional[synr.ast.Node] = None
        self.context: Optional[ContextMaintainer] = None

    def signature(self) -> Tuple[str, Tuple[list, list, Any]]:
        return "tir." + self.func.__name__, get_param_list(self.func)

    def enter_scope(
        self,
        node: synr.ast.Node,
        context: ContextMaintainer,
        arg_list: List[Any],
        span: synr.ast.Span,
    ):
        pass

    def exit_scope(
        self,
        node: synr.ast.Node,
        context: ContextMaintainer,
        arg_list: List[Any],
        span: synr.ast.Span,
    ):
        self.node = node
        self.context = context
        return call_with_error_reporting(
            context.report_error, span, self.func, *arg_list, span=tvm_span_from_synr(span)
        )


class WithScopeHandler(ScopeHandler):
    """Base class for all with scope handlers"""

    def __init__(self, func, concise_scope, def_symbol):
        super().__init__(func)
        self.concise_scope = concise_scope
        self.def_symbol = def_symbol

    @staticmethod
    def get_optional_vars(node, context):
        """Get a list synr.ast.With's optional_vars"""
        assert isinstance(
            node, synr.ast.With
        ), f"WithScopeHandler expected synr.ast.With but got {type(node)}"

        if isinstance(node.lhs, list):
            for var in node.lhs:
                if not isinstance(var, synr.ast.Var):
                    context.report_error(
                        f"Invalid optional var definition, expected Var but got {type(var)}",
                        node.span,
                    )
            vars = node.lhs
        else:
            context.report_error(
                f"Invalid optional var definition, expected list of Var but got {type(node.lhs)}",
                node.span,
            )
        return vars


@register
class Allocate(WithScopeHandler):
    """With scope handler T.allocate(extents, dtype, scope, condition, annotations)"""

    def __init__(self):
        def allocate(extents, dtype, scope, condition=True, annotations=None, span=None):
            condition = tvm.runtime.convert(condition)
            scope = tvm.runtime.convert(scope)

            # Currently, allocate nodes should only occur after buffer
            # flattening has been applied.  This can be simplified in
            # the future by having the AllocateNode hold a buffer
            # object directly.
            flattened = self.buffer.get_flattened_buffer()

            return tvm.tir.Allocate(
                self.buffer.data,
                flattened.dtype,
                flattened.shape,
                condition,
                self.body,
                annotations=annotations,
                span=span,
            )

        super().__init__(allocate, concise_scope=True, def_symbol=True)
        self.buffer = None

    def enter_scope(
        self,
        node: synr.ast.Node,
        context: ContextMaintainer,
        arg_list: List[Any],
        span: synr.ast.Span,
    ):
        # define buffer vars in symbol table
        if isinstance(node, synr.ast.With):
            vars = WithScopeHandler.get_optional_vars(node, context)
            if len(vars) != 1:
                context.report_error(f"Unexpected number of vars: 1 vs. {len(vars)}", node.span)
            name = vars[0].id.name
            var_span = vars[0].id.span
        elif isinstance(node, synr.ast.Assign):
            if len(node.lhs) != 1:
                context.report_error(f"Unexpected number of vars: 1 vs. {len(node.lhs)}", node.span)
            name = node.lhs[0].id.name
            var_span = node.lhs[0].id.span
        else:
            raise Exception("Internal Bug")

        def setup_buffer(
            extents, dtype, scope, condition=True, annotations=None, span: Span = None
        ):
            """Setup buffer object for a given type."""
            self.buffer = tvm.tir.decl_buffer(
                shape=extents,
                dtype=dtype,
                name=name,
                scope=scope,
                span=span,
            )

        setup_buffer(*arg_list, span=tvm_span_from_synr(var_span))
        context.update_symbol(name, self.buffer, node)


@register
class AllocateConst(WithScopeHandler):
    """With scope handler T.allocate_const(data, extents, dtype, condition)

    TIR constant node to represent non-scalar constant
    """

    def __init__(self):
        def allocate_const(raw_data, dtype, shape, span=None):
            list_data = []
            for i in raw_data:
                list_data.append(i.value)
            nd_data = tvm.nd.array(np.asarray(list_data, dtype=dtype))
            n = tvm.tir.AllocateConst(self.buffer.data, dtype, shape, nd_data, self.body, span=span)
            return n

        super().__init__(allocate_const, concise_scope=True, def_symbol=True)
        self.buffer = None

    def enter_scope(
        self,
        node: synr.ast.Node,
        context: ContextMaintainer,
        arg_list: List[Any],
        span: synr.ast.Span,
    ):
        # define buffer vars in symbol table
        if isinstance(node, synr.ast.With):
            vars = WithScopeHandler.get_optional_vars(node, context)
            if len(vars) != 1:
                context.report_error(f"Unexpected number of vars: 1 vs. {len(vars)}", node.span)
            name = vars[0].id.name
            var_span = vars[0].id.span
        elif isinstance(node, synr.ast.Assign):
            if len(node.lhs) != 1:
                context.report_error(f"Unexpected number of vars: 1 vs. {len(node.lhs)}", node.span)
            name = node.lhs[0].id.name
            var_span = node.lhs[0].id.span
        else:
            raise Exception("Internal Bug")

        def setup_buffer(data, dtype, shape, span: Span = None):
            """Setup buffer var for a given type."""
            self.buffer = tvm.tir.decl_buffer(
                shape=shape,
                dtype=dtype,
                name=name,
                span=span,
            )

        setup_buffer(*arg_list, span=tvm_span_from_synr(var_span))
        context.update_symbol(name, self.buffer, node)


@register
class LaunchThread(WithScopeHandler):
    """With scope handler T.launch_thread(env_var, extent)"""

    def __init__(self):
        def launch_thread(env_var, extent, span):
            extent = tvm.runtime.convert(extent, span=span)
            thread_id = self.context.func_var_env_dict[env_var]
            attr_key = "virtual_thread" if thread_id == "vthread" else "thread_extent"
            return tvm.tir.AttrStmt(
                IterVar(
                    (0, extent),
                    env_var,
                    getattr(IterVar, "ThreadIndex"),
                    thread_id,
                    span=span,
                ),
                attr_key,
                extent,
                self.body,
                span=span,
            )

        super().__init__(launch_thread, concise_scope=True, def_symbol=False)


@register
class Realize(WithScopeHandler):
    """With scope handler T.realize(buffer_bounds, scope, condition)"""

    def __init__(self):
        def realize(
            buffer_slice: BufferSlice, scope: str, condition: bool = True, span: bool = None
        ):
            assert self.context, "call 'exit_scope' before 'enter_scope'"
            buffer: Buffer = buffer_slice.buffer
            bounds: List[Range] = []
            for s in buffer_slice.slices:
                min: Union[PrimExpr, int] = s.start
                extent: Union[PrimExpr, int] = 1 if s.stop is None else s.stop - s.start
                if isinstance(extent, PrimExpr):
                    extent = self.context.analyzer.simplify(extent)
                bounds.append(Range.from_min_extent(min, extent, span=s.span))

            scope = tvm.runtime.convert(scope, span=span)
            return tvm.tir.AttrStmt(
                buffer,
                "realize_scope",
                scope,
                tvm.tir.BufferRealize(buffer, bounds, condition, self.body, span=span),
                span=span,
            )

        super().__init__(realize, concise_scope=True, def_symbol=False)


@register
class Attr(WithScopeHandler):
    """With scope handler T.attr(attr_node, attr_key, value)"""

    def __init__(self):
        def attr(attr_node, attr_key, value, span):
            attr_node = tvm.runtime.convert(attr_node, span=span)
            value = tvm.runtime.convert(value, span=span)
            return tvm.tir.AttrStmt(attr_node, attr_key, value, self.body, span=span)

        super().__init__(attr, concise_scope=True, def_symbol=False)


@register
class AssertHandler(WithScopeHandler):
    """With scope handler T.Assert(condition, message)"""

    def __init__(self):
        def Assert(condition, message, span):
            return tvm.tir.AssertStmt(condition, tvm.runtime.convert(message), self.body, span=span)

        super().__init__(Assert, concise_scope=True, def_symbol=False)


@register
class Let(WithScopeHandler):
    """With scope handler T.let(var, value)"""

    def __init__(self):
        def let(var, value, span):
            return tvm.tir.LetStmt(var, value, self.body, span=span)

        super().__init__(let, concise_scope=False, def_symbol=False)


@register
class Block(WithScopeHandler):
    """With scope handler T.block(name)"""

    def __init__(self):
        def block(name_hint: str = "", span: Optional[Span] = None):
            assert (
                self.node and self.context and self.body
            ), "call 'exit_scope' before 'enter_scope'"
            block_info = self.context.block_info_stack[-1]

            # create block read/write regions
            reads: List[BufferRegion] = (
                [read.as_buffer_region() for read in block_info.reads] if block_info.reads else []
            )
            writes: List[BufferRegion] = (
                [write.as_buffer_region() for write in block_info.writes]
                if block_info.writes
                else []
            )

            region_detect_mask: int = (block_info.reads is None) | (
                (block_info.writes is None) << 1
            )
            annotations = {} if block_info.annotations is None else block_info.annotations
            if region_detect_mask != 0:
                annotations["tir.script_parsing_detect_access"] = region_detect_mask
            inner = tvm.tir.Block(
                block_info.iter_vars,
                reads,
                writes,
                name_hint,
                self.body,
                block_info.init,
                block_info.alloc_buffers,
                block_info.match_buffers,
                annotations,
                span,
            )
            assert len(block_info.iter_vars) == len(block_info.iter_values)
            predicate = (
                tvm.tir.const(True, "bool")
                if block_info.predicate is None
                else block_info.predicate
            )
            body = tvm.tir.BlockRealize(block_info.iter_values, predicate, inner, span)
            return body

        super().__init__(func=block, concise_scope=False, def_symbol=True)
        self.block_vars = None

    def enter_scope(
        self,
        node: synr.ast.Node,
        context: ContextMaintainer,
        arg_list: List[Any],
        span: synr.ast.Span,
    ):
        # define block vars
        assert isinstance(
            node, synr.ast.With
        ), f"BlockScopeHandler expected to work on synr.ast.With but got {type(node)}"

        optional_vars = [var.id.name for var in WithScopeHandler.get_optional_vars(node, context)]
        if optional_vars:
            context.report_error(
                f"Block expected no optional_vars (e.g., `x` in `with block() as x`), "
                f"but got {optional_vars}",
                node.span,
            )


@register
class InitBlock(WithScopeHandler):
    """With scope handler T.init()"""

    def __init__(self):
        def init(span: Span = None):
            assert self.context, "call 'exit_scope' before 'enter_scope'"
            if self.context.block_info_stack[-2].init is not None:
                self.context.report_error("Duplicate init block declaration", span)
            self.context.block_info_stack[-2].init = self.body

        super().__init__(func=init, concise_scope=False, def_symbol=True)


class LoopInfo:
    """Helper class for loop information"""

    loop_var: Var
    begin: PrimExpr
    extent: PrimExpr
    kind: ForKind
    thread_binding: Optional[str]
    annotations: Optional[Mapping[str, Object]]

    def __init__(
        self,
        begin: PrimExpr,
        extent: PrimExpr,
        kind: ForKind,
        thread_binding: Optional[str] = None,
        annotations: Optional[Mapping[str, Object]] = None,
    ) -> None:
        self.begin = begin
        self.extent = extent
        self.kind = kind
        self.thread_binding = thread_binding
        self.annotations = annotations


class ForScopeHandler(ScopeHandler):
    """Base class for all for scope handlers"""

    def __init__(self, func):
        super().__init__(func)
        self.loop_vars: List[Var] = []
        self.loop_info: List[LoopInfo] = []

    def enter_scope(
        self,
        node: synr.ast.Node,
        context: ContextMaintainer,
        arg_list: List[Any],
        span: synr.ast.Span,
    ):
        assert isinstance(
            node, synr.ast.For
        ), f"ForScopeHandler expected synr.ast.For but got {type(node)}"

        loop_var_names = list()
        spans = list()
        if isinstance(node.lhs, synr.ast.Var):
            loop_var_names.append(node.lhs.id.name)
            spans.append(tvm_span_from_synr(node.lhs.id.span))
        elif isinstance(node.lhs, list):
            for elt in node.lhs:
                if not isinstance(elt, synr.ast.Var):
                    context.report_error(
                        f"Invalid loop var. Expected a var, but got {type(elt)}", elt.span
                    )
                loop_var_names.append(elt.id.name)
                spans.append(tvm_span_from_synr(elt.id.span))
        else:
            context.report_error(
                f"Invalid loop var. Expected var or list of vars as lhs, but got {type(node.lhs)}",
                span,
            )

        self.node = node
        self.context = context
        # collect loop infos by calling self.func
        call_with_error_reporting(context.report_error, span, self.func, *arg_list)
        if len(loop_var_names) != len(self.loop_info):
            self.context.report_error(
                f"Inconsistent number of vars and loops, got {len(loop_var_names)} "
                + f"vs {len(self.loop_info)}",
                self.node.span,
            )
        # generate loop vars
        self.loop_vars = []
        for name, lv_span, li in zip(loop_var_names, spans, self.loop_info):
            if not li.begin.dtype.startswith("int"):
                raise NotImplementedError(f"Unsupported dtype in loop begin: {li.begin.dtype}")
            if not li.extent.dtype.startswith("int"):
                raise NotImplementedError(f"Unsupported dtype in loop extent: {li.extent.dtype}")
            dtype = "int64" if "int64" in [li.begin.dtype, li.extent.dtype] else "int32"
            self.loop_vars.append(tvm.te.var(name, dtype=dtype, span=lv_span))

        for loop_var, loop_info in zip(self.loop_vars, self.loop_info):
            context.update_symbol(loop_var.name, loop_var, node)
            context.loop_stack[loop_var] = Range.from_min_extent(loop_info.begin, loop_info.extent)

    def exit_scope(
        self,
        node: synr.ast.Node,
        context: ContextMaintainer,
        arg_list: List[Any],
        span: synr.ast.Span,
    ):
        assert self.loop_vars, "call 'exit_scope' before 'enter_scope'"
        for loop_var in self.loop_vars:
            context.loop_stack.pop(loop_var)
        # Use assert here since we have check it in `enter_scope`
        assert len(self.loop_vars) == len(self.loop_info)

        body = self.body
        for var, info in zip(reversed(self.loop_vars), reversed(self.loop_info)):
            body = tvm.tir.For(
                var,
                info.begin,
                info.extent,
                info.kind,
                body,
                info.thread_binding,
                info.annotations,
                span=tvm_span_from_synr(span),
            )

        return body

    def create_loop_info(
        self,
        begin: PrimExpr,
        end: PrimExpr,
        kind: ForKind,
        thread_binding: Optional[str] = None,
        annotations: Optional[Mapping[str, Object]] = None,
    ) -> None:
        """
        Helper function for creating For in TVM Script parser.

        Parameters
        ----------
        begin : PrimExpr
            The beginning value.

        end : PrimExpr
            The endding value.

        kind : ForKind
            The type of the for.

        thread_binding: Optional[str]
            The thread this loop binds to.

        annotations : Optional[Mapping[str, Object]]
            Additional annotation hints.

        span : Optional[Span]
            The location of this for in the source code.

        Returns
        -------
        for : For
            The constructed For.
        """
        begin, end = [convert(_) for _ in [begin, end]]
        assert self.context and self.node, "call 'exit_scope' before 'enter_scope'"
        extent = end if begin == 0 else self.context.analyzer.simplify(end - begin)
        self.annotations: Mapping[str, Object] = {}
        if annotations is not None:
            self.annotations = {
                key: String(val) if isinstance(val, str) else val
                for key, val in annotations.items()
            }

        self.loop_info.append(LoopInfo(begin, extent, kind, thread_binding, annotations))


@register
class Serial(ForScopeHandler):
    """For scope handler T.serial(begin, end, annotations)"""

    def __init__(self):
        def serial(
            begin: PrimExpr,
            end: PrimExpr = None,
            annotations: Optional[Mapping[str, Object]] = None,
        ):
            if end is None:
                end = begin
                begin = 0
            self.create_loop_info(begin, end, ForKind.SERIAL, annotations=annotations)

        super().__init__(serial)


@register
class Parallel(ForScopeHandler):
    """For scope handler T.parallel(begin, end, annotations)"""

    def __init__(self):
        def parallel(
            begin: PrimExpr,
            end: PrimExpr = None,
            annotations: Optional[Mapping[str, Object]] = None,
        ):
            if end is None:
                end = begin
                begin = 0
            self.create_loop_info(begin, end, ForKind.PARALLEL, annotations=annotations)

        super().__init__(parallel)


@register
class Vectorized(ForScopeHandler):
    """For scope handler T.vectorized(begin, end, annotations)"""

    def __init__(self):
        def vectorized(
            begin: PrimExpr,
            end: PrimExpr = None,
            annotations: Optional[Mapping[str, Object]] = None,
        ):
            if end is None:
                end = begin
                begin = 0
            self.create_loop_info(begin, end, ForKind.VECTORIZED, annotations=annotations)

        super().__init__(vectorized)


@register
class Unroll(ForScopeHandler):
    """For scope handler T.unroll(begin, end, annotations)"""

    def __init__(self):
        def unroll(
            begin: PrimExpr,
            end: PrimExpr = None,
            annotations: Optional[Mapping[str, Object]] = None,
        ):
            if end is None:
                end = begin
                begin = 0
            self.create_loop_info(begin, end, ForKind.UNROLLED, annotations=annotations)

        super().__init__(unroll)


@register
class ThreadBinding(ForScopeHandler):
    """For scope handler T.thread_binding(begin, end, thread, annotations)"""

    def __init__(self):
        def thread_binding(
            begin: PrimExpr,
            end: PrimExpr = None,
            thread: str = None,
            annotations: Optional[Mapping[str, Object]] = None,
        ):
            if thread is None:
                if isinstance(end, str):  # handle case like thread_binding(128, "threadIdx.x")
                    thread = end
                    end = None
                else:
                    raise ValueError("Thread cannot be None for thread_binding")
            if end is None:
                end = begin
                begin = 0
            thread_iter_var = IterVar(None, None, IterVar.ThreadIndex, thread)
            self.create_loop_info(
                begin,
                end,
                ForKind.THREAD_BINDING,
                thread_binding=thread_iter_var,
                annotations=annotations,
            )

        super().__init__(thread_binding)


@register
class RangeHandler(ForScopeHandler):
    """For scope handler range(begin, end, annotations)
    Note that tir.range is totally the same as T.serial
    """

    def __init__(self):
        def for_range(
            begin: PrimExpr,
            end: PrimExpr = None,
            annotations: Optional[Mapping[str, Object]] = None,
        ):
            if end is None:
                end = begin
                begin = 0
            self.create_loop_info(begin, end, ForKind.SERIAL, annotations=annotations)

        super().__init__(for_range)

    def signature(self):
        return "range", get_param_list(self.func)


@register
class Grid(ForScopeHandler):
    """For scope handler T.grid(extents)"""

    def __init__(self):
        def grid(*extents: List[PrimExpr]):
            for extent in extents:
                self.create_loop_info(0, extent, ForKind.SERIAL)

        super().__init__(grid)
