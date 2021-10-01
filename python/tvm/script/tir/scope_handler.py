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
import tvm.tir
from tvm.runtime import Object
from tvm.ir import Span, Range
from tvm.tir import Stmt, PrimExpr, IterVar, Var, Buffer, BufferRegion, ForKind

from .node import BufferSlice
from .utils import buffer_slice_to_region

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
    """With scope handler T.allocate(extents, dtype, scope, condition)"""

    def __init__(self):
        def allocate(extents, dtype, scope, condition=True, span=None):
            condition = tvm.runtime.convert(condition)
            scope = tvm.runtime.convert(scope)
            return tvm.tir.Allocate(
                self.buffer_var, dtype, extents, condition, self.body, span=span
            )

        super().__init__(allocate, concise_scope=True, def_symbol=True)
        self.buffer_var = None

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
                context.report_error("Unexpected number of vars", node.span)
            name = vars[0].id.name
            var_span = vars[0].id.span
        elif isinstance(node, synr.ast.Assign):
            name = node.lhs.id.name
            var_span = node.lhs.id.span
        else:
            raise Exception("Internal Bug")

        def setup_buffer_var(extents, dtype, scope, condition=True, span: Span = None):
            """Setup buffer var for a given type."""
            buffer_ptr_type = tvm.ir.PointerType(tvm.ir.PrimType(dtype), scope)
            self.buffer_var = tvm.tir.Var(name, buffer_ptr_type, span)

        setup_buffer_var(*arg_list, span=tvm_span_from_synr(var_span))
        context.update_symbol(name, self.buffer_var, node)


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
    """With scope handler T.block(extents, name) as iter_vars"""

    def __init__(self):
        def block(axes=None, name_hint: str = "", span: Optional[Span] = None):
            assert (
                self.node and self.context and self.body
            ), "call 'exit_scope' before 'enter_scope'"
            block_info = self.context.block_info_stack[-1]
            if axes is None:
                axes = []
            if len(axes) != len(self.block_vars):
                self.context.report_error(
                    "Inconsistent number of block vars, "
                    + f"there are {len(axes)} axes but {len(self.block_vars)} block vars. "
                    + "The number of block vars should match the number of axes.",
                    self.node.span,
                )
            block_iters: List[IterVar] = []
            for i, axis in enumerate(axes):
                axis = tvm.runtime.convert(axis)
                if isinstance(axis, tvm.tir.PrimExpr):
                    block_var_dom = Range.from_min_extent(0, axis)
                    block_iters.append(IterVar(block_var_dom, self.block_vars[i], 0))
                elif isinstance(axis, Range):
                    block_iters.append(IterVar(axis, self.block_vars[i], 0))
                elif isinstance(axis, IterVar):
                    block_iters.append(IterVar(axis.dom, self.block_vars[i], axis.iter_type))
                else:
                    self.context.report_error(
                        "Invalid argument of T.block(), "
                        + f"expected PrimExpr, Range or IterVar, but got {type(axis)}",
                        self.node.span,
                    )

            # create block read/write regions

            reads: List[BufferRegion] = (
                [buffer_slice_to_region(read) for read in block_info.reads]
                if block_info.reads
                else []
            )
            writes: List[BufferRegion] = (
                [buffer_slice_to_region(write) for write in block_info.writes]
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
                block_iters,
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
            # create block var iter binding
            values: List[PrimExpr]
            if not block_info.iter_bindings:
                values = self.context.loop_stack[-2].copy()
                if len(block_iters) == 0:
                    # It is an opaque block without any bindings
                    values = []
                elif len(values) == 0:
                    values = [tvm.tir.const(float("nan"), dtype="float32")] * len(block_iters)
                elif len(values) != len(block_iters):
                    self.context.report_error(
                        "Number of block iter var and outer loop nesting mismatch, "
                        + f"{len(block_iters)} block iter vars but {len(values)} loops",
                        self.node.span,
                    )
            else:
                for block_var in self.block_vars:
                    if block_var not in block_info.iter_bindings:
                        self.context.report_error(
                            "Missing block iter var binding for " + block_var.name,
                            self.node.span,
                        )
                values = [block_info.iter_bindings[block_var] for block_var in self.block_vars]
            predicate = (
                tvm.tir.const(True, "bool")
                if block_info.predicate is None
                else block_info.predicate
            )
            body = tvm.tir.BlockRealize(values, predicate, inner, span)
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

        vars = WithScopeHandler.get_optional_vars(node, context)
        self.block_vars = [tvm.te.var(var.id.name) for var in vars]
        for block_var in self.block_vars:
            context.update_symbol(block_var.name, block_var, node)


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


class ForScopeHandler(ScopeHandler):
    """Base class for all for scope handlers"""

    def __init__(self, func):
        super().__init__(func)
        self.loop_vars: Optional[List[Var]] = None

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

        self.loop_vars = [
            tvm.te.var(name, dtype="int32", span=span) for name, span in zip(loop_var_names, spans)
        ]
        for loop_var in self.loop_vars:
            context.update_symbol(loop_var.name, loop_var, node)
            context.loop_stack[-1].append(loop_var)

    def exit_scope(
        self,
        node: synr.ast.Node,
        context: ContextMaintainer,
        arg_list: List[Any],
        span: synr.ast.Span,
    ):
        assert self.loop_vars, "call 'exit_scope' before 'enter_scope'"
        for _ in self.loop_vars:
            context.loop_stack[-1].pop()
        return super().exit_scope(node, context, arg_list, span)

    def create_loop(
        self,
        begin: PrimExpr,
        end: PrimExpr,
        kind: ForKind,
        thread_binding: Optional[str] = None,
        annotations: Optional[Mapping[str, Object]] = None,
        span: Optional[Span] = None,
    ) -> tvm.tir.For:
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
        assert (
            self.loop_vars and self.context and self.node
        ), "call 'exit_scope' before 'enter_scope'"
        if len(self.loop_vars) != 1:
            self.context.report_error(
                f"Expected exactly one loop var, but got {self.loop_vars}", self.node.span
            )
        extent = end if begin == 0 else self.context.analyzer.simplify(end - begin)
        annos: Mapping[str, Object] = {}
        if annotations is not None:
            annos = {
                key: tvm.tir.StringImm(val) if isinstance(val, str) else val
                for key, val in annotations.items()
            }
        return tvm.tir.For(
            self.loop_vars[0],
            begin,
            extent,
            kind,
            self.body,
            thread_binding=thread_binding,
            annotations=annos,
            span=span,
        )


@register
class Serial(ForScopeHandler):
    """For scope handler T.serial(begin, end, annotations)"""

    def __init__(self):
        def serial(
            begin: PrimExpr,
            end: PrimExpr,
            annotations: Optional[Mapping[str, Object]] = None,
            span: Optional[Span] = None,
        ):
            return self.create_loop(begin, end, ForKind.SERIAL, annotations=annotations, span=span)

        super().__init__(serial)


@register
class Parallel(ForScopeHandler):
    """For scope handler T.parallel(begin, end, annotations)"""

    def __init__(self):
        def parallel(
            begin: PrimExpr,
            end: PrimExpr,
            annotations: Optional[Mapping[str, Object]] = None,
            span: Optional[Span] = None,
        ):
            return self.create_loop(
                begin, end, ForKind.PARALLEL, annotations=annotations, span=span
            )

        super().__init__(parallel)


@register
class Vectorized(ForScopeHandler):
    """For scope handler T.vectorized(begin, end, annotations)"""

    def __init__(self):
        def vectorized(
            begin: PrimExpr,
            end: PrimExpr,
            annotations: Optional[Mapping[str, Object]] = None,
            span: Optional[Span] = None,
        ):
            return self.create_loop(
                begin, end, ForKind.VECTORIZED, annotations=annotations, span=span
            )

        super().__init__(vectorized)


@register
class Unroll(ForScopeHandler):
    """For scope handler T.unroll(begin, end, annotations)"""

    def __init__(self):
        def unroll(
            begin: PrimExpr,
            end: PrimExpr,
            annotations: Optional[Mapping[str, Object]] = None,
            span: Optional[Span] = None,
        ):
            return self.create_loop(
                begin, end, ForKind.UNROLLED, annotations=annotations, span=span
            )

        super().__init__(unroll)


@register
class ThreadBinding(ForScopeHandler):
    """For scope handler T.thread_binding(begin, end, thread, annotations)"""

    def __init__(self):
        def thread_binding(
            begin: PrimExpr,
            end: PrimExpr,
            thread: str,
            annotations: Optional[Mapping[str, Object]] = None,
            span: Optional[Span] = None,
        ):
            thread_iter_var = IterVar(None, None, IterVar.ThreadIndex, thread, span=span)
            return self.create_loop(
                begin,
                end,
                ForKind.THREAD_BINDING,
                thread_binding=thread_iter_var,
                annotations=annotations,
                span=span,
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
            span: Optional[Span] = None,
        ):
            if end is None:
                end = begin
                begin = 0
            return self.create_loop(begin, end, ForKind.SERIAL, annotations=annotations, span=span)

        super().__init__(for_range)

    def signature(self):
        return "range", get_param_list(self.func)


@register
class Grid(ForScopeHandler):
    """For scope handler T.grid(extents)"""

    def __init__(self):
        def grid(*extents: List[PrimExpr], span: Span):
            assert (
                self.node and self.context and self.loop_vars
            ), "call 'exit_scope' before 'enter_scope'"
            if len(self.loop_vars) != len(extents):
                self.context.report_error(
                    "Inconsistent number of loop vars and extents, "
                    + f"got {len(self.loop_vars)} vs {len(extents)}",
                    self.node.span,
                )
            body = self.body
            for loop_var, extent in zip(reversed(self.loop_vars), reversed(extents)):
                body = tvm.tir.For(loop_var, 0, extent, ForKind.SERIAL, body, span=span)
            return body

        super().__init__(grid)
