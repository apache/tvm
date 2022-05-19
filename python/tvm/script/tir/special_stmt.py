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
"""TVM Script Parser Special Stmt Classes"""
# pylint: disable=unused-argument, no-self-argument, inconsistent-return-statements
# pylint: disable=relative-beyond-top-level
from typing import Callable, List, Optional, Tuple, Any, Mapping, Union

import synr
from synr import ast
from tvm.ir.expr import PrimExpr, Range

import tvm.tir
from tvm.runtime import Object, String
from tvm.target import Target
from tvm.ir import Span
from tvm.tir import IntImm, IterVar, Var

from .node import BufferSlice

from ..context_maintainer import BlockInfo, ContextMaintainer
from ..registry import register
from ..utils import (
    get_param_list,
    tvm_span_from_synr,
    call_with_error_reporting,
)


def convert_to_int(
    value: Union[IntImm, int],
    arg_name: str,
    report_error: Callable,
    span: Union[Span, synr.ast.Span],
) -> int:
    """convert a const int or TVM IntImm to Python int.
    Reports an error when input cannot be converted to int.

    Parameters
    ----------
    value : Union[tvm.tir.IntImm, int]
        The input value to be converted.
    arg_name : str
        Function argument name for error reporting.
    report_error: Callable
        The report error function handle
    span : Union[synr.ast.Span, tvm.ir.Span]
        Location of the error
    """
    if isinstance(value, IntImm):
        return value.value
    if isinstance(value, int):
        return value
    report_error(
        f"Expected int or IntImm for {arg_name}, but got {str(type(value))}",
        span,
    )


class SpecialStmt:
    """Base class for all Special Stmts"""

    def __init__(self, func: Callable, def_symbol: bool):
        self.func: Callable = func
        self.def_symbol: bool = def_symbol
        self.node: Optional[synr.ast.Node] = None
        self.context: Optional[ContextMaintainer] = None

    def signature(self) -> Tuple[str, Tuple[list, list, Any]]:
        return "tir." + self.func.__name__, get_param_list(self.func)

    def handle(
        self,
        node: ast.Node,
        context: ContextMaintainer,
        arg_list: List[Any],
        span: synr.ast.Span,
    ):
        self.node = node
        self.context = context
        return call_with_error_reporting(
            context.report_error, span, self.func, *arg_list, span=tvm_span_from_synr(span)
        )


@register
class MatchBuffer(SpecialStmt):
    """Special Stmt match_buffer(param, shape, dtype, data, strides, elem_offset, scope, align,
                                 offset_factor, buffer_type, axis_separators)

    Note
    ----
    This Special Stmt will perform different behavior depends on the type of param.
    If the param is a var in function parameter, it will create a buffer from DLTensor.
    Else if the param is a subregion of other buffers, then create a subregion match inside a block.

    Example
    -------
    Match buffer from function parameter
    .. code-block:: python
        A = T.match_buffer(a, (128, 128), dtype="float32")

    Match buffer from Buffer subregion
    .. code-block:: python
        A = T.match_buffer(B[0:128, i * 128 : i * 128 + 128], (128, 128), dtype="float32")
    """

    def __init__(self):
        def match_buffer(
            param,
            shape,
            dtype="float32",
            data=None,
            strides=None,
            elem_offset=None,
            scope="global",
            align=-1,
            offset_factor=0,
            buffer_type="default",
            axis_separators=None,
            span=None,
        ):
            if not isinstance(self.node, ast.Assign) or not len(self.node.lhs) == 1:
                self.context.report_error(
                    "`match_buffer` must be assigned to a single buffer, "
                    "e.g. A = match_buffer(...)",
                    self.node.span,
                )
            if strides is None:
                strides = []
            align = convert_to_int(align, "align", self.context.report_error, self.node.span)
            offset_factor = convert_to_int(
                offset_factor, "offset_factor", self.context.report_error, self.node.span
            )
            buffer_name: str = self.node.lhs[0].id.name
            buffer = tvm.tir.decl_buffer(
                shape,
                dtype,
                buffer_name,
                data,
                strides,
                elem_offset,
                scope,
                align,
                offset_factor,
                buffer_type,
                axis_separators,
                span=span,
            )
            if isinstance(param, tvm.tir.Var):
                if param not in self.context.func_params:
                    self.context.report_error(
                        "Can not bind non-input param to buffer", self.node.rhs.params[0].span
                    )
                self.context.func_buffer_map[param] = buffer
            elif isinstance(param, BufferSlice):
                buffer_region = param.as_buffer_region()
                self.context.current_block_scope().match_buffers.append(
                    tvm.tir.MatchBufferRegion(buffer, buffer_region)
                )
            else:
                self.context.report_error(
                    "The source of match_buffer expected Var or BufferSlice, but got "
                    + str(type(param)),
                    self.node.rhs.params[0].span,
                )
            self.context.update_symbol(buffer_name, buffer, self.node)

        super().__init__(match_buffer, def_symbol=True)


@register
class BufferDeclare(SpecialStmt):
    """Special Stmt buffer_decl(shape, dtype, data, strides, elem_offset, scope, align,
                                offset_factor, buffer_type, axis_separators)
    Example
    -------
    .. code-block:: python
        A = T.buffer_decl((128, 128), dtype="float32")
    """

    def __init__(self):
        def buffer_decl(
            shape,
            dtype="float32",
            data=None,
            strides=None,
            elem_offset=None,
            scope="global",
            align=-1,
            offset_factor=0,
            buffer_type="default",
            axis_separators=None,
            span=None,
        ):
            if not isinstance(self.node, ast.Assign) or not len(self.node.lhs) == 1:
                self.context.report_error(
                    "`buffer_decl` must be assigned to a single buffer, e.g. A = buffer_decl(...)",
                    self.node.span,
                )

            if strides is None:
                strides = []
            align = convert_to_int(align, "align", self.context.report_error, self.node.span)
            offset_factor = convert_to_int(
                offset_factor, "offset_factor", self.context.report_error, self.node.span
            )
            buffer_name: str = self.node.lhs[0].id.name
            buffer = tvm.tir.decl_buffer(
                shape,
                dtype,
                buffer_name,
                data,
                strides,
                elem_offset,
                scope,
                align,
                offset_factor,
                buffer_type,
                axis_separators,
                span=span,
            )
            self.context.update_symbol(buffer_name, buffer, self.node)
            return buffer

        super().__init__(buffer_decl, def_symbol=True)


@register
class AllocBuffer(SpecialStmt):
    """Special function alloc_buffer(shape, dtype, data, strides, elem_offset, scope, align,
                                     offset_factor, buffer_type, axis_separators)

    Example
    -------
    .. code-block:: python

        A = T.alloc_buffer((128, 128), dtype="float32")
    """

    def __init__(self):
        def alloc_buffer(
            shape,
            dtype="float32",
            data=None,
            strides=None,
            elem_offset=None,
            scope="global",
            align=-1,
            offset_factor=0,
            buffer_type="default",
            axis_separators=None,
            span=None,
        ):
            if not isinstance(self.node, ast.Assign) or not len(self.node.lhs) == 1:
                self.context.report_error(
                    "`alloc_buffer` must be assigned to a single buffer, "
                    "e.g. A = alloc_buffer(...)",
                    self.node.span,
                )

            if strides is None:
                strides = []
            align = convert_to_int(align, "align", self.context.report_error, self.node.span)
            offset_factor = convert_to_int(
                offset_factor, "offset_factor", self.context.report_error, self.node.span
            )
            buffer_name: str = self.node.lhs[0].id.name
            buffer = tvm.tir.decl_buffer(
                shape,
                dtype,
                buffer_name,
                data,
                strides,
                elem_offset,
                scope,
                align,
                offset_factor,
                buffer_type,
                axis_separators,
                span=span,
            )
            if self.context.current_block_scope():
                self.context.current_block_scope().alloc_buffers.append(buffer)
            else:
                # If it is allocated outside all blocks, allocate it under root block.
                self.context.root_alloc_buffers.append(buffer)
            self.context.update_symbol(buffer_name, buffer, self.node)

        super().__init__(alloc_buffer, def_symbol=True)


@register
class BlockReads(SpecialStmt):
    """Special function reads([read_regions], *other_regions)

    Note
    ----
    *other_region is an unpackable list of BufferSlice to support
    reads syntax sugar like reads(BufferRegion1, BufferRegion2, ...)

    Example
    -------
    .. code-block:: python

        T.reads([A[vi: vi + 4, vk: vk + 4], B[vk: vk + 4, vj]])
    """

    def __init__(self):
        def reads(
            *read_regions: Union[BufferSlice, List[BufferSlice]],
            span: Span = None,
        ):
            assert self.context, "call 'exit_scope' before 'enter_scope'"
            block_scope = self.context.current_block_scope()
            if block_scope is None:
                self.context.report_error(
                    "Expected to declare read regions inside a block.",
                    span,
                )
            if block_scope.reads is not None:
                self.context.report_error(
                    "Duplicate write region declaration, "
                    + "previous one is "
                    + str(", ".join(str(x) for x in block_scope.reads)),
                    span,
                )
            if len(read_regions) > 1:
                for read_region in read_regions:
                    if not isinstance(read_region, BufferSlice):
                        self.context.report_error(
                            "Incorrect input type. Expected *BufferSlice or List[BufferSlice],"
                            + f" but got {type(read_regions)}",
                            span,
                        )
            elif len(read_regions) == 1:
                if isinstance(read_regions[0], list):
                    read_regions = read_regions[0]

            block_scope.reads = read_regions

        super().__init__(reads, def_symbol=False)


@register
class BlockWrites(SpecialStmt):
    """Special function writes([write_regions], *other_regions)

    Note
    ----
    *other_region is an unpackable list of BufferSlice to support
    writes syntax sugar like writes(BufferRegion1, BufferRegion2, ...)

    Example
    -------
    .. code-block:: python

        T.writes([C[vi: vi + 4, vj])
    """

    def __init__(self):
        def writes(
            *write_regions: Union[BufferSlice, List[BufferSlice]],
            span: Span = None,
        ):
            assert self.context, "call 'exit_scope' before 'enter_scope'"
            block_scope = self.context.current_block_scope()
            if block_scope is None:
                self.context.report_error(
                    "Expected to declare write regions inside a block.",
                    span,
                )
            if block_scope.writes is not None:
                self.context.report_error(
                    "Duplicate write region declaration, "
                    + "previous one is "
                    + str(", ".join(str(x) for x in block_scope.writes)),
                    span,
                )
            if len(write_regions) > 1:
                for write_region in write_regions:
                    if not isinstance(write_region, BufferSlice):
                        self.context.report_error(
                            "Incorrect input type. Expected *BufferSlice or List[BufferSlice],"
                            + f" but got {type(write_regions)}",
                            span,
                        )
            elif len(write_regions) == 1:
                if isinstance(write_regions[0], list):
                    write_regions = write_regions[0]
            block_scope.writes = write_regions

        super().__init__(writes, def_symbol=False)


@register
class BlockAttr(SpecialStmt):
    """Special function block_attr({attr_key: attr_value})

    Example
    -------
    .. code-block:: python

        T.block_attr({"double_buffer_scope": 1})
    """

    def __init__(self):
        def block_attr(attrs: Mapping[str, Object], span: Span = None):
            assert self.context, "call 'exit_scope' before 'enter_scope'"
            block_scope = self.context.current_block_scope()
            if block_scope is None:
                self.context.report_error(
                    "Expected to declare block annotations inside a block.",
                    span,
                )
            if block_scope.annotations is not None:
                self.context.report_error(
                    "Duplicate block annotations declaration, "
                    + "previous one is "
                    + str(block_scope.annotations),
                    span,
                )
            attrs = {
                key: String(val) if isinstance(val, str) else val for key, val in attrs.items()
            }
            block_scope.annotations = attrs

        super().__init__(block_attr, def_symbol=False)


class BlockAxis(SpecialStmt):
    """Special stmt for defining a spatial block axis
    axis.S(dom, iter_value)

    Example
    -------
    .. code-block:: python

        vi = T.axis.S(128, i * 4 + j)
    """

    def axis(
        self,
        var_name: str,
        dom: Union[PrimExpr, Range],
        value: PrimExpr,
        iter_type: int,
        span: Optional[Span] = None,
    ) -> None:
        """
        Helper function for creating block axis

        Parameters
        ----------
        var_name : str
            The name_hint of var

        dom : Union[PrimExpr, Range]
            The iter domain.

        value : PrimExpr
            The binding value

        iter_type : int
            The iteration type.

        span : Optional[Span]
            The location of this for in the source code.
        """
        assert self.context, "call 'exit_scope' before 'enter_scope'"
        block_scope: BlockInfo = self.context.current_block_scope()
        if block_scope is None:
            self.context.report_error(
                "Expected to declare block axes inside a block.",
                self.node.span,
            )
        if var_name in [iter_var.var.name for iter_var in block_scope.iter_vars]:
            self.context.report_error("Duplicate block axis " + var_name, self.node.span)

        dom = tvm.runtime.convert(dom)
        if isinstance(dom, PrimExpr):
            dom = tvm.ir.Range(dom)
        elif isinstance(dom, tvm.ir.container.Array) and len(dom) == 2:
            dom = tvm.ir.Range(dom[0], dom[1])
        elif not isinstance(dom, tvm.ir.Range):
            self.context.report_error(
                f"Block axis domain expected PrimExpr or Range, but got {type(dom)}",
                self.node.span,
            )
        block_var = tvm.tir.Var(var_name, dtype=dom.extent.dtype)
        value = tvm.runtime.convert(value)
        if not isinstance(value, PrimExpr):
            self.context.report_error(
                f"Block axis value expected PrimExpr, but got {type(value)}",
                self.node.span,
            )
        iter_var = tvm.tir.IterVar(dom, block_var, iter_type)
        block_scope.iter_vars.append(iter_var)
        block_scope.iter_values.append(value)
        self.context.update_symbol(var_name, block_var, self.node)


@register
class BlockAxisSpatial(BlockAxis):
    """Special stmt for defining a spatial block axis
    axis.spatial(dom, iter_value)

    Example
    -------
    .. code-block:: python

        vi = T.axis.spatial(128, k)
    """

    def __init__(self):
        def axis_spatial(
            dom: Union[PrimExpr, Tuple[PrimExpr, PrimExpr]], value: PrimExpr, span: Span = None
        ):
            if not isinstance(self.node, ast.Assign) or not len(self.node.lhs) == 1:
                self.context.report_error(
                    "`axis.spatial` must be assigned to a var, e.g. vi = axis.spatial(...)",
                    self.node.span,
                )
            self.axis(self.node.lhs[0].id.name, dom, value, IterVar.DataPar)

        super().__init__(axis_spatial, def_symbol=True)

    def signature(self) -> Tuple[str, Tuple[list, list, Any]]:
        return "tir.axis.spatial", get_param_list(self.func)


@register
class BlockAxisS(BlockAxis):
    """The sugar special stmt for defining a spatial block axis
    axis.S(dom, iter_value)

    Example
    -------
    .. code-block:: python

        vi = T.axis.S(128, k)
    """

    def __init__(self):
        def axis_spatial(
            dom: Union[PrimExpr, Tuple[PrimExpr, PrimExpr]], value: PrimExpr, span: Span = None
        ):
            if not isinstance(self.node, ast.Assign) or not len(self.node.lhs) == 1:
                self.context.report_error(
                    "`axis.S` must be assigned to a var, e.g. vi = axis.S(...)",
                    self.node.span,
                )
            self.axis(self.node.lhs[0].id.name, dom, value, IterVar.DataPar)

        super().__init__(axis_spatial, def_symbol=True)

    def signature(self) -> Tuple[str, Tuple[list, list, Any]]:
        return "tir.axis.S", get_param_list(self.func)


@register
class BlockAxisReduce(BlockAxis):
    """Special stmt for defining a reduce block axis
    axis.reduce(dom, iter_value)

    Example
    -------
    .. code-block:: python

        vi = T.axis.reduce(128, k)
    """

    def __init__(self):
        def axis_reduce(
            dom: Union[PrimExpr, Tuple[PrimExpr, PrimExpr]], value: PrimExpr, span: Span = None
        ):
            if not isinstance(self.node, ast.Assign) or not len(self.node.lhs) == 1:
                self.context.report_error(
                    "`axis.reduce` must be assigned` to a var, e.g. vi = axis.reduce(...)",
                    self.node.span,
                )
            self.axis(self.node.lhs[0].id.name, dom, value, IterVar.CommReduce)

        super().__init__(axis_reduce, def_symbol=True)

    def signature(self) -> Tuple[str, Tuple[list, list, Any]]:
        return "tir.axis.reduce", get_param_list(self.func)


@register
class BlockAxisR(BlockAxis):
    """The sugar special stmt for defining a reduce block axis
    axis.R(dom, iter_value)

    Example
    -------
    .. code-block:: python

        vi = T.axis.R(128, k)
    """

    def __init__(self):
        def axis_reduce(
            dom: Union[PrimExpr, Tuple[PrimExpr, PrimExpr]], value: PrimExpr, span: Span = None
        ):
            if not isinstance(self.node, ast.Assign) or not len(self.node.lhs) == 1:
                self.context.report_error(
                    "`axis.R` must be assigned to a var, e.g. vi = axis.R(...)",
                    self.node.span,
                )
            self.axis(self.node.lhs[0].id.name, dom, value, IterVar.CommReduce)

        super().__init__(axis_reduce, def_symbol=True)

    def signature(self) -> Tuple[str, Tuple[list, list, Any]]:
        return "tir.axis.R", get_param_list(self.func)


@register
class BlockAxisScan(BlockAxis):
    """Special stmt for defining a ordered block axis
    axis.scan(dom, iter_value)

    Example
    -------
    .. code-block:: python

        vi = T.axis.scan(128, k)
    """

    def __init__(self):
        def axis_scan(
            dom: Union[PrimExpr, Tuple[PrimExpr, PrimExpr]], value: PrimExpr, span: Span = None
        ):
            if not isinstance(self.node, ast.Assign) or not len(self.node.lhs) == 1:
                self.context.report_error(
                    "`axis.scan` must be assigned to a var, e.g. vi = axis.scan(...)",
                    self.node.span,
                )
            self.axis(self.node.lhs[0].id.name, dom, value, IterVar.Ordered)

        super().__init__(axis_scan, def_symbol=True)

    def signature(self) -> Tuple[str, Tuple[list, list, Any]]:
        return "tir.axis.scan", get_param_list(self.func)


@register
class BlockAxisOpaque(BlockAxis):
    """Special stmt for defining a opaque block axis
    axis.opaque(dom, iter_value)

    Example
    -------
    .. code-block:: python

        vi = T.axis.opaque(128, k)
    """

    def __init__(self):
        def axis_opaque(
            dom: Union[PrimExpr, Tuple[PrimExpr, PrimExpr]], value: PrimExpr, span: Span = None
        ):
            if not isinstance(self.node, ast.Assign) or not len(self.node.lhs) == 1:
                self.context.report_error(
                    "`axis.opaque` must be assigned to a var, e.g. vi = axis.opaque(...)",
                    self.node.span,
                )
            self.axis(self.node.lhs[0].id.name, dom, value, IterVar.DimInfo)

        super().__init__(axis_opaque, def_symbol=True)

    def signature(self) -> Tuple[str, Tuple[list, list, Any]]:
        return "tir.axis.opaque", get_param_list(self.func)


@register
class BlockAxisRemap(BlockAxis):
    """Special stmt for remapping loops vars to block axes.
    axis.remap(iter_type, iter_value)

    Note
    ----
    Iter_type is a string consisting of 'S' and 'R', where 'S' means
    for spatial and 'R' means for reduce.

    Example
    -------
    .. code-block:: python

        vi, vj = T.axis.remap("SS", [i, j])
    """

    def __init__(self):
        def axis_remap(iter_types: str, loop_vars: List[tvm.tir.expr.Var], span: Span = None):
            if not isinstance(self.node, ast.Assign) or not len(self.node.lhs) >= 1:
                self.context.report_error(
                    "`axis.remap` must be assigned to one or more vars, "
                    "e.g. vi, vj = axis.remap(...)",
                    self.node.span,
                )
            var_num: int = len(self.node.lhs)
            if var_num != len(iter_types):
                self.context.report_error(
                    f"`iter_type` expected {var_num} charactor(s), "
                    f"but got {len(iter_types)}: {iter_types}",
                    span,
                )
            if var_num != len(loop_vars):
                self.context.report_error(
                    f"`iter_type` expected {var_num} loop var(s), "
                    f"but got {len(loop_vars)}: {loop_vars}",
                    span,
                )
            for var, iter_ty, loop_var in zip(self.node.lhs, iter_types, loop_vars):
                iter_type: int
                if iter_ty == "S":
                    iter_type = IterVar.DataPar
                elif iter_ty == "R":
                    iter_type = IterVar.CommReduce
                else:
                    self.context.report_error(
                        f'`iter_type` only expected "S" (for spatial) or "R" (for reduce), '
                        f'but got "{iter_ty}"',
                        span,
                    )

                if not isinstance(loop_var, tvm.tir.expr.Var):
                    self.context.report_error(
                        f"Values of `axis.remap` expected single loop var, but got {loop_var}",
                        loop_var.span,
                    )
                loops = self.context.loop_stack
                if loop_var not in loops:
                    self.context.report_error(
                        f"Cannot find loop var {loop_var} in loop nesting.",
                        span,
                    )
                self.axis(var.id.name, loops[loop_var], loop_var, iter_type)

        super().__init__(axis_remap, def_symbol=True)

    def signature(self) -> Tuple[str, Tuple[list, list, Any]]:
        return "tir.axis.remap", get_param_list(self.func)


@register
class BlockPredicate(SpecialStmt):
    """Special function where(predicate)

    Example
    -------
    .. code-block:: python

        T.where(i < 4)
    """

    def __init__(self):
        def where(predicate, span=None):
            assert self.context, "call 'exit_scope' before 'enter_scope'"
            block_scope = self.context.current_block_scope()
            if block_scope is None:
                self.context.report_error(
                    "Expected to declare the predicate inside a block.",
                    span,
                )
            if block_scope.predicate is not None:
                self.context.report_error(
                    "Duplicate block predicate declaration, "
                    + "previous one is "
                    + str(block_scope.predicate),
                    span,
                )

            block_scope.predicate = predicate

        super().__init__(where, def_symbol=False)


@register
class VarDef(SpecialStmt):
    """Special function for defining a Var"""

    def __init__(self):
        def var(dtype, span):
            assert isinstance(
                self.node, ast.Assign
            ), f"VarDef expected ast.Assign but got {type(self.node)}"
            names = [x.id.name for x in self.node.lhs]
            if len(names) != 1:
                self.context.report_error(
                    f"VarDef expected assign to only one var, but got {names}", span
                )
            v = Var(names[0], dtype, span=span)
            self.context.update_symbol(v.name, v, self.node)

        super().__init__(var, def_symbol=True)


@register
class BufferVarDef(SpecialStmt):
    """Special function for defining a variable of pointer type"""

    def __init__(self):
        def buffer_var(dtype, storage_scope, span):
            assert isinstance(
                self.node, ast.Assign
            ), f"BufferVarDef expected ast.Assign but got {type(self.node)}"
            names = [x.id.name for x in self.node.lhs]
            if len(names) != 1:
                self.context.report_error(
                    f"VarDef expected assign to only one var, but got {names}", span
                )
            ptr_type = tvm.ir.PointerType(tvm.ir.PrimType(dtype), storage_scope)
            v = Var(names[0], ptr_type, span=span)
            self.context.update_symbol(v.name, v, self.node)

        super().__init__(buffer_var, def_symbol=True)


@register
class EnvThread(SpecialStmt):
    """Bind a var to thread env"""

    def __init__(self):
        def env_thread(env_name, span):
            assert isinstance(
                self.node, ast.Assign
            ), f"EnvThread expected ast.Assign but got {type(self.node)}"
            names = [x.id.name for x in self.node.lhs]
            if len(names) != 1:
                self.context.report_error(
                    f"VarDef expected assign to only one var, but got {names}", span
                )
            v = Var(names[0], dtype="int32", span=span)
            self.context.func_var_env_dict[v] = env_name
            self.context.update_symbol(v.name, v, self.node)

        super().__init__(env_thread, def_symbol=True)


@register
class FuncAttr(SpecialStmt):
    """Special Stmt for declaring the DictAttr of PrimFunc
    Example
    -------
    .. code-block:: python
         T.func_attr({"tir.noalias": True, "global_symbol"})
    """

    def __init__(self):
        def func_attr(dict_attr, span):
            self.context.func_dict_attr = dict_attr

        super().__init__(func_attr, def_symbol=False)


@register
class PreflattenedBufferMap(SpecialStmt):
    """Special Stmt for declaring the PrimFunc::preflattened_buffer_map

    Example
    -------
    .. code-block:: python
         A0 = T.match_buffer(A, (48,), dtype="float32")
         T.preflattened_buffer_map(A, (1, 4, 4, 3), elem_offset=1, align=4, dtype="float32")
    """

    def __init__(self):
        def preflattened_buffer(
            postflattened,
            shape,
            dtype="float32",
            data=None,
            strides=None,
            elem_offset=None,
            scope="global",
            align=-1,
            offset_factor=0,
            buffer_type="default",
            span=None,
        ):

            param = None
            for key, value in self.context.func_buffer_map.items():
                if value.same_as(postflattened):
                    param = key
                    break

            assert (
                param is not None
            ), f"Post-flatten buffer {postflattened.name} does not appear in the buffer map."

            if data is None:
                data = self.context.func_buffer_map[param].data

            buffer_name: str = f"{postflattened.name}_preflatten"
            if align != -1:
                if isinstance(align, IntImm):
                    align = align.value
                else:
                    assert isinstance(align, int), f"align: want int or IntImm, got {align!r}"

            if offset_factor != 0:
                if isinstance(offset_factor, IntImm):
                    offset_factor = offset_factor.value
                else:
                    assert isinstance(
                        offset_factor, int
                    ), f"offset_factor: want int or IntImm, got {offset_factor!r}"

            preflattened = tvm.tir.decl_buffer(
                shape,
                dtype,
                buffer_name,
                data,
                strides,
                elem_offset,
                scope,
                align,
                offset_factor,
                buffer_type,
                span=span,
            )

            self.context.func_preflattened_buffer_map[param] = preflattened

        super().__init__(preflattened_buffer, def_symbol=False)


@register
class TargetAttrValue(SpecialStmt):
    """Special Stmt for target attr value.
    Example
    -------
    .. code-block:: python
        T.target("llvm")
    """

    def __init__(self):
        def target(*args, span):
            self.context.report_error(f"T.target should not appear as a stmt", span)

        super().__init__(target, def_symbol=False)

    def __call__(self, target_config):
        if not isinstance(target_config, (str, dict)):
            raise ValueError(
                f"T.target expected a config dict or string, but got {type(target_config)}"
            )
        return Target(target_config)
