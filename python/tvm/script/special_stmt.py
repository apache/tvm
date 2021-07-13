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

import tvm.tir
from tvm.runtime import Object
from tvm import te
from tvm.ir import Span
from tvm.tir import IntImm
from .utils import (
    get_param_list,
    tvm_span_from_synr,
    buffer_slice_to_region,
    call_with_error_reporting,
)
from .registry import register
from .context_maintainer import ContextMaintainer
from .node import BufferSlice


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
    """Special Stmt match_buffer(var, shape, dtype, data, strides, elem_offset, scope, align,
                                 offset_factor, buffer_type)
    Example
    -------
    .. code-block:: python
        A = tir.match_buffer(a, (128, 128), dtype="float32")
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
            span=None,
        ):
            if not isinstance(self.node, ast.Assign):
                self.context.report_error(
                    "match_buffer must be assigned to a buffer, e.g. A = match_buffer(...)",
                    self.node.span,
                )
            if param not in self.context.func_params:
                self.context.report_error(
                    "Can not bind non-input param to buffer", self.node.rhs.params[0].span
                )
            if strides is None:
                strides = []
            align = convert_to_int(align, "align", self.context.report_error, self.node.span)
            offset_factor = convert_to_int(
                offset_factor, "offset_factor", self.context.report_error, self.node.span
            )
            buffer = tvm.tir.decl_buffer(
                shape,
                dtype,
                self.node.lhs.id.name,
                data,
                strides,
                elem_offset,
                scope,
                align,
                offset_factor,
                buffer_type,
                span=span,
            )
            self.context.func_buffer_map[param] = buffer
            self.context.update_symbol(self.node.lhs.id.name, buffer, self.node)

        super().__init__(match_buffer, def_symbol=True)


@register
class BufferDeclare(SpecialStmt):
    """Special Stmt buffer_decl(shape, dtype, data, strides, elem_offset, scope, align,
                                offset_factor, buffer_type)
    Example
    -------
    .. code-block:: python
        A = tir.buffer_decl((128, 128), dtype="float32")
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
            span=None,
        ):
            if not isinstance(self.node, ast.Assign):
                self.context.report_error(
                    "buffer_decl must be assigned to a buffer, e.g. A = buffer_decl(...)",
                    self.node.span,
                )

            if strides is None:
                strides = []
            align = convert_to_int(align, "align", self.context.report_error, self.node.span)
            offset_factor = convert_to_int(
                offset_factor, "offset_factor", self.context.report_error, self.node.span
            )
            buffer = tvm.tir.decl_buffer(
                shape,
                dtype,
                self.node.lhs.id.name,
                data,
                strides,
                elem_offset,
                scope,
                align,
                offset_factor,
                buffer_type,
                span=span,
            )
            self.context.update_symbol(self.node.lhs.id.name, buffer, self.node)
            return buffer

        super().__init__(buffer_decl, def_symbol=True)


@register
class AllocBuffer(SpecialStmt):
    """Special function alloc_buffer(shape, dtype, data, strides, elem_offset, scope, align,
                                     offset_factor, buffer_type)

    Example
    -------
    .. code-block:: python

        A = tir.alloc_buffer((128, 128), dtype="float32")
    """

    def __init__(self):
        def alloc_buffer(
            shape,
            dtype="float32",
            data=None,
            strides=None,
            elem_offset=None,
            scope="",
            align=-1,
            offset_factor=0,
            buffer_type="default",
            span=None,
        ):
            if not isinstance(self.node, ast.Assign):
                self.context.report_error(
                    "alloc_buffer must be assigned to a buffer, e.g. A = alloc_buffer(...)",
                    self.node.span,
                )

            if strides is None:
                strides = []
            align = convert_to_int(align, "align", self.context.report_error, self.node.span)
            offset_factor = convert_to_int(
                offset_factor, "offset_factor", self.context.report_error, self.node.span
            )
            buffer = tvm.tir.decl_buffer(
                shape,
                dtype,
                self.node.lhs.id.name,
                data,
                strides,
                elem_offset,
                scope,
                align,
                offset_factor,
                buffer_type,
                span=span,
            )
            self.context.current_block_scope().alloc_buffers.append(buffer)
            self.context.update_symbol(self.node.lhs.id.name, buffer, self.node)

        super().__init__(alloc_buffer, def_symbol=True)


@register
class BlockVarBind(SpecialStmt):
    """Special function bind(block_iter, binding_value)

    Example
    -------
    .. code-block:: python

        tir.bind(vx, i)
    """

    def __init__(self):
        def bind(iter_var, values, span=None):
            block_scope = self.context.current_block_scope()
            if iter_var in block_scope.iter_bindings:
                self.context.report_error("Duplicate iter_var bindings of " + str(iter_var), span)
            block_scope.iter_bindings[iter_var] = values

        super().__init__(bind, def_symbol=False)


@register
class BlockReads(SpecialStmt):
    """Special function reads([read_buffer_regions])

    Example
    -------
    .. code-block:: python

        tir.reads([A[vi: vi + 4, vk: vk + 4], B[vk: vk + 4, vj]])
    """

    def __init__(self):
        def reads(read_regions: Union[BufferSlice, List[BufferSlice]], span: Span = None):
            assert self.context, "call 'exit_scope' before 'enter_scope'"
            block_scope = self.context.current_block_scope()
            if block_scope.reads is not None:
                self.context.report_error(
                    "Duplicate write region declaration, "
                    + "previous one is "
                    + str(", ".join(str(x) for x in block_scope.reads)),
                    span,
                )
            if isinstance(read_regions, BufferSlice):
                read_regions = [read_regions]
            if not isinstance(read_regions, list):
                self.context.report_error(
                    "Incorrect input type. "
                    + f"Expected BufferSlice or List[BufferSlice], but got {type(read_regions)}",
                    span,
                )
            block_scope.reads = read_regions

        super().__init__(reads, def_symbol=False)


@register
class BlockWrites(SpecialStmt):
    """Special function writes([write_buffer_regions])

    Example
    -------
    .. code-block:: python

        tir.writes([C[vi: vi + 4, vj])
    """

    def __init__(self):
        def writes(write_region: Union[BufferSlice, List[BufferSlice]], span: Span = None):
            assert self.context, "call 'exit_scope' before 'enter_scope'"
            block_scope = self.context.current_block_scope()
            if block_scope.writes is not None:
                self.context.report_error(
                    "Duplicate write region declaration, "
                    + "previous one is "
                    + str(", ".join(str(x) for x in block_scope.writes)),
                    span,
                )
            if isinstance(write_region, list):
                pass
            elif isinstance(write_region, BufferSlice):
                write_region = [write_region]
            else:
                self.context.report_error(
                    "Incorrect input type. "
                    + f"Expected BufferSlice or List[BufferSlice], but got {type(write_region)}",
                    span,
                )
            block_scope.writes = write_region

        super().__init__(writes, def_symbol=False)


@register
class BlockAttr(SpecialStmt):
    """Special function block_attr({attr_key: attr_value})

    Example
    -------
    .. code-block:: python

        tir.block_attr({"double_buffer_scope": 1})
    """

    def __init__(self):
        def block_attr(attrs: Mapping[str, Object], span: Span = None):
            assert self.context, "call 'exit_scope' before 'enter_scope'"
            block_scope = self.context.current_block_scope()
            if block_scope.annotations is not None:
                self.context.report_error(
                    "Duplicate block annotations declaration, "
                    + "previous one is "
                    + str(block_scope.annotations),
                    span,
                )
            attrs = {
                key: tvm.tir.StringImm(val) if isinstance(val, str) else val
                for key, val in attrs.items()
            }
            block_scope.annotations = attrs

        super().__init__(block_attr, def_symbol=False)


@register
class BlockPredicate(SpecialStmt):
    """Special function where(predicate)

    Example
    -------
    .. code-block:: python

        tir.where(i < 4)
    """

    def __init__(self):
        def where(predicate, span=None):
            assert self.context, "call 'exit_scope' before 'enter_scope'"
            block_scope = self.context.current_block_scope()
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
class BlockMatchBufferRegion(SpecialStmt):
    """Special function match_buffer_region(source, strides, elem_offset, align, offset_factor)

    Example
    -------
    .. code-block:: python

        B = tir.match_buffer_region(A[0: 4])
    """

    def __init__(self):
        def match_buffer_region(
            source,
            strides=None,
            elem_offset=None,
            align=-1,
            offset_factor=0,
            span=None,
        ):
            assert self.context, "call 'exit_scope' before 'enter_scope'"
            if not isinstance(self.node, ast.Assign):
                self.context.report_error(
                    "match_buffer_region must be assigned to a buffer, "
                    + "e.g. A = match_buffer_region(...)",
                    self.node.span,
                )

            if strides is None:
                strides = []
            align = convert_to_int(align, "align", self.context.report_error, self.node.span)
            offset_factor = convert_to_int(
                offset_factor, "offset_factor", self.context.report_error, self.node.span
            )

            if not isinstance(source, BufferSlice):
                self.context.report_error(
                    "match_buffer_region needs a buffer region as source",
                    span=span,
                )
            buffer_region = buffer_slice_to_region(source)
            shape = [r.extent for r in buffer_region.region]
            buffer = tvm.tir.decl_buffer(
                shape,
                buffer_region.buffer.dtype,
                self.node.lhs.id.name,
                data=None,
                strides=strides,
                elem_offset=elem_offset,
                scope=buffer_region.buffer.scope,
                data_alignment=align,
                offset_factor=offset_factor,
                span=span,
            )
            self.context.current_block_scope().match_buffers.append(
                tvm.tir.MatchBufferRegion(buffer, buffer_region)
            )
            self.context.update_symbol(self.node.lhs.id.name, buffer, self.node)

        super().__init__(match_buffer_region, def_symbol=True)


@register
class VarDef(SpecialStmt):
    """Special function for defining a Var"""

    def __init__(self):
        def var(dtype, span):
            assert isinstance(
                self.node, ast.Assign
            ), f"VarDef expected ast.Assign but got {type(self.node)}"
            v = te.var(self.node.lhs.id.name, dtype, span=span)
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
            ptr_type = tvm.ir.PointerType(tvm.ir.PrimType(dtype), storage_scope)
            v = te.var(self.node.lhs.id.name, ptr_type, span=span)
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
            v = te.var(self.node.lhs.id.name, span=span)
            self.context.func_var_env_dict[v] = env_name
            self.context.update_symbol(v.name, v, self.node)

        super().__init__(env_thread, def_symbol=True)


@register
class FuncAttr(SpecialStmt):
    """Special Stmt for declaring the DictAttr of PrimFunc
    Example
    -------
    .. code-block:: python
         tir.func_attr({"tir.noalias": True, "global_symbol"})
    """

    def __init__(self):
        def func_attr(dict_attr, span):
            self.context.func_dict_attr = dict_attr

        super().__init__(func_attr, def_symbol=False)
