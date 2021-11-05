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
"""TVM Script Interface for Sparse TIR"""
import synr
import tvm
from synr import ast
from tvm.ir.base import Span
from tvm.ir.expr import PrimExpr, Range

from tvm.script.tir.node import BufferSlice
from tvm.script.tir.utils import buffer_slice_to_region
from tvm.tir.expr import PrimExprWithOp
from .scope_handler import ScopeHandler, LoopInfo
from .intrin import Intrin
from ..context_maintainer import BlockInfo, ContextMaintainer
from .special_stmt import SpecialStmt
from tvm.tir.sparse import Axis, AxisTree, DenseFixedAxis, DenseVariableAxis, SpIterVar, SparseFixedAxis, SparseVariableAxis
from typing import List, Mapping, Optional, Tuple, Any
from tvm.runtime.object import Object
from tvm.script.registry import register
from ..utils import (
    tvm_span_from_synr,
    call_with_error_reporting,
)


@register
class DenseFixed(SpecialStmt):
    """Special Stmt for creating dense fixed axis.
    """

    def __init__(self):
        def dense_fixed(
            name: str,
            length: PrimExpr,
            idtype: str = 'int32',
            span: Optional[Span] = None
        ):
            var_name = self.node.lhs[0].id.name
            axis = DenseFixedAxis(name, length, idtype=idtype)
            self.context.update_symbol(var_name, axis, self.node)
        super().__init__(dense_fixed, def_symbol=True)


@register
class DenseVariable(SpecialStmt):
    """Special Stmt for creating dense variable axis.
    """

    def __init__(self):
        def dense_variable(
            name: str,
            shape: Tuple[PrimExpr, PrimExpr],
            indptr: tvm.tir.Var,
            idtype: str = 'int32',
            span: Optional[Span] = None
        ):
            indptr_len, length = shape
            var_name = self.node.lhs[0].id.name
            indptr_buf = tvm.tir.decl_buffer(
                (indptr_len,),
                dtype=idtype,
                name=name + "_indptr",
                span=span
            )
            axis = DenseVariableAxis(name, length, indptr_buf, idtype=idtype)
            self.context.func_buffer_map[indptr] = indptr_buf
            self.context.update_symbol(var_name, axis, self.node)
        super().__init__(dense_variable, def_symbol=True)


@register
class SparseFixed(SpecialStmt):
    """Special Stmt for creating sparse fixed axis.
    """

    def __init__(self):
        def sparse_fixed(
            name: str,
            shape: Tuple[PrimExpr, PrimExpr, PrimExpr],
            indices: tvm.tir.Var,
            idtype: str = 'int32',
            span: Optional[Span] = None
        ):
            var_name = self.node.lhs[0].id.name
            length, nnz, nnz_cols = shape
            indices_buf = tvm.tir.decl_buffer(
                (nnz,),
                dtype=idtype,
                name=name+"_indices",
                span=span
            )
            axis = SparseFixedAxis(name, length, indices_buf, nnz_cols, idtype=idtype)
            self.context.func_buffer_map[indices] = indices_buf
            self.context.update_symbol(var_name, axis, self.node)
        super().__init__(sparse_fixed, def_symbol=True)


@register
class SparseVariable(SpecialStmt):
    """Special Stmt for creating sparse variable axis:
    """

    def __init__(self):
        def sparse_variable(
            name: str,
            shape: Tuple[PrimExpr, PrimExpr],
            data: Tuple[tvm.tir.Var, tvm.tir.Var],
            idtype: str = 'int32',
            span: Optional[Span] = None
        ):
            var_name = self.node.lhs[0].id.name
            length, indptr_len, nnz = shape
            indptr, indices = data
            indptr_buf = tvm.tir.decl_buffer(
                (indptr_len,),
                dtype=idtype,
                name=name+"_indptr",
                span=span
            )
            indices_buf = tvm.tir.decl_buffer(
                (nnz,),
                dtype=idtype,
                name=name+"_indices",
                span=span
            )
            axis = SparseVariableAxis(name, length, indptr_buf, indices_buf, idtype=idtype)
            self.context.func_buffer_map[indices] = indices_buf
            self.context.func_buffer_map[indptr] = indptr_buf
            self.context.update_symbol(var_name, axis, self.node)
        super().__init__(sparse_variable, def_symbol=True)


@register
class MatchSparseBuffer(SpecialStmt):
    """Special Stmt match_sparse_buffer()
    """

    def __init__(self):
        def match_sparse_buffer(
            param: tvm.tir.Var,
            axes: List[Axis],
            dtype: str = 'float32',
            span: Optional[Span] = None,
        ):
            if not isinstance(self.node, ast.Assign) or not len(self.node.lhs) == 1:
                self.context.report_error(
                    "`match_sparse_buffer` must be assigned to a single sparse buffer, "
                    "e.g. A = match_sparse_buffer(...)"
                )

            buffer_name: str = self.node.lhs[0].id.name
            if not isinstance(param, tvm.tir.Var):
                self.context.report_error(
                    "The source of match_sparse_buffer expected Var, but got"
                    + str(type(param)),
                    self.node.rhs.params[0].span
                )
 
            if param in self.context.func_params:
                buffer = tvm.tir.sparse.decl_buffer(
                    axes,
                    param,
                    buffer_name,
                    dtype,
                    span=span
                )
                self.context.func_sparse_buffer_map[param] = buffer
                self.context.update_symbol(buffer_name, buffer, self.node)
            else:
                self.context.report_error(
                    "Can not bind non-input param to sparse buffer", self.node.rhs.params[0].span
                )

        super().__init__(match_sparse_buffer, def_symbol=True)


@register
def to_dense(axis: Axis, span: Optional[Span] = None):
    if isinstance(axis, (SparseFixedAxis, SparseVariableAxis)):
        return DenseFixedAxis(axis.name, axis.length, axis.idtype) 
    else:
        return axis


@register
def cord(axis: Axis, span: Optional[Span] = None):
    return 'cord', axis


@register
def pos(axis: Axis, span: Optional[Span] = None):
    return 'pos', axis
