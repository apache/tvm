# isort: skip_file
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
# pylint: disable=unused-import
"""Common data structures across all IR variants."""

from . import instrument, transform
from ._constant import const
from .attrs import Attrs, DictAttrs, make_node
from .base import (
    EnvFunc,
    Node,
    Scriptable,
    SourceName,
    Span,
    SequentialSpan,
    assert_structural_equal,
    load_json,
    save_json,
)

# Register Type before Expr.  Expr's reflected ``ty`` field otherwise creates
# an auto-generated Type wrapper before the concrete Python class is available.
from .type import (
    AnyType,
    FuncType,
    MissingType,
    OpaqueType,
    PointerType,
    PrimType,
    StringType,
    TensorRegionType,
    TupleType,
    Type,
)
from .expr import (
    Call,
    Constant,
    GenericConst,
    DataTypeImm,
    StringImm,
    Expr,
    ExprOperand,
    ExprWithOp,
    GlobalVar,
    LambdaExpr,
    StagingExpr,
    OpaqueExpr,
    Range,
    TensorLoad,
    TensorRegion,
    Tuple,
    TupleGetItem,
    Var,
    is_prim_expr,
    is_prim_var,
    reinfer_type,
)
from . import prim
from .function import BaseFunc, CallingConv
from .global_info import GlobalInfo
from .module import IRModule
from .op import Op, register_op_attr
from .stmt import (
    Stmt,
    SeqStmt,
    Bind,
    Evaluate,
    Return,
    If,
    ForKind,
    For,
    While,
    Break,
    Continue,
    AssertStmt,
    RegionStmt,
    TensorStore,
    stmt_seq,
    stmt_list,
)

from tvm_ffi import Array, Map
