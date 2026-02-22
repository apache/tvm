# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, wrong-import-position
"""The Relax IR namespace containing the IR, type, operator, builder, vm, etc."""

from tvm.runtime import vm
from tvm.runtime.vm import VirtualMachine, VMInstrumentReturnKind

# Import submodules in the last to avoid dependency
from . import (
    analysis,
    backend,
    block_builder,
    distributed,
    exec_builder,
    expr,
    frontend,
    op,
    struct_info,
    training,
    transform,
    ty,
    utils,
)

# BasePyModule
from .base_py_module import BasePyModule
from .binding_rewrite import DataflowBlockRewrite

# BlockBuilder
from .block_builder import BlockBuilder

# VM
from .exec_builder import ExecBuilder

# Expr
from .expr import (
    Binding,
    BindingBlock,
    Call,
    Constant,
    DataflowBlock,
    DataflowVar,
    DataTypeImm,
    Expr,
    ExternFunc,
    Function,
    GlobalVar,
    Id,
    If,
    MatchCast,
    PrimValue,
    SeqExpr,
    ShapeExpr,
    Span,
    StringImm,
    Tuple,
    TupleGetItem,
    Var,
    VarBinding,
    const,
    extern,
    get_shape_of,
)

# ExprFunctor
from .expr_functor import ExprFunctor, PyExprMutator, PyExprVisitor

# Operator
from .op.base import (
    call_dps_packed,
    call_pure_packed,
    call_tir,
    call_tir_inplace,
    call_tir_with_grad,
)

# pipeline
from .pipeline import get_default_pipeline, get_pipeline, register_pipeline

# StructInfo
from .struct_info import (
    FuncStructInfo,
    ObjectStructInfo,
    PrimStructInfo,
    ShapeStructInfo,
    StructInfo,
    TensorStructInfo,
    TupleStructInfo,
)

# Type
from .ty import (
    FuncType,
    ObjectType,
    PackedFuncType,
    ShapeType,
    TensorType,
    TupleType,
    Type,
)
from .type_converter import args_converter

# utils
from .utils import convert_to_expr

# VM
from .vm_build import VMExecutable, build
