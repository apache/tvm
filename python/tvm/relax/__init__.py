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
from tvm.runtime import relax_vm as vm
from tvm.runtime.relax_vm import VirtualMachine, VMInstrumentReturnKind

# Import submodules in the last to avoid dependency
from . import (
    analysis,
    backend,
    block_builder,
    exec_builder,
    expr,
    frontend,
    op,
    struct_info,
    transform,
    ty,
)
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
    SourceName,
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
from .op import call_dps_packed, call_tir

# pipeline
from .pipeline import get_pipeline

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
    DynTensorType,
    FuncType,
    ObjectType,
    PackedFuncType,
    ShapeType,
    TupleType,
    Type,
)

# VM
from .vm_build import Executable, build
