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

# Expr
from .expr import (
    Expr,
    Span,
    Id,
    GlobalVar,
    Var,
    DataflowVar,
    Binding,
    MatchCast,
    VarBinding,
    BindingBlock,
    DataflowBlock,
    SeqExpr,
    ShapeExpr,
    Tuple,
    TupleGetItem,
    Function,
    ExternFunc,
    Call,
    If,
    Constant,
    PrimValue,
    DataTypeImm,
    StringImm,
)

from .expr import const, extern, get_shape_of

# Type
from .ty import (
    Type,
    ObjectType,
    ShapeType,
    DynTensorType,
    TupleType,
    FuncType,
    PackedFuncType,
)

# VM
from .exec_builder import ExecBuilder

# Operator
from .op.base import (
    call_tir,
    call_tir_inplace,
    call_pure_packed,
    call_dps_packed,
    call_tir_with_grad,
)

# BlockBuilder
from .block_builder import BlockBuilder

# ExprFunctor
from .expr_functor import ExprFunctor, PyExprVisitor, PyExprMutator

# StructInfo
from .struct_info import (
    StructInfo,
    ObjectStructInfo,
    PrimStructInfo,
    ShapeStructInfo,
    TensorStructInfo,
    TupleStructInfo,
    FuncStructInfo,
)

# pipeline
from .pipeline import get_pipeline
from .pipeline import register_pipeline

# Import submodules in the last to avoid dependency
from . import exec_builder
from . import expr
from . import ty
from . import analysis
from . import transform
from . import block_builder
from . import op
from . import struct_info
from . import backend
from . import training
from . import distributed
from . import frontend

# VM
from .vm_build import build, Executable

from .binding_rewrite import DataflowBlockRewrite
