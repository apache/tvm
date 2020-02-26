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
# pylint: disable=wildcard-import, redefined-builtin, invalid-name
"""The Relay IR namespace containing the IR definition and compiler."""
import os
from sys import setrecursionlimit
from ..api import register_func
from . import base
from . import ty
from . import expr
from . import type_functor
from . import expr_functor
from . import adt
from . import analysis
from . import transform
from .build_module import build, create_executor, optimize
from .transform import build_config
from . import prelude
from . import parser
from . import debug
from . import param_dict
from . import feature
from .backend import vm

# Root operators
from .op import Op
from .op.reduce import *
from .op.tensor import *
from .op.transform import *
from .op.algorithm import *
from . import nn
from . import annotation
from . import vision
from . import contrib
from . import image
from . import frontend
from . import backend
from . import quantize

# Dialects
from . import qnn

from .scope_builder import ScopeBuilder
# Load Memory pass
from . import memory_alloc

# Required to traverse large programs
setrecursionlimit(10000)

# Span
Span = base.Span

# Type
Type = ty.Type
TupleType = ty.TupleType
TensorType = ty.TensorType
TypeKind = ty.TypeKind
TypeVar = ty.TypeVar
ShapeVar = ty.ShapeVar
TypeConstraint = ty.TypeConstraint
FuncType = ty.FuncType
TypeRelation = ty.TypeRelation
IncompleteType = ty.IncompleteType
scalar_type = ty.scalar_type
RefType = ty.RefType
GlobalTypeVar = ty.GlobalTypeVar
TypeCall = ty.TypeCall
Any = ty.Any

# Expr
Expr = expr.RelayExpr
Constant = expr.Constant
Tuple = expr.Tuple
Var = expr.Var
GlobalVar = expr.GlobalVar
Function = expr.Function
Call = expr.Call
Let = expr.Let
If = expr.If
TupleGetItem = expr.TupleGetItem
RefCreate = expr.RefCreate
RefRead = expr.RefRead
RefWrite = expr.RefWrite

# ADT
PatternWildcard = adt.PatternWildcard
PatternVar = adt.PatternVar
PatternConstructor = adt.PatternConstructor
PatternTuple = adt.PatternTuple
Constructor = adt.Constructor
TypeData = adt.TypeData
Clause = adt.Clause
Match = adt.Match

# helper functions
var = expr.var
const = expr.const
bind = expr.bind
module_pass = transform.module_pass
function_pass = transform.function_pass
alpha_equal = analysis.alpha_equal

# TypeFunctor
TypeFunctor = type_functor.TypeFunctor
TypeVisitor = type_functor.TypeVisitor
TypeMutator = type_functor.TypeMutator

# ExprFunctor
ExprFunctor = expr_functor.ExprFunctor
ExprVisitor = expr_functor.ExprVisitor
ExprMutator = expr_functor.ExprMutator

# Parser
fromtext = parser.fromtext

# Param Serialization
save_param_dict = param_dict.save_param_dict
load_param_dict = param_dict.load_param_dict

# Pass manager
PassInfo = transform.PassInfo
PassContext = transform.PassContext
Pass = transform.Pass
ModulePass = transform.ModulePass
FunctionPass = transform.FunctionPass
Sequential = transform.Sequential

# Feature
Feature = feature.Feature
