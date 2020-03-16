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

from . import ir
from .ir import adt, expr, ty, base, scope_builder
from .ir import prelude, loops, parser

from . import transform
from . import analysis
from .analysis import call_graph, feature, alpha_equal
from .build_module import build, create_executor, optimize
from .transform import build_config
from . import debug
from . import param_dict
from .backend import vm

# Root operators
from .op import Op
from .op import nn
from .op import image
from .op import vision
from .op import annotation
from .op.reduce import *
from .op.tensor import *
from .op.transform import *
from .op.algorithm import *
from .op.nn import *
from .op.vision import *
from .op.contrib import *
from .op.image import *
from . import frontend
from . import backend
from . import quantize

# Dialects
from . import qnn

# Load Memory pass
from .transform import memory_alloc

# Required to traverse large programs
setrecursionlimit(10000)

# Span
Span = ir.Span

# Type
Type = ir.Type
TupleType = ir.TupleType
TensorType = ir.TensorType
TypeKind = ir.TypeKind
TypeVar = ir.TypeVar
ShapeVar = ir.ShapeVar
TypeConstraint = ir.TypeConstraint
FuncType = ir.FuncType
TypeRelation = ir.TypeRelation
IncompleteType = ir.IncompleteType
scalar_type = ir.scalar_type
RefType = ir.RefType
GlobalTypeVar = ir.GlobalTypeVar
TypeCall = ir.TypeCall
Any = ir.Any

# Expr
Expr = ir.Expr
Constant = ir.Constant
Tuple = ir.Tuple
Var = ir.Var
GlobalVar = ir.GlobalVar
Function = ir.Function
Call = ir.Call
Let = ir.Let
If = ir.If
TupleGetItem = ir.TupleGetItem
RefCreate = ir.RefCreate
RefRead = ir.RefRead
RefWrite = ir.RefWrite

# ADT
Pattern = ir.Pattern
PatternWildcard = ir.PatternWildcard
PatternVar = ir.PatternVar
PatternConstructor = ir.PatternConstructor
PatternTuple = ir.PatternTuple
Constructor = ir.Constructor
TypeData = ir.TypeData
Clause = ir.Clause
Match = ir.Match

# helper functions
var = ir.var
const = ir.const
bind = ir.bind

# TypeFunctor
TypeFunctor = ir.TypeFunctor
TypeVisitor = ir.TypeVisitor
TypeMutator = ir.TypeMutator

# ExprFunctor
ExprFunctor = ir.ExprFunctor
ExprVisitor = ir.ExprVisitor
ExprMutator = ir.ExprMutator

# Prelude
Prelude = prelude.Prelude

# Scope builder
ScopeBuilder = scope_builder.ScopeBuilder

module_pass = transform.module_pass
function_pass = transform.function_pass

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

# CallGraph
CallGraph = call_graph.CallGraph
