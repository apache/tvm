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
# pylint: disable=unused-import, redefined-builtin
"""Namespace for Tensor-level IR"""
from tvm.ir import PrimExpr
from tvm.runtime import const

from .buffer import Buffer, decl_buffer
from .data_layout import Layout, BijectiveLayout, bijective_layout, layout
from .expr import Var, SizeVar, Reduce, FloatImm, IntImm, StringImm, Cast
from .expr import Add, Sub, Mul, Div, Mod, FloorDiv, FloorMod
from .expr import Min, Max, EQ, NE, LT, LE, GT, GE, And, Or, Not
from .expr import Select, Load, Ramp, Broadcast, Shuffle, Call, Let
from .expr import IterVar, Any

from .stmt import Stmt, LetStmt, AssertStmt, ProducerConsumer, For
from .stmt import Store, Provide, Allocate, AttrStmt, Free, Realize, SeqStmt
from .stmt import IfThenElse, Evaluate, Prefetch, LoweredFunc, stmt_seq, stmt_list

from .op import call_packed, call_pure_intrin, call_intrin, call_pure_extern, call_extern
from .op import call_llvm_intrin, all, any, min_value, max_value, trace
from .op import exp, erf, tanh, sigmoid, log, tan, cos, sin, atan, sqrt, rsqrt, floor, ceil
from .op import trunc, abs, round, nearbyint, isnan, power, popcount, fmod, if_then_else
from .op import div, indexdiv, indexmod, truncdiv, truncmod, floordiv, floormod
from .op import comm_reducer, min, max, sum

from . import ir_builder
from . import ir_pass
