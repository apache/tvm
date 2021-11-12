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

from .buffer import Buffer, decl_buffer, DataProducer
from .data_layout import Layout, BijectiveLayout, bijective_layout, layout
from .expr import Var, SizeVar, Reduce, FloatImm, IntImm, StringImm, Cast
from .expr import Add, Sub, Mul, Div, Mod, FloorDiv, FloorMod
from .expr import Min, Max, EQ, NE, LT, LE, GT, GE, And, Or, Not
from .expr import Select, BufferLoad, ProducerLoad, Load, Ramp, Broadcast, Shuffle
from .expr import Call, CallEffectKind, Let, IterVar, CommReducer, Any

from .stmt import Stmt, LetStmt, AssertStmt, ForKind, For, While
from .stmt import BufferStore, BufferRealize, Store, ProducerStore, Allocate, AttrStmt
from .stmt import ProducerRealize, SeqStmt
from .stmt import IfThenElse, Evaluate, Prefetch, stmt_seq, stmt_list
from .stmt import BufferRegion, MatchBufferRegion, Block, BlockRealize

from .function import PrimFunc

from .op import call_packed, call_intrin, call_pure_extern, call_extern
from .op import call_llvm_intrin, call_llvm_pure_intrin, ret, all, any, min_value, max_value, trace
from .op import exp, exp2, exp10, log, log2, log10, log1p, ldexp, clz
from .op import sin, sinh, asin, asinh
from .op import cos, cosh, acos, acosh
from .op import tan, tanh, atan, atan2, atanh
from .op import erf, sigmoid, sqrt, rsqrt, floor, ceil, hypot
from .op import trunc, abs, round, nextafter, nearbyint, power, popcount, fmod, if_then_else
from .op import isnan, isfinite, isinf, copysign
from .op import div, indexdiv, indexmod, truncdiv, truncmod, floordiv, floormod
from .op import comm_reducer, min, max, sum
from .op import q_multiply_shift

from .schedule import StmtSRef, BlockScope, ScheduleState, Schedule, ScheduleError

from . import schedule
from . import ir_builder
from . import transform
from . import analysis
from . import stmt_functor
