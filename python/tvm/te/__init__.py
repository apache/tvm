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
# pylint: disable=unused-import, redefined-builtin, wildcard-import
"""Namespace for Tensor Expression Language
"""
# expose all operators in tvm tir.op
from tvm.tir import any, all, min_value, max_value, trace
from tvm.tir import exp, erf, tanh, sigmoid, log, tan, cos, sin, sqrt, rsqrt, floor, ceil
from tvm.tir import sinh, cosh, log2, log10
from tvm.tir import asin, asinh, acos, acosh, atan, atanh
from tvm.tir import trunc, abs, round, nearbyint, power, popcount, fmod, if_then_else
from tvm.tir import isnan, isfinite, isinf
from tvm.tir import div, indexdiv, indexmod, truncdiv, truncmod, floordiv, floormod
from tvm.tir import comm_reducer, min, max, sum

from .schedule import (
    Schedule,
    Stage,
    create_schedule,
    SpecializedCondition,
    AXIS_SEPARATOR,
)
from .tensor import TensorSlice, Tensor
from .tensor_intrin import decl_tensor_intrin
from .tag import tag_scope
from .operation import placeholder, compute, scan, extern, var, size_var, const
from .operation import thread_axis, reduce_axis
from .operation import create_prim_func
from .operation import extern_primfunc

from .tensor import PlaceholderOp, ComputeOp, TensorComputeOp, ScanOp, ExternOp, HybridOp
from .autodiff import gradient
from . import hybrid
