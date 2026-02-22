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
"""Namespace for Tensor Expression Language"""

# expose all operators in tvm tir.op
from tvm.tir import (
    abs,
    acos,
    acosh,
    add,
    all,
    any,
    asin,
    asinh,
    atan,
    atanh,
    ceil,
    comm_reducer,
    cos,
    cosh,
    div,
    erf,
    exp,
    floor,
    floordiv,
    floormod,
    fmod,
    if_then_else,
    indexdiv,
    indexmod,
    isfinite,
    isinf,
    isnan,
    log,
    log2,
    log10,
    logaddexp,
    max,
    max_value,
    min,
    min_value,
    multiply,
    nearbyint,
    popcount,
    power,
    round,
    rsqrt,
    sigmoid,
    sin,
    sinh,
    sqrt,
    subtract,
    sum,
    tan,
    tanh,
    trace,
    trunc,
    truncdiv,
    truncmod,
)

from .operation import (
    AXIS_SEPARATOR,
    compute,
    const,
    create_prim_func,
    extern,
    extern_primfunc,
    placeholder,
    reduce_axis,
    scan,
    size_var,
    thread_axis,
    var,
)
from .tag import tag_scope
from .tensor import ComputeOp, ExternOp, PlaceholderOp, ScanOp, Tensor, TensorSlice
