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
# pylint: disable=redefined-builtin
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    Iterable,
    Object,
    Optional,
    Tuple,
    Union,
    Sequence,
    List,
    Mapping,
)
from tvm.tir.function import PrimFunc
from tvm.tir import PrimExpr, Buffer, IterVar, Var
from .node import BufferSlice

"""
Variables and constants
"""

def bool(imm: int) -> PrimExpr: ...
def int8(imm: int) -> PrimExpr: ...
def int16(imm: int) -> PrimExpr: ...
def int32(imm: int) -> PrimExpr: ...
def int64(imm: int) -> PrimExpr: ...
def uint8(imm: int) -> PrimExpr: ...
def uint16(imm: int) -> PrimExpr: ...
def uint32(imm: int) -> PrimExpr: ...
def uint64(imm: int) -> PrimExpr: ...
def float8(imm: int) -> PrimExpr: ...
def float16(imm: int) -> PrimExpr: ...
def float32(imm: int) -> PrimExpr: ...
def float64(imm: int) -> PrimExpr: ...

"""
Intrinsic
"""

def min_value(dtype): ...
def max_value(dtype): ...
def floordiv(x: PrimExpr, y: PrimExpr): ...
def floormod(x: PrimExpr, y: PrimExpr): ...
def abs(x): ...
def load(dtype, var, index, predicate=None): ...
def cast(value, dtype): ...
def ramp(base, stride, lanes): ...
def broadcast(value, lanes): ...
def iter_var(var, dom, iter_type, thread_tag): ...
def max(a, b): ...
def min(a, b): ...
def get_axis(begin, end, iter_type): ...
def range(begin, end): ...
def reduce_axis(begin, end): ...
def scan_axis(begin, end): ...
def opaque_axis(begin, end): ...
def Select(cond, if_body, else_body): ...
def evaluate(value): ...
def store(var, index, value, predicate=True): ...
def comm_reducer(lambda_io, identities): ...

"""
Unary operator
"""

def exp2(x: PrimExpr) -> PrimExpr: ...
def exp10(x: PrimExpr) -> PrimExpr: ...
def erf(x: PrimExpr) -> PrimExpr: ...
def tanh(x: PrimExpr) -> PrimExpr: ...
def sigmoid(x: PrimExpr) -> PrimExpr: ...
def log(x: PrimExpr) -> PrimExpr: ...
def log2(x: PrimExpr) -> PrimExpr: ...
def log10(x: PrimExpr) -> PrimExpr: ...
def log1p(x: PrimExpr) -> PrimExpr: ...
def tan(x: PrimExpr) -> PrimExpr: ...
def cos(x: PrimExpr) -> PrimExpr: ...
def cosh(x: PrimExpr) -> PrimExpr: ...
def acos(x: PrimExpr) -> PrimExpr: ...
def acosh(x: PrimExpr) -> PrimExpr: ...
def sin(x: PrimExpr) -> PrimExpr: ...
def sinh(x: PrimExpr) -> PrimExpr: ...
def asin(x: PrimExpr) -> PrimExpr: ...
def asinh(x: PrimExpr) -> PrimExpr: ...
def atan(x: PrimExpr) -> PrimExpr: ...
def atanh(x: PrimExpr) -> PrimExpr: ...
def atan2(x: PrimExpr) -> PrimExpr: ...
def sqrt(x: PrimExpr) -> PrimExpr: ...
def rsqrt(x: PrimExpr) -> PrimExpr: ...

"""
Axis
"""

def reduce_axis(begin: Union[PrimExpr, int], end: Union[PrimExpr, int]) -> IterVar: ...
def range(begin: Union[PrimExpr, int], end: Union[PrimExpr, int]) -> IterVar: ...
def scan_axis(begin: Union[PrimExpr, int], end: Union[PrimExpr, int]) -> IterVar: ...
def opaque_axis(begin: Union[PrimExpr, int], end: Union[PrimExpr, int]) -> IterVar: ...

"""
Buffers
"""

def match_buffer(
    param: Union[Var, BufferSlice],
    shape: Sequence[Union[PrimExpr, int]],
    dtype: str = "float32",
    data=None,
    strides: Optional[Sequence[int]] = None,
    elem_offset: Optional[int] = None,
    scope: str = "global",
    align: int = -1,
    offset_factor: int = 0,
    buffer_type: str = "default",
) -> Buffer: ...
def buffer_decl(
    shape: Sequence[Union[PrimExpr, int]],
    dtype: str = "float32",
    data=None,
    strides: Optional[Sequence[int]] = None,
    elem_offset: Optional[int] = None,
    scope: str = "global",
    align: int = -1,
    offset_factor: int = 0,
    buffer_type: str = "default",
) -> Buffer: ...
def alloc_buffer(
    shape: Sequence[Union[PrimExpr, int]],
    dtype: str = "float32",
    data=None,
    strides: Optional[Sequence[int]] = None,
    elem_offset: Optional[int] = None,
    scope: str = "global",
    align: int = -1,
    offset_factor: int = 0,
    buffer_type: str = "default",
) -> Buffer: ...

"""
Reads/Writes
"""

def reads(read_regions: Union[BufferSlice, List[BufferSlice]]) -> None: ...
def writes(write_region: Union[BufferSlice, List[BufferSlice]]) -> None: ...
def block_attr(attrs: Mapping[str, Object]) -> None: ...

"""
Scope handler
"""

class block(ContextManager):
    def __init__(self, axes: Sequence[Union[int, PrimExpr, slice]], name: str = "") -> None: ...
    def __enter__(self) -> Sequence[IterVar]: ...

class init(ContextManager):
    def __init__(self) -> None: ...

class let(ContextManager):
    def __init__(self, var: Var, value: PrimExpr) -> None: ...

def where(cond: PrimExpr) -> None: ...
def allocate(extents, dtype, scope: str, condition: bool = True, annotations=None) -> None: ...
def launch_thread(env_var, extent): ...
def realize(buffer_slice: BufferSlice, scope: str, condition: bool = True) -> None: ...
def attr(attr_node, attr_key, value) -> None: ...
def Assert(condition, message): ...
def let(var, value): ...
def block(name_hint: str = ""): ...
def init(): ...

"""
Scope handler - Loops
"""

def serial(
    begin: PrimExpr,
    end: PrimExpr,
    annotations: Optional[Mapping[str, Object]] = None,
) -> None: ...
def parallel(
    begin: PrimExpr,
    end: PrimExpr,
    annotations: Optional[Mapping[str, Object]] = None,
) -> None: ...
def vectorized(
    begin: PrimExpr,
    end: PrimExpr,
    annotations: Optional[Mapping[str, Object]] = None,
) -> None: ...
def unroll(
    begin: PrimExpr,
    end: PrimExpr,
    annotations: Optional[Mapping[str, Object]] = None,
) -> None: ...
def thread_binding(
    begin: PrimExpr,
    end: PrimExpr,
    thread: str,
    annotations: Optional[Mapping[str, Object]] = None,
) -> None: ...
def for_range(
    begin: PrimExpr,
    end: PrimExpr = None,
    annotations: Optional[Mapping[str, Object]] = None,
) -> None: ...
def grid(*extents: List[PrimExpr]) -> None: ...

"""
Threads and Bindings
"""

def env_thread(thread: str) -> IterVar: ...
def bind(iter_var: IterVar, expr: PrimExpr) -> None: ...

"""
Annotations
"""

def func_attr(attrs: Dict) -> None: ...
def block_attr(attrs: Dict) -> None: ...
def attr(node: PrimExpr, attr_key: str, value: PrimExpr) -> None: ...
def prim_func(input_func: Callable) -> PrimFunc: ...