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
    Optional,
    Tuple,
    Union,
    Sequence,
    List,
    Mapping,
    overload,
)
from numbers import Number
import builtins

from tvm.tir.function import PrimFunc
from tvm.tir import PrimExpr, Range, IterVar, Var
from tvm.runtime import Object
from .node import BufferSlice

"""
redefine types
"""

class Buffer(Var):
    def __getitem__(
        self: Buffer, pos: Tuple[Union[int, PrimExpr], Union[int, PrimExpr]]
    ) -> Buffer: ...
    @property
    def data(self: Buffer) -> Ptr: ...

"""
Variables and constants
"""

def bool(imm: Union[PrimExpr, builtins.bool, Number]) -> PrimExpr: ...
def int8(imm: Union[PrimExpr, Number]) -> PrimExpr: ...
def int16(imm: Union[PrimExpr, Number]) -> PrimExpr: ...
def int32(imm: Union[PrimExpr, Number]) -> PrimExpr: ...
def int64(imm: Union[PrimExpr, Number]) -> PrimExpr: ...
def uint8(imm: Union[PrimExpr, Number]) -> PrimExpr: ...
def uint16(imm: Union[PrimExpr, Number]) -> PrimExpr: ...
def uint32(imm: Union[PrimExpr, Number]) -> PrimExpr: ...
def uint64(imm: Union[PrimExpr, Number]) -> PrimExpr: ...
def float8(imm: Union[PrimExpr, Number]) -> PrimExpr: ...
def float16(imm: Union[PrimExpr, Number]) -> PrimExpr: ...
def float32(imm: Union[PrimExpr, Number]) -> PrimExpr: ...
def float64(imm: Union[PrimExpr, Number]) -> PrimExpr: ...

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
special_stmt - Buffers
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
special_stmt - Reads/Writes
"""

def reads(read_regions: Union[BufferSlice, List[BufferSlice]]) -> None: ...
def writes(write_region: Union[BufferSlice, List[BufferSlice]]) -> None: ...
def block_attr(attrs: Mapping[str, Object]) -> None: ...

"""
special_stmt - Axis
"""

class axis:
    @staticmethod
    def spatial(dom: Union[PrimExpr, Tuple[PrimExpr, PrimExpr]], value: PrimExpr) -> IterVar: ...
    @staticmethod
    def S(dom: Union[PrimExpr, Tuple[PrimExpr, PrimExpr]], value: PrimExpr) -> IterVar: ...
    @staticmethod
    def reduce(dom: Union[PrimExpr, Tuple[PrimExpr, PrimExpr]], value: PrimExpr) -> IterVar: ...
    @staticmethod
    def R(dom: Union[PrimExpr, Tuple[PrimExpr, PrimExpr]], value: PrimExpr) -> IterVar: ...
    @staticmethod
    def scan(dom: Union[PrimExpr, Tuple[PrimExpr, PrimExpr]], value: PrimExpr) -> IterVar: ...
    @staticmethod
    def opaque(dom: Union[PrimExpr, Tuple[PrimExpr, PrimExpr]], value: PrimExpr) -> IterVar: ...
    @staticmethod
    def remap(iter_types: str, loop_vars: List[Var]) -> List[IterVar]: ...

"""
special_stmt - Annotations
"""

def buffer_var(dtype, storage_scope) -> IterVar: ...
def func_attr(attrs: Dict) -> None: ...
def prim_func(input_func: Callable) -> PrimFunc: ...

"""
special_stmt - Threads and Bindings
"""

def env_thread(env_name: str) -> IterVar: ...
def bind(iter_var: IterVar, expr: PrimExpr) -> None: ...

"""
Scope handler
"""

class block(ContextManager):
    def __init__(self, name_hint: str = "") -> None: ...
    def __enter__(self) -> Sequence[IterVar]: ...

class init(ContextManager):
    def __init__(self) -> None: ...

class let(ContextManager):
    def __init__(self, var: Var, value: PrimExpr) -> None: ...

def where(cond: PrimExpr) -> None: ...
def allocate(
    extents, dtype, scope: str, condition: builtins.bool = True, annotations=None
) -> None: ...
def launch_thread(env_var, extent) -> None: ...
def realize(buffer_slice: BufferSlice, scope: str, condition: builtins.bool = True) -> None: ...
def attr(node: PrimExpr, attr_key: str, value: PrimExpr) -> None: ...
def Assert(condition, message): ...

"""
Scope handler - Loops
"""

def serial(
    begin: Union[PrimExpr, int],
    end: Union[PrimExpr, int],
    annotations: Optional[Mapping[str, Object]] = None,
) -> Iterable[IterVar]: ...
def parallel(
    begin: Union[PrimExpr, int],
    end: Union[PrimExpr, int],
    annotations: Optional[Mapping[str, Object]] = None,
) -> Iterable[IterVar]: ...
def vectorized(
    begin: Union[PrimExpr, int],
    end: Union[PrimExpr, int],
    annotations: Optional[Mapping[str, Object]] = None,
) -> Iterable[IterVar]: ...
def unroll(
    begin: Union[PrimExpr, int],
    end: Union[PrimExpr, int],
    annotations: Optional[Mapping[str, Object]] = None,
) -> Iterable[IterVar]: ...
def thread_binding(
    begin: Union[PrimExpr, int],
    end: Union[PrimExpr, int],
    thread: str,
    annotations: Optional[Mapping[str, Object]] = None,
) -> Iterable[IterVar]: ...
def for_range(
    begin: Union[PrimExpr, int],
    end: Union[PrimExpr, int] = None,
    annotations: Optional[Mapping[str, Object]] = None,
) -> Iterable[IterVar]: ...
def grid(*extents: List[Union[PrimExpr, int]]) -> Iterable[Tuple[IterVar]]: ...

"""
ty - redefine types
"""

class boolean: ...
class handle: ...
class Ptr: ...
