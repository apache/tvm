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
"""Iteration-axis metadata for schedulable tensor computations."""

import tvm_ffi

from tvm import ir
from tvm.ir import Var
from tvm.ir._overload_prim_expr import ExprOp
from tvm.ir.base import Span
from tvm.runtime import Object, Scriptable

from . import _ffi_api


def _axis_value(value):
    return value.var if isinstance(value, IterVar) else value


@tvm_ffi.register_object("s_tir.IterVar")
class IterVar(Object, Scriptable):
    """Represent iteration variable.

    IterVar represents axis iterations in the computation.
    Arithmetic explicitly projects its variable. Use ``.var`` when passing an
    axis to primitive-expression constructors and core operator functions (including
    their TE re-exports). Equality compares metadata identity; use ``.equal`` for
    symbolic value equality.

    Parameters
    ----------
    dom : Range
        The domain of the iteration.

    var : Union[Var, str]
        The internal variable that is used for iteration.

    iter_type : int
        The iteration type.

    thread_tag : str
        The thread type tag.

    span : Optional[Span]
        The location of this expression in the source code.

    See Also
    --------
    te.thread_axis: Create thread axis IterVar.
    te.reduce_axis: Create reduce axis IterVar.
    """

    DataPar = 0
    ThreadIndex = 1
    CommReduce = 2
    Ordered = 3
    Opaque = 4
    Unrolled = 5
    Vectorized = 6
    Parallelized = 7
    Tensorized = 8

    dom: ir.Range
    var: Var
    iter_type: int
    thread_tag: str

    def __init__(
        self,
        dom: ir.Range,
        var: Var | str,
        iter_type: int,
        thread_tag: str = "",
        span: Span | None = None,
    ) -> None:
        if dom is not None:
            if isinstance(dom, list | tuple):
                if len(dom) != 2:
                    raise TypeError("need to be list of ranges")
                dom = ir.Range(dom[0], dom[1])

            if not isinstance(dom, ir.Range):
                raise TypeError("dom need to be Range")

        name = var if var is not None else "iter"
        dtype = "int32" if dom is None else dom.extent.ty
        var = Var(name, ty=dtype, span=span) if not isinstance(var, Var) else var
        if dom is not None:
            assert var.ty == dom.extent.ty, "IterVar's Var type must match its domain's extent type"
        self.__init_handle_by_constructor__(
            _ffi_api.IterVar,
            dom,
            var,
            iter_type,
            thread_tag,
            span,  # type: ignore
        )

    def expr_ty(self) -> ir.PrimType:
        """Compile-time type of the iteration variable."""
        return self.var.ty

    def __add__(self, other):
        return ExprOp.__add__(self.var, _axis_value(other))

    def __radd__(self, other):
        return ExprOp.__radd__(self.var, _axis_value(other))

    def __sub__(self, other):
        return ExprOp.__sub__(self.var, _axis_value(other))

    def __rsub__(self, other):
        return ExprOp.__rsub__(self.var, _axis_value(other))

    def __mul__(self, other):
        return ExprOp.__mul__(self.var, _axis_value(other))

    def __rmul__(self, other):
        return ExprOp.__rmul__(self.var, _axis_value(other))

    def __div__(self, other):
        return ExprOp.__div__(self.var, _axis_value(other))

    def __rdiv__(self, other):
        return ExprOp.__rdiv__(self.var, _axis_value(other))

    def __truediv__(self, other):
        return ExprOp.__truediv__(self.var, _axis_value(other))

    def __rtruediv__(self, other):
        return ExprOp.__rtruediv__(self.var, _axis_value(other))

    def __floordiv__(self, other):
        return ExprOp.__floordiv__(self.var, _axis_value(other))

    def __rfloordiv__(self, other):
        return ExprOp.__rfloordiv__(self.var, _axis_value(other))

    def __mod__(self, other):
        return ExprOp.__mod__(self.var, _axis_value(other))

    def __rmod__(self, other):
        return ExprOp.__rmod__(self.var, _axis_value(other))

    def __lshift__(self, other):
        return ExprOp.__lshift__(self.var, _axis_value(other))

    def __rlshift__(self, other):
        return ExprOp.__rlshift__(self.var, _axis_value(other))

    def __rshift__(self, other):
        return ExprOp.__rshift__(self.var, _axis_value(other))

    def __rrshift__(self, other):
        return ExprOp.__rrshift__(self.var, _axis_value(other))

    def __and__(self, other):
        return ExprOp.__and__(self.var, _axis_value(other))

    def __rand__(self, other):
        return ExprOp.__rand__(self.var, _axis_value(other))

    def __or__(self, other):
        return ExprOp.__or__(self.var, _axis_value(other))

    def __ror__(self, other):
        return ExprOp.__ror__(self.var, _axis_value(other))

    def __xor__(self, other):
        return ExprOp.__xor__(self.var, _axis_value(other))

    def __rxor__(self, other):
        return ExprOp.__rxor__(self.var, _axis_value(other))

    def __lt__(self, other):
        return ExprOp.__lt__(self.var, _axis_value(other))

    def __le__(self, other):
        return ExprOp.__le__(self.var, _axis_value(other))

    def __gt__(self, other):
        return ExprOp.__gt__(self.var, _axis_value(other))

    def __ge__(self, other):
        return ExprOp.__ge__(self.var, _axis_value(other))

    def __neg__(self):
        return ExprOp.__neg__(self.var)

    def __invert__(self):
        return ExprOp.__invert__(self.var)

    def __bool__(self):
        return ExprOp.__bool__(self.var)

    def equal(self, other, span=None):
        """Compare the value of this axis with another primitive value."""
        return self.var.equal(_axis_value(other), span)

    def astype(self, dtype, span=None):
        """Cast the value of this axis."""
        return self.var.astype(dtype, span)
