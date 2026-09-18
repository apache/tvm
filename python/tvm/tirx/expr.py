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
"""TIR expression nodes.

Each expression node have subfields that can be visited from python side.
For example, you can use addexp.a to get the left operand of an Add node.

.. code-block:: python

  x = tvm.tirx.Var("n", "int32")
  y = x + 2
  assert(isinstance(y, tvm.tirx.Add))
  assert(y.a == x)
"""

import functools

import tvm_ffi

from tvm import ir
from tvm.ir import Expr
from tvm.ir._overload_prim_expr import (  # noqa: F401
    EqualOp,
    ExprOp,
    NotEqualOp,
    div_ambiguity_error,
)
from tvm.ir.base import Span
from tvm.ir.prim import convert as convert

# Retain historical imports as aliases of the canonical shared definitions.
from tvm.ir.prim.expr import (  # noqa: F401
    EQ,
    GE,
    GT,
    LE,
    LT,
    NE,
    Add,
    And,
    BinaryOpExpr,
    Broadcast,
    Cast,
    CmpExpr,
    Div,
    FloatImm,
    FloorDiv,
    FloorMod,
    IntImm,
    Let,
    LogicalExpr,
    Max,
    Min,
    Mod,
    Mul,
    Not,
    Or,
    Ramp,
    Select,
    Shuffle,
    Sub,
)
from tvm.runtime import Object, ObjectConvertible, Scriptable
from tvm.runtime import const as const

from . import _ffi_api
from .buffer import Buffer


@functools.cache
def _get_te_constructor(name: str):
    """Get a TE constructor lazily, while caching it across constructions."""
    return tvm_ffi.get_global_func(f"te.{name}")


class IntImmEnum(ObjectConvertible):
    """Lazily evaluate an IntImm in case
    the constructor is not available in runtime.

    Parameters
    ----------
    value : int
        The enum value

    span : Optional[Span]
        The location of the cast in the source.
    """

    def __init__(self, value: int, span: Span | None = None) -> None:
        self.value = value
        self.span = span

    def asobject(self) -> "IntImm":
        """Convert object."""
        return IntImm("int32", self.value, self.span)  # type: ignore


Var = ir.Var


@tvm_ffi.register_object("tirx.IterVar")
class IterVar(ExprOp, Object, Scriptable):
    """Represent iteration variable.

    IterVar represents axis iterations in the computation.

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


@tvm_ffi.register_object("te.CommReducer")
class CommReducer(Object, Scriptable):
    """Commutative reduce operator

    Parameters
    ----------
    lhs : List[Var]
       The left arguments of the reducer.

    rhs : List[Var]
       The right arguments of the reducer.

    result : List[Expr]
       The reduction results.

    identity_element : List[Expr]
       The identity elements.

    span : Optional[Span]
        The location of this expression in the source code.
    """

    lhs: list[Var]
    rhs: list[Var]
    result: list[Expr]
    identity_element: list[Expr]

    def __init__(
        self,
        lhs: list[Var],
        rhs: list[Var],
        result: list[Expr],
        identity_element: list[Expr],
        span: Span | None = None,
    ) -> None:
        self.__init_handle_by_constructor__(
            _get_te_constructor("CommReducer"),
            lhs,
            rhs,
            result,
            identity_element,
            span,  # type: ignore
        )


@tvm_ffi.register_object("te.Reduce")
class Reduce(ir.ExprWithOp):
    """Reduce node.

    Parameters
    ----------
    combiner : CommReducer
        The combiner.

    src : list of Expr
        The source expression.

    rdom : list of IterVar
        The iteration domain

    condition : Expr
        The reduce condition.

    value_index : int
        The value index.

    init : list of Expr
        The initial value for output. This can be an int, float, or TE tensor-load Call.

    span : Optional[Span]
        The location of this expression in the source code.
    """

    combiner: CommReducer
    source: list[Expr]
    init: list[Expr]
    axis: list[IterVar]
    condition: Expr
    value_index: int

    def __init__(
        self,
        combiner: CommReducer,
        src: list[Expr],
        rdom: list[IterVar],
        condition: Expr,
        value_index: int,
        init: list[Expr] | None = None,
        span: Span | None = None,
    ) -> None:
        init = [] if init is None else init
        self.__init_handle_by_constructor__(
            _get_te_constructor("Reduce"),
            combiner,
            src,
            rdom,
            condition,
            value_index,
            init,
            span,  # type: ignore
        )


def BufferLoad(buffer: Buffer, indices: list[Expr], span: Span | None = None) -> ir.TensorLoad:
    """Construct a validated buffer load.

    Parameters
    ----------
    buffer : Buffer
        The buffer to be loaded.

    indices : List[Expr]
        The buffer indices to load values from.

    span : Optional[Span]
        The location of this expression in the source code.

    """

    return _ffi_api.BufferLoad(buffer, indices, span)


class CallEffectKind:
    """Possible kinds of Call effects."""

    # only expose up to opaque
    ExprAnnotation = IntImmEnum(0)
    Pure = IntImmEnum(1)
    ReadState = IntImmEnum(2)
    UpdateState = IntImmEnum(3)
    Opaque = UpdateState
