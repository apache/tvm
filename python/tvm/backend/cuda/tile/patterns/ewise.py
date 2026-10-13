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

"""One explicitly supplied scalar TIRx function per register element.

The public function is converted to the existing LambdaExpr staging node at
construction. Only typed IR survives serialization, and lowering substitutes
that expression directly into each register store.
"""

import tvm_ffi

from tvm.ir import Bind, If, LambdaExpr, PrimType, Return, TensorRegion, TensorStore, Tuple, Var
from tvm.script import tirx as T
from tvm.tirx import Function
from tvm.tirx.tile.instruction import Instruction, Operand

from .common import Context


def _scalar_body(statements, env):
    def substitute(expr):
        return tvm_ffi.structural_map(expr, (Var, lambda v: env.get(v, v)), order="pre")

    for statement in statements.seq:
        if isinstance(statement, Bind) and isinstance(statement.var.ty, PrimType):
            env[statement.var] = substitute(statement.value)
        elif isinstance(statement, Return):
            return substitute(statement.value)
        elif isinstance(statement, If) and statement.else_case is not None:
            return T.if_then_else(
                substitute(statement.cond),
                _scalar_body(statement.then_case, dict(env)),
                _scalar_body(statement.else_case, dict(env)),
            )
        else:
            raise ValueError(
                "ewise function requires scalar bindings and a scalar return on every path"
            )
    raise ValueError("ewise scalar function must return a value")


def _as_lambda(function):
    if isinstance(function, LambdaExpr):
        return function
    if not isinstance(function, Function):
        raise TypeError("ewise requires a scalar TIRx Function")
    if not all(isinstance(p.ty, PrimType) for p in function.params):
        raise TypeError("ewise function parameters must be scalars")

    def expression(*values):
        return _scalar_body(function.body, dict(zip(function.params, values)))

    return LambdaExpr([p.ty for p in function.params], expression, ret_type=function.ret_type)


def _lower(view, sctx):
    function, dst, packed = view.args
    inputs = list(packed.fields) if packed is not None else []
    if len(inputs) != len(function.vars):
        raise ValueError("ewise input count must match the scalar function parameters")
    ctx = Context(sctx)
    out = ctx.region("dst", dst, (), ("local",))
    out.owner(())
    out.nonoverlapping()
    target = out.value(())
    values = []
    for i, (param, value) in enumerate(zip(function.vars, inputs)):
        if isinstance(value, TensorRegion):
            operand = ctx.region(f"src{i}", value, (), ("local",))
            operand.owner(())
            value = operand.value(())
        if value.ty != param.ty:
            raise TypeError(f"ewise input {i}: dtype must match the scalar function parameter")
        values.append(value)
    result = function.apply(values)
    if result.ty != target.ty:
        raise TypeError("ewise return dtype must match the destination dtype")
    return ctx.finish([TensorStore(target.source, target.indices, result)])


_construct = Instruction(
    "tirx.cuda.tile.ewise",
    "cuda_ewise",
    (Operand("function"), Operand("dst", "region"), Operand("inputs", default=None)),
    ("thread", "warp", "warpgroup", "cta", "cluster"),
    lower=_lower,
).register()


def ewise(function, dst, *inputs, scope="thread", **kwargs):
    """Inline a scalar TIRx function at every explicit register coordinate.

    Tensor inputs have exactly the destination's group and repeat axes. Scalar
    inputs are broadcast; tensor broadcasting is expressed with a zero-stride
    view. There is no implicit cast, vectorization, memory staging, or barrier.
    """
    if len(inputs) == 1 and isinstance(inputs[0], Tuple | tuple | list):
        values = inputs[0]
        inputs = tuple(values.fields if isinstance(values, Tuple) else values)
    return _construct(_as_lambda(function), dst, inputs, scope=scope, **kwargs)
