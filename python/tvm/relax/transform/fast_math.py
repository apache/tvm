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
# pylint: disable=invalid-name, unused-argument, redefined-argument-from-local
"""Relax Use Fast Math pass."""
import tvm
from tvm import topi, tir
from tvm.ir.module import IRModule
from tvm.relax import Expr, Call, expr_functor, PyExprMutator, expr


@expr_functor.mutator
class FastMathCodeGenerator(PyExprMutator):
    """
    Converts the expensive non linear functions to their fast but approximate counterparts.

    Parameters
    ----------
    mod: IRModule
        The module to be transformed
    """

    def __init__(self, mod):
        super().__init__(mod)

    def visit_call_(self, call: Call) -> Expr:
        if call.op.name == "relax.nn.softmax":
            return self.builder_.call_te(topi.nn.fast_softmax, call.args[0], call.attrs.axis)
        if call.op.name == "relax.exp":
            return self.builder_.call_te(topi.fast_exp, call.args[0])
        if call.op.name == "relax.erf":
            return self.builder_.call_te(topi.fast_erf, call.args[0])
        if call.op.name == "relax.tanh":
            return self.builder_.call_te(topi.fast_tanh, call.args[0])
        if call.op.name == "relax.power":

            def te_fast_power_const(x, y):
                """power(x, y) = exp(log(x) * y) when x > 0 or y is not an integer."""
                return topi.fast_exp(topi.multiply(topi.log(x), y))

            def te_fast_power(x, y):
                """
                power(x, y) = exp(log(abs(x)) * y) * (log((x / abs(x) - 1) *
                (ceil(y) - floor(y)) + 1) + (x / abs(x) - 1) * (y % 2) + 1).
                When x < 0 and y is not an integer, power(x, y) =
                log((x / abs(x) - 1) * (ceil(y) - floor(y)) + 1) = nan,
                otherwise log((x / abs(x) - 1) * (ceil(y) - floor(y)) + 1) =
                log(1) = 0. When x < 0 and y is an integer, the sign of
                power(x, y) = (x / abs(x) - 1) * (y % 2) + 1.
                """
                dtype = x.dtype
                return topi.multiply(
                    topi.add(
                        topi.add(
                            topi.log(
                                topi.add(
                                    topi.multiply(
                                        topi.subtract(
                                            topi.divide(x, topi.abs(x)), tir.const(1.0, dtype)
                                        ),
                                        topi.ceil(y) - topi.floor(y),
                                    ),
                                    tir.const(1.0, dtype),
                                )
                            ),
                            topi.multiply(
                                topi.subtract(topi.divide(x, topi.abs(x)), tir.const(1.0, dtype)),
                                topi.mod(y, tir.const(2.0, dtype)),
                            ),
                        ),
                        tir.const(1.0, dtype),
                    ),
                    topi.fast_exp(topi.multiply(topi.log(topi.abs(x)), y)),
                )

            if isinstance(call.args[1], expr.Constant):
                if call.args[1].data.numpy().tolist().is_integer():
                    return self.builder_.call_te(topi.power, call.args[0], call.args[1])
                return self.builder_.call_te(te_fast_power_const, call.args[0], call.args[1])
            return self.builder_.call_te(te_fast_power, call.args[0], call.args[1])

        return super().visit_call_(call)


@tvm.transform.module_pass(opt_level=0, name="FastMathTransform")
class FastMathTransform:
    """
    Pass to convert the expensive non linear functions to their fast but approximate counterparts.
    """

    def transform_module(self, mod: IRModule, ctx: tvm.transform.PassContext) -> IRModule:
        fast_math_codegen = FastMathCodeGenerator(mod)
        for gv in mod.functions:
            func = mod[gv]
            if not isinstance(func, tvm.relax.Function):
                continue
            func = fast_math_codegen.visit_expr(func)
            fast_math_codegen.builder_.update_func(gv, func)

        return fast_math_codegen.builder_.get()
