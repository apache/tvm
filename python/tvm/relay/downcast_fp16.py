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
# pylint: disable=invalid-name,arguments-differ,no-else-return,unused-argument,missing-docstring
from __future__ import absolute_import
from .expr_functor import ExprMutator
from .expr import Function, Call, Let, Var, GlobalVar, Constant
from . import transform as _transform
from . import module as _module
from . import cast

"""
Relay Downcast from Full-precision to Half-precision floating-point Pass
"""

def downcast_fp16(graph):
    """
    Parameters
    ---------
    graph: Function
        The original graph.

    Retruns
    -------
    The graph after dowmcasting to half-precision floating-point.
    """
    class DowncastMutator(ExprMutator):
        def visit_call(self, call):
            new_fn = self.visit(call.op)
            args = [self.visit(arg) for arg in call.args]
            new_args = list()
            for arg in args:
                if isinstance(arg, Var) or isinstance(arg, Constant):
                    new_args.append(cast(arg, dtype='float16'))
                else:
                    new_args.append(arg)
            return Call(new_fn, new_args, call.attrs)

        def visit_function(self, fn):
            new_params = [self.visit(x) for x in fn.params]
            new_body = cast(self.visit(fn.body), dtype='float32')
            return Function(
                            list(new_params),
                            new_body,
                            fn.ret_type,
                            fn.type_params,
                            fn.attrs)
    def infer_type(expr):
        mod = _module.Module.from_expr(expr)
        mod = _transform.InferType()(mod)
        entry = mod["main"]
        return entry if isinstance(expr, Function) else entry.body

    downcast_pass = DowncastMutator()
    func = downcast_pass.visit(graph)
    func = infer_type(func)
    return func

