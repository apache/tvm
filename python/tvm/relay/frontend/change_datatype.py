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

from ..function import Function
from ..expr_functor import ExprMutator
from ..transform.transform import function_pass
from ..expr import var, bind

# TODO(@gussmith23) what's the right opt level here?
@function_pass(opt_level=0)
class ChangeDatatype(ExprMutator):
    """Mutator for changing the datatype of Relay programs.

    Example usage:
    ```python
    from tvm.relay.testing.inception_v3 import get_workload
    expr, params = get_workload()

    def change_dtype(src, dst, expr, params):
        cdtype = ChangeDatatype(src, dst)
        expr = cdtype.visit(expr)
        expr = relay.ir_pass.infer_type(expr)
        params = dict((p, tvm.nd.array(params[p].asnumpy().astype(dst))) for p in params)
        return expr, params
    ```
    """

    def __init__(self, src, dst):
        self.src = src
        self.dst = dst
        super().__init__()

    def transform_function(self, func, mod, ctx):
        return self.visit(func)

    def visit_constant(self, const):
        raise "need to rewrite embedded constants"

    def visit_function(self, func):
        new_params = []
        binds = {}

        for param in func.params:
            # Get the parameter's type annotation.
            var_type = param.type_annotation

            # See if we want to replace dtype.
            if var_type.dtype == self.src:
                dtype = self.dst
            else:
                dtype = var_type.dtype

            # Generate new variable.
            new_param = var(
                param.name_hint, shape=var_type.shape, dtype=dtype)

            new_params.append(new_param)
            binds[param] = new_param

        new_body = self.visit(func.body)
        # Rewrite the body to use new parameters.
        new_body = bind(new_body, binds)

        # Construct the updated function and return.
        return Function(
            new_params,
            new_body,
            # You could change the return type, if you use None it will re-infer.
            None,
            type_params=func.type_params,
            attrs=func.attrs)
