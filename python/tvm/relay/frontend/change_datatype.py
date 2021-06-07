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
# pylint: disable=unused-argument
"""Change Datatype Pass"""
from ..function import Function
from ..expr_functor import ExprMutator
from ..transform.transform import function_pass
from ..expr import var, bind


@function_pass(opt_level=0)
class ChangeDatatype(ExprMutator):
    """Mutator for changing the datatype of Relay programs.

    This pass should be useful for users of the Bring Your Own Datatypes
    framework.
    TODO(@gussmith23 @hypercubestart) Add link to documentation when it exists

    Example:

    .. code-block:: python

        from tvm.relay.testing.inception_v3 import get_workload
        mod, params = get_workload()

        def change_dtype(mod, params, src, dst):
            mod = ChangeDatatype(src, dst)(mod)
            params = dict((p, tvm.nd.array(params[p].numpy().astype(dst))) for p in params)
            return mod, params

        mod, params = change_dtype(mod, params, "float32", "custom[posites2]32")

    Parameters
    ----------
    src : String
        The source datatype name, e.g. "float" or "posites2" (but not "float32"
        or "custom[posites2]32").
    dst : String
        The destination datatype name, in the same format.

    Returns
    -------
    mod : tvm.IRModule
        Module where all nodes of dtype `src` have been changed to have dtype
        `dst`.
    """

    def __init__(self, src, dst):
        self.src = src
        self.dst = dst
        super().__init__()

    def transform_function(self, func, mod, ctx):
        return self.visit(func)

    def visit_constant(self, const):
        if const.data.dtype == self.src:
            return const.astype(self.dst)
        return const

    def visit_function(self, fn):
        new_params = []
        binds = {}

        for param in fn.params:
            # Get the parameter's type annotation.
            var_type = param.type_annotation

            # See if we want to replace dtype.
            if var_type.dtype == self.src:
                dtype = self.dst
            else:
                dtype = var_type.dtype

            # Generate new variable.
            new_param = var(param.name_hint, shape=var_type.shape, dtype=dtype)

            new_params.append(new_param)
            binds[param] = new_param

        new_body = self.visit(fn.body)
        # Rewrite the body to use new parameters.
        new_body = bind(new_body, binds)

        # Construct the updated function and return.
        return Function(
            new_params,
            new_body,
            # You could change the return type, if you use None it will re-infer.
            None,
            type_params=fn.type_params,
            attrs=fn.attrs,
        )
