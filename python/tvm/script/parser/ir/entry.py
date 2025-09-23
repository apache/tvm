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
"""The entry point of TVM parser for ir module."""

import inspect
from typing import Callable, Optional, Type

from tvm.ir import IRModule, GlobalVar
from tvm.relax.expr import ExternFunc
from tvm.relax.base_py_module import BasePyModule
from tvm import cpu, ir

from .._core import parse, utils


# this formulation allows us to support having @I.ir_module
# appear as a decorator by itself or to have optional arguments
# like @I.ir_module(check_well_formed=False)
def ir_module(mod: Optional[Type] = None, check_well_formed: bool = True) -> IRModule:
    """The parsing method for ir module, by using `@ir_module` as decorator.

    Parameters
    ----------
    mod : Type
        The class to be parsed as ir module.

    check_well_formed : bool
        Whether to check well-formedness during parsing.

    Returns
    -------
    ir_module : IRModule
        The parsed ir module.
    """

    def decorator_wrapper(mod):
        if not inspect.isclass(mod):
            raise TypeError(f"Expect a class, but got: {mod}")

        # Check BasePyModule inheritance
        base_py_module_inherited = any(base.__name__ == "BasePyModule" for base in mod.__bases__)

        m = parse(mod, utils.inspect_class_capture(mod), check_well_formed=check_well_formed)

        if base_py_module_inherited:
            # Collect pyfunc methods
            pyfunc_methods = [
                name
                for name, attr in mod.__dict__.items()
                if hasattr(attr, "dispatch_token") and attr.dispatch_token == "pyfunc"
            ]

            mod._pyfunc_methods = pyfunc_methods

            # Create ExternFunc nodes

            for method_name in pyfunc_methods:
                try:
                    existing_gvars = [
                        global_var
                        for global_var in m.get_global_vars()
                        if global_var.name_hint == method_name
                    ]

                    extern_func = ExternFunc(method_name)
                    extern_func = extern_func.with_attr("is_pyfunc", True)
                    extern_func = extern_func.with_attr("function_type", "python")
                    extern_func = extern_func.with_attr("python_function_name", method_name)
                    extern_func = extern_func.with_attr(
                        "python_source", f"# Source for {method_name}"
                    )
                    extern_func = extern_func.with_attr("python_packed_func", None)

                    if existing_gvars:
                        m[existing_gvars[0]] = extern_func
                    else:
                        m[GlobalVar(method_name)] = extern_func

                except Exception:  # pylint: disable=broad-exception-caught
                    continue

            class ModuleFactory:
                """Factory class for creating BasePyModule instances with Python functions."""

                def __init__(self, module, pyfunc_methods, original_class):
                    self.ir_module = module
                    self.pyfunc_methods = pyfunc_methods
                    self.original_class = original_class

                def __call__(self, device=None, target=None):

                    if device is None:
                        device = cpu(0)

                    instance_ir_mod = ir.IRModule()
                    for global_var, func in self.ir_module.functions_items():
                        instance_ir_mod[global_var] = func

                    instance = BasePyModule(instance_ir_mod, device, target)

                    for method_name in self.pyfunc_methods:
                        if hasattr(self.original_class, method_name):
                            method = getattr(self.original_class, method_name)
                            instance.add_python_function(method_name, method)

                    return instance

                def __getattr__(self, name):
                    if hasattr(self.ir_module, name):
                        return getattr(self.ir_module, name)
                    raise AttributeError(
                        f"'{self.__class__.__name__}' object has no attribute '{name}'"
                    )

            factory = ModuleFactory(m, pyfunc_methods, mod)
            setattr(factory, "__name__", mod.__name__)
            return factory

        setattr(m, "__name__", mod.__name__)
        return m

    if mod is not None:
        # if there are no optional args given, this will directly invoke the wrapper
        return decorator_wrapper(mod)
    else:
        # if there is a optional arg given, it returns the wrapper function
        # as a new decorator and applies it
        setattr(decorator_wrapper, "dispatch_token", "ir")
        return decorator_wrapper


def pyfunc(func: Callable):
    # Set the dispatch_token on the decorated function
    setattr(func, "dispatch_token", "pyfunc")
    return func


setattr(pyfunc, "dispatch_token", "pyfunc")
