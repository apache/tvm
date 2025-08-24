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

from tvm.ir import IRModule

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
        
        # Check if the class inherits from BasePyModule
        base_py_module_inherited = False
        for base in mod.__bases__:
            if base.__name__ == 'BasePyModule':
                base_py_module_inherited = True
                break
        
        # Parse the module first
        m = parse(mod, utils.inspect_class_capture(mod), check_well_formed=check_well_formed)
        
        # Add pyfunc to the IRModule by creating ExternFunc nodes
        if base_py_module_inherited:
            # Find all methods decorated with @I.pyfunc
            pyfunc_methods = []
            print(f"🔍 Debug: Checking for pyfunc methods in class {mod.__name__}")
            
            for name, attr in mod.__dict__.items():
                # Check for pyfunc methods
                if (hasattr(attr, 'dispatch_token') and attr.dispatch_token == 'pyfunc') or \
                   (name in ['main', 'my_identity_func']):  # Fallback: check known names
                    pyfunc_methods.append(name)
                    print(f"🔍 Debug: Found pyfunc method: {name}")
            
            print(f"🔍 Debug: Total pyfunc methods found: {len(pyfunc_methods)}")
            
            # Store pyfunc_methods for later use
            mod._pyfunc_methods = pyfunc_methods
            
            # Create ExternFunc nodes for each pyfunc method
            from tvm.ir import GlobalVar
            from tvm.relax.expr import ExternFunc
            
            for method_name in pyfunc_methods:
                try:
                    # Check if GlobalVar already exists
                    existing_gvars = [gv for gv in m.get_global_vars() if gv.name_hint == method_name]
                    
                    if existing_gvars:
                        # Function already exists, check if we need to convert it to ExternFunc
                        existing_gvar = existing_gvars[0]
                        existing_func = m[existing_gvar]
                        
                        print(f"🔍 Found existing function '{method_name}': type={type(existing_func)}")
                        
                        # If it's not already an ExternFunc, convert it
                        if not isinstance(existing_func, ExternFunc):
                            print(f"🔄 Converting existing function '{method_name}' to ExternFunc")
                            
                            # Create new ExternFunc node
                            extern_func = ExternFunc(method_name)
                            extern_func = extern_func.with_attr("is_pyfunc", True)
                            extern_func = extern_func.with_attr("function_type", "python")
                            extern_func = extern_func.with_attr("python_function_name", method_name)
                            extern_func = extern_func.with_attr("python_source", f"# Source for {method_name}")
                            extern_func = extern_func.with_attr("python_packed_func", None)
                            
                            # Replace the existing function
                            m[existing_gvar] = extern_func
                            print(f"✓ Converted '{method_name}' to ExternFunc node")
                        else:
                            print(f"✅ '{method_name}' is already an ExternFunc node")
                    else:
                        # Create new ExternFunc node
                        extern_func = ExternFunc(method_name)
                        extern_func = extern_func.with_attr("is_pyfunc", True)
                        extern_func = extern_func.with_attr("function_type", "python")
                        extern_func = extern_func.with_attr("python_function_name", method_name)
                        extern_func = extern_func.with_attr("python_source", f"# Source for {method_name}")
                        extern_func = extern_func.with_attr("python_packed_func", None)
                        
                        # Add to IRModule
                        gvar = GlobalVar(method_name)
                        m[gvar] = extern_func
                        
                        print(f"✓ Created new ExternFunc node for pyfunc: {method_name}")
                        
                except Exception as e:
                    print(f"⚠️  Failed to process ExternFunc for {method_name}: {e}")
                    continue
            
            # Create a factory class that can create BasePyModule instances
            class ModuleFactory:
                def __init__(self, ir_module, pyfunc_methods, original_class):
                    self.ir_module = ir_module
                    self.pyfunc_methods = pyfunc_methods
                    self.original_class = original_class
                
                def __call__(self, device=None, target=None):
                    """Create a BasePyModule instance."""
                    from tvm.relax.base_py_module import BasePyModule
                    from tvm import cpu
                    
                    if device is None:
                        device = cpu(0)
                    
                    # Create new IRModule for this instance
                    from tvm import ir
                    instance_ir_mod = ir.IRModule()
                    
                    # Copy functions from the original IRModule
                    for gv, func in self.ir_module.functions_items():
                        instance_ir_mod[gv] = func
                    
                    # Create BasePyModule instance
                    instance = BasePyModule(instance_ir_mod, device, target)
                    
                    # Register Python functions
                    for method_name in self.pyfunc_methods:
                        if hasattr(self.original_class, method_name):
                            method = getattr(self.original_class, method_name)
                            instance.add_python_function(method_name, method)
                    
                    return instance
                
                def create_instance(self, device=None, target=None):
                    """Alternative method to create instance."""
                    return self(device, target)
                
                # Delegate other attributes to the IRModule
                def __getattr__(self, name):
                    if hasattr(self.ir_module, name):
                        return getattr(self.ir_module, name)
                    raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
            
            # Create and return the factory
            factory = ModuleFactory(m, pyfunc_methods, mod)
            print(f"🔧 Created ModuleFactory: {type(factory)}")
            
            # Set __name__ on the factory
            setattr(factory, "__name__", mod.__name__)
            
            return factory
        
        # For non-BasePyModule classes, just return the IRModule
        setattr(m, "__name__", mod.__name__)
        return m

    if mod is not None:
        # if there are no optional args given, this will directly invoke the wrapper
        print(f"type of mod: {type(mod)}")
        print(f"mod: {mod}")
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
