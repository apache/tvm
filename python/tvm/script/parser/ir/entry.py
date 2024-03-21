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
from typing import Optional, Type

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
        m = parse(mod, utils.inspect_class_capture(mod), check_well_formed=check_well_formed)
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


setattr(ir_module, "dispatch_token", "ir")
