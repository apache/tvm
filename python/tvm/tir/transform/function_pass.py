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
"""TIR specific function pass support."""
import inspect
import functools
from typing import Callable, List, Optional, Union

import tvm._ffi
from tvm.ir.transform import Pass, PassInfo

from . import _ffi_api


@tvm._ffi.register_object("tir.PrimFuncPass")
class PrimFuncPass(Pass):
    """A pass that works on each :py:func:`tvm.tir.PrimFunc` in a module. A function
    pass class should be created through py:func:`tvm.tir.transform.function_pass`.
    """


def _wrap_class_function_pass(pass_cls, pass_info):
    """Wrap a python class as function pass"""

    class PyFunctionPass(PrimFuncPass):
        """Internal wrapper class to create a class instance."""

        def __init__(self, *args, **kwargs):
            # initialize handle in cass pass_cls creation failed.fg
            self.handle = None
            inst = pass_cls(*args, **kwargs)
            # it is important not to capture self to
            # avoid a cyclic dependency
            def _pass_func(func, mod, ctx):
                return inst.transform_function(func, mod, ctx)

            self.__init_handle_by_constructor__(
                _ffi_api.CreatePrimFuncPass, _pass_func, pass_info  # type: ignore
            )

            self._inst = inst

        def __getattr__(self, name):
            # fall back to instance attribute if there is not any
            return self._inst.__getattribute__(name)

    functools.update_wrapper(PyFunctionPass.__init__, pass_cls.__init__)
    PyFunctionPass.__name__ = pass_cls.__name__
    PyFunctionPass.__doc__ = pass_cls.__doc__
    PyFunctionPass.__module__ = pass_cls.__module__
    return PyFunctionPass


def prim_func_pass(
    pass_func=None,
    opt_level: int = None,
    name: Optional[str] = None,
    required: Optional[List[str]] = None,
    traceable=False,
) -> Union[Callable, PrimFuncPass]:
    """Decorate a function pass.

    This function returns a callback when pass_func
    is provided. Otherwise, it returns the created function pass using the
    given optimization function.

    Parameters
    ----------
    pass_func : Optional[Callable[(tvm.tir.PrimFunc, IRModule, PassContext) -> tvm.tir.PrimFunc]]
        The transformation function or class.

    opt_level : int
        The optimization level of this module pass.

    name : Optional[str]
        The name of the function pass. The name could be empty. In this case, the
        name of the optimization function will be used as the pass name.

    required : Optional[List[str]]
        The list of passes that the function pass is dependent on.

    Returns
    -------
    create_function_pass : Union[Callable, FunctionPass]

        A decorator will be returned if pass_func is not provided,
        otherwise return the decorated result.
        The returned decorator has two behaviors depending on the input:
        A new FunctionPass will be returned when we decorate a pass function.
        A new FunctionPass class will be returned when we decorate a class type.

    Examples
    --------
    The following code block decorates a function pass class.

    .. code-block:: python

        @tvm.tir.transform.prim_func_pass(opt_level=1)
        class TestReplaceFunc:
            def __init__(self, new_func):
                self.new_func = new_func

            def transform_function(self, func, mod, ctx):
                # just for demo purposes
                # transform func to new_func
                return self.new_func

    The following code creates a function pass by decorating
    a user defined transform function.

    .. code-block:: python

        @tvm.tir.transform.prim_func_pass(opt_level=2)
        def transform(func, mod, ctx):
            # my transformations here.
            return func

        function_pass = transform
        assert isinstance(function_pass, transform.FunctionPass)
        assert function_pass.info.opt_level == 2

        # Given a module m, the optimization could be invoked as the following:
        updated_mod = function_pass(m)
        # Now constant folding should have been applied to every function in
        # the provided module m. And the updated module will be returned.
    """

    if opt_level is None:
        raise ValueError("Please provide opt_level for the function pass.")

    required = required if required else []
    if not isinstance(required, (list, tuple)):
        raise TypeError("Required is expected to be the type of " + "list/tuple.")

    def create_function_pass(pass_arg):
        """Internal function that creates a function pass"""
        fname = name if name else pass_arg.__name__
        info = PassInfo(opt_level, fname, required, traceable)
        if inspect.isclass(pass_arg):
            return _wrap_class_function_pass(pass_arg, info)
        if not callable(pass_arg):
            raise TypeError("pass_func must be a callable for Module pass")
        return _ffi_api.CreatePrimFuncPass(pass_arg, info)  # type: ignore

    if pass_func:
        return create_function_pass(pass_func)
    return create_function_pass
