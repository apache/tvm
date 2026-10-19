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

import functools
import inspect
from collections.abc import Callable

import tvm_ffi

from tvm_tirx.ir.transform import Pass, PassInfo

from . import _ffi_api


@tvm_ffi.register_object("tirx.FunctionPass")
class FunctionPass(Pass):
    """A pass that transforms each :py:class:`tvm_tirx.tirx.Function` in a module.

    Create instances or subclasses with :py:func:`function_pass`, which supplies
    pass metadata and adapts a Python callback to the pass manager.
    """


def _wrap_class_function_pass(pass_cls, pass_info):
    """Wrap a python class as function pass"""

    class PyFunctionPass(FunctionPass):
        """Internal wrapper class to create a class instance."""

        def __init__(self, *args, **kwargs):
            inst = pass_cls(*args, **kwargs)

            # it is important not to capture self to
            # avoid a cyclic dependency
            def _pass_func(func, mod, ctx):
                return inst.transform_function(func, mod, ctx)

            self.__init_handle_by_constructor__(
                _ffi_api.CreateFunctionPass,
                _pass_func,
                pass_info,  # type: ignore
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


def function_pass(
    pass_func=None,
    opt_level: int | None = None,
    name: str | None = None,
) -> Callable | FunctionPass:
    """Create a TIRx function pass from a callback or class.

    The callback receives ``(function, module, context)`` and returns the
    transformed Function. A decorated class instead implements
    ``transform_function(self, function, module, context)``. With no
    ``pass_func``, returns a decorator for either form.

    Parameters
    ----------
    pass_func : Callable or type, optional
        Transformation callback, or class implementing ``transform_function``.
    opt_level : int
        Required optimization level for the pass.
    name : str, optional
        Pass name. Defaults to the callback or class name.

    Returns
    -------
    result : FunctionPass, type, or Callable
        A pass instance for a callback, a pass class for a class, or a decorator
        when ``pass_func`` is omitted.

    Examples
    --------
    Create and invoke a pass that adds a function attribute::

        @tvm_tirx.tirx.transform.function_pass(opt_level=0)
        def mark(func, mod, ctx):
            return func.with_attr("example", 1)

        after = mark(before)  # before is an IRModule

    TVMScript body fragments for a function in the module:

    .. code-block:: python

        # Before
        T.evaluate(0)
        # After
        T.func_attr({"example": 1})
        T.evaluate(0)

    The class form allows pass parameters::

        @tvm_tirx.tirx.transform.function_pass(opt_level=0)
        class Mark:
            def __init__(self, value):
                self.value = value

            def transform_function(self, func, mod, ctx):
                return func.with_attr("example", self.value)

        after = Mark(1)(before)
    """

    if opt_level is None:
        raise ValueError("Please provide opt_level for the function pass.")

    def create_function_pass(pass_arg):
        """Internal function that creates a function pass"""
        fname = name if name else pass_arg.__name__
        info = PassInfo(opt_level, fname)
        if inspect.isclass(pass_arg):
            return _wrap_class_function_pass(pass_arg, info)
        if not callable(pass_arg):
            raise TypeError("pass_func must be a callable for Module pass")
        return _ffi_api.CreateFunctionPass(pass_arg, info)  # type: ignore

    if pass_func:
        return create_function_pass(pass_func)
    return create_function_pass
