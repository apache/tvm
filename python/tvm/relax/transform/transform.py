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
# pylint: disable=invalid-name
"""Relax transformation passes."""
import functools
import inspect
import types
from typing import Callable, Union

import tvm.ir
from . import _ffi_api


@tvm._ffi.register_object("relax.FunctionPass")
class FunctionPass(tvm.ir.transform.Pass):
    """A pass that works on each tvm.relax.Function in a module. A function
    pass class should be created through `function_pass`.
    """


@tvm._ffi.register_object("relax.DataflowBlockPass")
class DataflowBlockPass(tvm.ir.transform.Pass):
    """A pass that works on each tvm.relax.DataflowBlock in a module."""


def ToNonDataflow() -> tvm.ir.transform.Pass:
    """Transform all dataflow structure to non-dataflow version.

    Returns
    -------
    ret: tvm.ir.transform.Pass
    """
    return _ffi_api.ToNonDataflow()  # type: ignore


def CallTIRRewrite() -> tvm.ir.transform.Pass:
    """Perform explicit tensor allocation for call_tir.

    Returns
    -------
    ret: tvm.ir.transform.Pass
    """
    return _ffi_api.CallTIRRewrite()  # type: ignore


def RewriteDataflowReshape() -> tvm.ir.transform.Pass:
    """Convert all reshape-like call_tir to VM reshape operator call.
    The VM reshape operator calls will be further lowered to a CreateView
    operation at runtime, instead of doing real data copy.
    Here "reshape-like" includes reshape, expand_dims, flatten, etc.

    Returns
    -------
    ret : tvm.ir.transform.Pass
    """
    return _ffi_api.RewriteDataflowReshape()  # type: ignore


def VMBuiltinLower() -> tvm.ir.transform.Pass:
    """Lowering generic intrinsic to VM intrinsics.

    Returns
    -------
    ret: tvm.ir.transform.Pass
    """
    return _ffi_api.VMBuiltinLower()  # type: ignore


def VMShapeLower(*, emit_err_ctx: bool = True) -> tvm.ir.transform.Pass:
    """Lower the symbolic shape and argument and match-cast structinfo matching.

    Parameters
    ----------
    emit_err_ctx: Optional[bool]
        Whether emit err context string, can be turned off for testing purposes.

    Returns
    -------
    ret: tvm.ir.transform.Pass
    """
    return _ffi_api.VMShapeLower(emit_err_ctx)  # type: ignore


def AttachGlobalSymbol() -> tvm.ir.transform.Pass:
    """Attach global_symbol to Relax functions and TIR Primfuncs for codegen.

    Returns
    -------
    ret: tvm.ir.transform.Pass
    """
    return _ffi_api.AttachGlobalSymbol()  # type: ignore


def _wrap_class_function_pass(pass_cls, pass_info):
    """Wrap a python class as function pass."""

    class PyFunctionPass(FunctionPass):
        """Internal wrapper class to create a class instance."""

        def __init__(self, *args, **kwargs):
            # initialize handle in case pass_cls creation failed.
            self.handle = None
            inst = pass_cls(*args, **kwargs)

            # it is important not to capture self to
            # avoid a cyclic dependency
            def _pass_func(func, mod, ctx):
                return inst.transform_function(func, mod, ctx)

            self.__init_handle_by_constructor__(
                _ffi_api.MakeFunctionPass, _pass_func, pass_info  # type: ignore
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
    opt_level=None,
    name=None,
    required=None,
) -> Union[Callable, FunctionPass]:
    """Decorate a function pass.

    This function returns a callback when pass_func
    is provided. Otherwise, it returns the created function pass using the
    given optimization function.

    Parameters
    ----------
    pass_func : Optional[Callable[(Function, Module, PassContext) -> Function]]
        The transformation function or class.

    opt_level : int
        The optimization level of this function pass.

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

        @relax.transform.function_pass(opt_level=1)
        class TestReplaceFunc:
            def __init__(self, new_func):
                self.new_func = new_func

            def transform_function(self, func, mod, ctx):
                # just for demo purposes
                # transform func to new_func
                return self.new_func

        @R.function
        def f1(x: Tensor[(m, n), "float32"]):
            return x

        @tvm.script.ir_module
        class InputMod:
            @R.function
            def f2(x: Tensor[(m, n), "float32"]):
                gv0 = relax.add(x, x)
                return gv0
        # fpass is now a special pass that replaces every
        # function to f1
        fpass = TestReplaceFunc(f1)
        # now every function in InputMod is replaced by f1
        updated_mod = fpass(InputMod)


    The following code creates a function pass by decorating
    a user defined transform function.

    .. code-block:: python

        @relax.transform.function_pass(opt_level=2)
        def transform(func, mod, ctx):
            # my transformations here.
            return func

        function_pass = transform
        assert isinstance(function_pass, relax.transform.FunctionPass)
        assert function_pass.info.opt_level == 2

        # Given a module m, the optimization could be invoked as the follwoing:
        updated_mod = function_pass(m)
        # Now transform should have been applied to every function in
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
        info = tvm.transform.PassInfo(opt_level, fname, required)
        if inspect.isclass(pass_arg):
            return _wrap_class_function_pass(pass_arg, info)
        if not isinstance(pass_arg, (types.FunctionType, types.LambdaType)):
            raise TypeError("pass_func must be a callable for Function pass")
        return _ffi_api.MakeFunctionPass(pass_arg, info)  # type: ignore

    if pass_func:
        return create_function_pass(pass_func)
    return create_function_pass


def _wrap_class_dataflowblock_pass(pass_cls, pass_info):
    """Wrap a python class as dataflowblock pass"""

    class PyDataflowBlockPass(DataflowBlockPass):
        """Internal wrapper class to create a class instance."""

        def __init__(self, *args, **kwargs):
            # initialize handle in case pass_cls creation failed.
            self.handle = None
            inst = pass_cls(*args, **kwargs)

            # it is important not to capture self to
            # avoid a cyclic dependency
            def _pass_func(func, mod, ctx):
                return inst.transform_dataflowblock(func, mod, ctx)

            self.__init_handle_by_constructor__(
                _ffi_api.MakeDataflowBlockPass, _pass_func, pass_info  # type: ignore
            )
            self._inst = inst

        def __getattr__(self, name):
            # fall back to instance attribute if there is not any
            return self._inst.__getattribute__(name)

    functools.update_wrapper(PyDataflowBlockPass.__init__, pass_cls.__init__)
    PyDataflowBlockPass.__name__ = pass_cls.__name__
    PyDataflowBlockPass.__doc__ = pass_cls.__doc__
    PyDataflowBlockPass.__module__ = pass_cls.__module__
    return PyDataflowBlockPass


def dataflowblock_pass(
    pass_func=None, opt_level=None, name=None, required=None
) -> Union[Callable, DataflowBlockPass]:
    """Decorate a dataflowblock pass.

    This function returns a callback when pass_func
    is provided. Otherwise, it returns the created dataflowblock pass using the
    given optimization function.

    Parameters
    ----------
    pass_func : Optional[Callable[(DataflowBlock, Module, PassContext) -> DataflowBlock]]
        The transformation function or class.

    opt_level : int
        The optimization level of this dataflowblock pass.

    name : Optional[str]
        The name of the dataflowblock pass. The name could be empty. In this case, the
        name of the optimization function will be used as the pass name.

    required : Optional[List[str]]
        The list of passes that the dataflowblock pass is dependent on.

    Returns
    -------
    create_dataflowblock_pass : Union[Callable, DataflowBlockPass]

        A decorator will be returned if pass_func is not provided,
        otherwise return the decorated result.
        The returned decorator has two behaviors depending on the input:
        A new DataflowBlockPass will be returned when we decorate a pass function.
        A new DataflowBlockPass class will be returned when we decorate a class type.

    Examples
    --------
    The following code block decorates a dataflowblock pass class.

    .. code-block:: python

        @relax.transform.dataflowblock_pass(opt_level=1)
        class TestReplaceBinding:
            # Simple test function to replace the first VarBinding to another.

            def __init__(self):
                # create a new VarBinding
                m, n = tir.Var("m", "int64"), tir.Var("n", "int64")
                lv0 = relax.Var("lv1", relax.TensorStructInfo([m, n], "float32"))
                val = relax.const(np.random.rand(24, 56))
                self.new_binding = relax.VarBinding(lv0, val)

            def transform_dataflowblock(self, block, mod, ctx):
                # just for demo purposes
                # Replace the first binding in the DataflowBlock
                new_bindings = [self.new_binding, block.bindings[1]]
                new_block = relax.expr.DataflowBlock(new_bindings, block.span)
                return new_block

        @tvm.script.ir_module
        class InputMod:
            @R.function
            def f1(x: Tensor[(m, n), "float32"]):
                with relax.dataflow():
                    lv0 = relax.multiply(x, x)
                    gv0 = relax.add(x, x)
                    relax.output(gv0)
                return gv0
        # block_pass is now a special pass that replaces every
        # first binding to the constant value binding
        block_pass = TestReplaceBinding()
        # now every first binding in DataflowBlock of InputMod
        # is replaced by new_binding
        updated_mod = block_pass(InputMod)


    The following code creates a dataflowblock pass by decorating
    a user defined transform function.

    .. code-block:: python

        @relax.transform.dataflowblock_pass(opt_level=2)
        def transform(block, mod, ctx):
            # my transformations here.
            return block

        block_pass = transform
        assert isinstance(block_pass, relax.transform.DataflowBlockPass)
        assert block_pass.info.opt_level == 2

        # Given a module m, the optimization could be invoked as the follwoing:
        updated_mod = block_pass(m)
        # Now transform should have been applied to every DataflowBlock in
        # the provided module m. And the updated module will be returned.
    """

    if opt_level is None:
        raise ValueError("Please provide opt_level for the dataflowblock pass.")

    required = required if required else []
    if not isinstance(required, (list, tuple)):
        raise TypeError("Required is expected to be the type of " + "list/tuple.")

    def create_dataflowblock_pass(pass_arg):
        """Internal function that creates a dataflowblock pass"""
        fname = name if name else pass_arg.__name__
        info = tvm.transform.PassInfo(opt_level, fname, required)
        if inspect.isclass(pass_arg):
            return _wrap_class_dataflowblock_pass(pass_arg, info)
        if not isinstance(pass_arg, (types.FunctionType, types.LambdaType)):
            raise TypeError("pass_func must be a callable for DataflowBlock pass")
        return _ffi_api.MakeDataflowBlockPass(pass_arg, info)  # type: ignore

    if pass_func:
        return create_dataflowblock_pass(pass_func)
    return create_dataflowblock_pass
