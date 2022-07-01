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
# pylint: disable=no-else-return
"""The Python interface to the Relay reference interpreter."""
from __future__ import absolute_import

import numpy as np

import tvm._ffi
from tvm.runtime import container, Object

from . import _backend
from .. import _make, analysis
from ... import nd
from ..expr import Tuple, RefCreate, Call, Constant, GlobalVar, const
from ..function import Function
from ..scope_builder import ScopeBuilder


@tvm._ffi.register_object("relay.ConstructorValue")
class ConstructorValue(Object):
    def __init__(self, tag, fields, constructor):
        self.__init_handle_by_constructor__(_make.ConstructorValue, tag, fields, constructor)


@tvm._ffi.register_object("relay.RefValue")
class RefValue(Object):
    def __init__(self, value):
        self.__init_handle_by_constructor__(_make.RefValue, value)


def _arg_to_ast(mod, arg):
    if isinstance(arg, nd.NDArray):
        return Constant(arg.copyto(nd.cpu(0)))
    elif isinstance(arg, container.ADT):
        return Tuple([_arg_to_ast(mod, field) for field in arg])
    elif isinstance(arg, tuple):
        return Tuple([_arg_to_ast(mod, field) for field in arg])
    elif isinstance(arg, RefValue):
        return RefCreate(_arg_to_ast(mod, arg.value))
    elif isinstance(arg, ConstructorValue):
        return Call(mod.get_constructor(arg.tag), [_arg_to_ast(mod, field) for field in arg.fields])
    elif isinstance(arg, np.ndarray):
        return Constant(nd.array(arg))
    elif isinstance(arg, Constant):
        return arg
    else:
        return const(arg)


class Executor(object):
    """An abstract interface for executing Relay programs."""

    def _convert_args(self, expr, args, kwargs):
        """
        Convert the combination of arguments and keyword arguments
        into a sequence of arguments that may be passed to
        a Relay evaluator.

        We first provide all positional arguments, and then attempt
        to fill in the remaining arguments using the keyword arguments. We
        map the keyword arguments to the corresponding parameters, if there
        is an ambiguity between positional and keyword arguments this
        procedure will raise an error.

        Parameters
        ----------
        expr: relay.Expr
            The expression to evaluate

        args: List[tvm.nd.NDArray]
            The arguments to pass to the evaluator.

        kwargs: Dict[str, tvm.NDArrray]
            The keyword arguments to pass to the evaluator.

        Returns:
            args: List[tvm.nd.NDArray]
                The new arguments with all keyword arguments placed in the correct slot.
        """
        assert expr is not None

        if not kwargs:
            return args

        if kwargs and not isinstance(expr, Function):
            raise Exception(
                "can only supply keyword parameters for a " "relay.Function, found {0}".format(expr)
            )

        params = expr.params
        param_names = [p.name_hint for p in params]
        num_of_args = len(args)

        cargs = list(args)[:]
        for i, name in enumerate(param_names):
            if i < num_of_args:
                if kwargs.get(name):
                    raise Exception(
                        "duplicate argument supplied in "
                        "both positional args (at position: {0}), "
                        "and keyword argument (with name: {1})".format(i, name)
                    )
            else:
                cargs.append(kwargs[name])

        if len(cargs) != len(params):
            raise Exception(
                "insufficient arguments, expected "
                "{0}, provided {1}".format(len(cargs), len(params))
            )

        return tuple(cargs)

    def _make_executor(self, expr=None):
        """
        Construct a Python function that implements the evaluation
        of expression.

        Parameters
        ----------
        expr: Optional[relay.Expr]
            The Relay expression to execute.

        Returns
        -------
        executor: function,
            A Python function which implements the behavior of `expr`.
        """
        raise NotImplementedError()

    def evaluate(self, expr=None, binds=None):
        """
        Evaluate a Relay expression on the executor.

        Parameters
        ----------
        expr: Optional[tvm.relay.Expr]
            The expression to evaluate.

        binds: Optional[Map[tvm.relay.Var, tvm.relay.Expr]]
            Additional binding of free variable.

        Returns
        -------
        val : Union[function, Object]
            The evaluation result.
        """
        if binds:
            scope_builder = ScopeBuilder()
            for key, value in binds.items():
                scope_builder.let(key, _arg_to_ast(self.mod, value))
            scope_builder.ret(expr)
            expr = scope_builder.get()

        if not expr:
            return self._make_executor()

        if isinstance(expr, Function):
            assert not analysis.free_vars(expr)

        if isinstance(expr, (Function, GlobalVar)):
            return self._make_executor(expr)

        # normal expression evaluated by running a function.
        # TODO(mbs): This should really be type rather than syntax driven.
        func = Function([], expr)
        return self._make_executor(func)()


class Interpreter(Executor):
    """
    Simple interpreter interface.

    Parameters
    ----------
    mod : tvm.IRModule
        The module to support the execution.

    device : Device
        The runtime device to run the code on.

    target : tvm.Target
        The target option to build the function. Only homogeneous execution is supported.

    CAUTION: Despite the API the module is prepared upon each call to evaluate
    rather than once in create_executor.
    That is:
    .. code-block:: python

        executor = relay.create_executor(kind="debug", mod=module)
        a = executor.evaluate(expr)(args1)
        b = executor.evaluate(expr)(args2)

    will prepare all the bindings in module twice. For efficiency, try to hoist
    calls to evaluate as high as possible, preferably immediately after create_executor:
    .. code-block:: python

        func = relay.create_executor(kind="debug", mod=module).evaluate(expr)
        a = func(args1)
        b = func(args2)
    """

    def __init__(self, mod, device, target):
        self.mod = mod
        self.device = device
        self.target = target

    def _make_executor(self, expr=None):
        if expr is None or isinstance(expr, GlobalVar):
            assert self.mod is not None

        if expr is None:
            # A missing expr denotes 'main' in the given module.
            expr = self.mod.get_global_var("main")

        # Evaluate expr to a packed function we can efficiently re-apply
        # to Relay arguments.
        func = _backend.EvalFunction(self.mod, expr, self.device, self.target)

        def _apply_args(*args, **kwargs):
            if isinstance(expr, GlobalVar):
                # When expanding args, look inside the actual global definition so kwargs
                # can be matched.
                args = self._convert_args(self.mod[expr.name_hint], args, kwargs)
            else:
                args = self._convert_args(expr, args, kwargs)
            # Reflect python arguments up into Relay.
            relay_args = []
            for arg in args:
                relay_args.append(_arg_to_ast(self.mod, arg))
            # Apply func to Relay args
            return func(relay_args)

        return _apply_args
