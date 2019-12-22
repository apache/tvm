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
#pylint: disable=no-else-return
"""The Python interface to the Relay reference interpreter."""
from __future__ import absolute_import

import numpy as np

from . import _backend
from .. import _make, analysis, transform
from .. import module
from ... import register_func, nd
from ..base import NodeBase, register_relay_node
from ..expr import Tuple, RefCreate, Call, Constant, GlobalVar, Function, const
from ..scope_builder import ScopeBuilder
from . import _vm

class Value(NodeBase):
    """Base class of all values.
    """
    @staticmethod
    @register_func("relay.from_scalar")
    def from_scalar(value, dtype=None):
        """Convert a Python scalar to a Relay scalar."""
        return TensorValue(const(value, dtype).data)

    def to_vm(self):
        return _vm._ValueToVM(self)


@register_relay_node
class TupleValue(Value):
    """A tuple value produced by the interpreter."""
    def __init__(self, *fields):
        self.__init_handle_by_constructor__(
            _make.TupleValue, fields)

    def __getitem__(self, field_no):
        return self.fields[field_no]

    def __len__(self):
        return len(self.fields)

    def __str__(self):
        body = ','.join(str(f) for f in self.fields)
        return '({0})'.format(body)

    def __repr__(self):
        body = ','.join(repr(f) for f in self.fields)
        return '({0})'.format(body)

    def __iter__(self):
        return iter(self.fields)


@register_relay_node
class Closure(Value):
    """A closure produced by the interpreter."""


@register_relay_node
class RecClosure(Value):
    """A recursive closure produced by the interpreter."""


@register_relay_node
class ConstructorValue(Value):
    def __init__(self, tag, fields, constructor):
        self.__init_handle_by_constructor__(
            _make.ConstructorValue, tag, fields, constructor)


@register_relay_node
class TensorValue(Value):
    """A Tensor value produced by the interpreter."""

    def __init__(self, data):
        """Allocate a new TensorValue and copy the data from `array` into
           the new array.
        """
        if isinstance(data, np.ndarray):
            data = nd.array(data)

        self.__init_handle_by_constructor__(
            _make.TensorValue, data)

    def asnumpy(self):
        """Convert a Relay TensorValue into a numpy.ndarray."""
        return self.data.asnumpy()

    def __eq__(self, other):
        return self.data == other.data

    def __repr__(self):
        return repr(self.data)

    def __str__(self):
        return str(self.data)


@register_relay_node
class RefValue(Value):
    def __init__(self, value):
        self.__init_handle_by_constructor__(
            _make.RefValue, value)


def _arg_to_ast(mod, arg):
    if isinstance(arg, TensorValue):
        return Constant(arg.data.copyto(nd.cpu(0)))
    elif isinstance(arg, TupleValue):
        return Tuple([_arg_to_ast(mod, field) for field in arg.fields])
    elif isinstance(arg, tuple):
        return Tuple([_arg_to_ast(mod, field) for field in arg])
    elif isinstance(arg, RefValue):
        return RefCreate(_arg_to_ast(mod, arg.value))
    elif isinstance(arg, ConstructorValue):
        return Call(mod.get_constructor(arg.tag),
                    [_arg_to_ast(mod, field) for field in arg.fields])
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

        args: List[tvm.NDArray]
            The arguments to pass to the evaluator.

        kwargs: Dict[str, tvm.NDArrray]
            The keyword arguments to pass to the evaluator.

        Returns:
            args: List[tvm.NDArray]
                The new arguments with all keyword arguments placed in the correct slot.
        """
        assert expr is not None

        if not kwargs:
            return args

        if kwargs and not isinstance(expr, Function):
            raise Exception("can only supply keyword parameters for a "
                            "relay.Function, found {0}".format(expr))

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
                        "and keyword argument (with name: {1})".format(i, name))
            else:
                cargs.append(kwargs[name])

        if len(cargs) != len(params):
            raise Exception(
                "insufficient arguments, expected "
                "{0}, provided {1}".format(len(cargs), len(params)))

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
        val : Union[function, Value]
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
        func = Function([], expr)
        return self._make_executor(func)()


class Interpreter(Executor):
    """
    Simple interpreter interface.

    Parameters
    ----------
    mod : tvm.relay.Module
        The module to support the execution.

    ctx : tvm.TVMContext
        The runtime context to run the code on.

    target : tvm.Target
        The target option to build the function.
    """
    def __init__(self, mod, ctx, target):
        self.mod = mod
        self.ctx = ctx
        self.target = target

    def optimize(self):
        """Optimize functions in a module.

        Returns
        -------
        opt_mod : tvm.relay.Module
            The optimized module.
        """
        seq = transform.Sequential([transform.SimplifyInference(),
                                    transform.FuseOps(0),
                                    transform.ToANormalForm(),
                                    transform.InferType()])
        return seq(self.mod)

    def _make_executor(self, expr=None):
        if expr is None or isinstance(expr, GlobalVar):
            assert self.mod is not None
        def _interp_wrapper(*args, **kwargs):
            if expr is None:
                args = self._convert_args(self.mod["main"], args, kwargs)
            else:
                args = self._convert_args(expr, args, kwargs)

            relay_args = []
            for arg in args:
                relay_args.append(_arg_to_ast(self.mod, arg))

            # Set the entry function for the module.
            if expr is None:
                pass
            elif isinstance(expr, GlobalVar):
                self.mod["main"] = self.mod[expr]
            else:
                assert isinstance(expr, Function)
                func = Function([], Call(expr, relay_args))
                relay_args = []
                if self.mod:
                    self.mod["main"] = func
                else:
                    self.mod = module.Module.from_expr(func)

            mod = self.optimize()
            opt_expr = Call(mod["main"], relay_args)
            _intrp = _backend.CreateInterpreter(mod, self.ctx, self.target)
            return _intrp(opt_expr)
        return _interp_wrapper
