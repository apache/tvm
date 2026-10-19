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
# pylint: disable=invalid-name,unused-argument
"""Common pass infrastructure across IR variants."""

import functools
import inspect

import tvm_ffi

from . import _ffi_transform_api


@tvm_ffi.register_object("transform.PassInfo")
class PassInfo(tvm_ffi.Object):
    """The class contains the meta data required by a pass. It is the
    container of information needed by running an optimization or analysis.
    This class can be extended by adding new members when more meta data is
    needed.

    Parameters
    ----------
    opt_level : int
        The optimization level of this pass.

    name : str
        The pass name.
    """

    def __init__(self, opt_level, name):
        self.__init_handle_by_constructor__(_ffi_transform_api.PassInfo, opt_level, name)


@tvm_ffi.register_object("transform.PassInstrument")
class PassInstrument(tvm_ffi.Object):
    """A pass instrument implementation.

    To use, a user class can either subclass from PassInstrument
    directly, or can apply the :py:func:`pass_instrument` wrapper.  In
    either case, the `enter_pass_ctx`, `exit_pass_ctx`, `should_run`,
    `run_before_pass`, and `run_after_pass` methods can be defined to
    adjust the instrument's behavior.  See the no-op implementations
    in this class definition for more information on each.

    """

    def __init__(self):
        # initialize handle in case pi_cls creation failed.
        cls = type(self)

        # If the child class declared the method, then use it.
        # Otherwise, pass None to avoid a C++ -> Python round trip for
        # a no-op.
        def get_child_method(name):
            if getattr(cls, name) is getattr(PassInstrument, name):
                return None

            return getattr(self, name)

        # Create runtime pass instrument object.
        # register instance's enter_pass_ctx,exit_pass_ctx, should_run, run_before_pass and
        # run_after_pass methods to it if present.
        self.__init_handle_by_constructor__(
            _ffi_transform_api.PassInstrument,
            cls.__name__,
            get_child_method("enter_pass_ctx"),
            get_child_method("exit_pass_ctx"),
            get_child_method("should_run"),
            get_child_method("run_before_pass"),
            get_child_method("run_after_pass"),
        )

    def enter_pass_ctx(self):
        """Called when entering the instrumented context.

        Returns
        -------
        None
        """

    def exit_pass_ctx(self):
        """Called when exiting the instrumented context.

        Returns
        -------
        None
        """

    def should_run(self, mod, info):
        """Determine whether to run the pass or not.

        Called once for each pass that is run while the instrumented
        context is active.

        Parameters
        ----------
        mod : tvm_tirx.ir.module.IRModule

            The module on which an optimization pass is being run.

        info : tvm_tirx.ir.transform.PassInfo

            The pass information.

        Returns
        -------
        should_run : bool

            True to run the pass, or False to skip the pass.
        """

    def run_before_pass(self, mod, info):
        """Instrument before the pass runs.

        Called once for each pass that is run while the instrumented
        context is active.

        Parameters
        ----------
        mod : tvm_tirx.ir.module.IRModule

            The module on which an optimization pass is being run.

        info : tvm_tirx.ir.transform.PassInfo

            The pass information.

        Returns
        -------
        None
        """

    def run_after_pass(self, mod, info):
        """Instrument after the pass runs.

        Called once for each pass that is run while the instrumented
        context is active.

        Parameters
        ----------
        mod : tvm_tirx.ir.module.IRModule

            The module on which an optimization pass is being run.

        info : tvm_tirx.ir.transform.PassInfo

            The pass information.

        Returns
        -------
        None
        """


def _wrap_class_pass_instrument(pi_cls):
    """Wrap a python class as pass instrument"""

    # No additional wrapping needed if the user class already
    # inherits.
    if issubclass(pi_cls, PassInstrument):
        return pi_cls

    class PyPassInstrument(pi_cls, PassInstrument):
        """Internal wrapper class to create a class instance."""

        def __init__(self, *args, **kwargs):
            # initialize handle in case pi_cls creation failed.
            pi_cls.__init__(self, *args, **kwargs)
            PassInstrument.__init__(self)

    functools.update_wrapper(PyPassInstrument.__init__, pi_cls.__init__)
    PyPassInstrument.__name__ = pi_cls.__name__
    PyPassInstrument.__doc__ = pi_cls.__doc__
    PyPassInstrument.__module__ = pi_cls.__module__
    return PyPassInstrument


def pass_instrument(pi_cls=None):
    """Decorate a pass instrument.

    Parameters
    ----------
    pi_cls : class
        Instrument class. See example below.
    """

    def create_pass_instrument(pi_cls):
        if not inspect.isclass(pi_cls):
            raise TypeError("pi_cls must be a class")

        return _wrap_class_pass_instrument(pi_cls)

    if pi_cls:
        return create_pass_instrument(pi_cls)
    return create_pass_instrument


@tvm_ffi.register_object("transform.PassContext")
class PassContext(tvm_ffi.Object):
    """Scope optimization levels, pass selection, instruments, and pass options.

    Parameters
    ----------
    opt_level : Optional[int]
        Maximum optimization level enabled in sequential pipelines (default 2).

    required_pass : Optional[Union[List[str], Set[str], Tuple[str]]]
        Pass names to enable regardless of their optimization level.

    disabled_pass : Optional[Union[List[str], Set[str], Tuple[str]]]
        Pass names to skip in sequential pipelines, even if required.

    instruments : Optional[Sequence[PassInstrument]]
        The list of pass instrument implementations.

    config : Optional[Dict[str, Object]]
        Additional configurations for specific passes.
    """

    def __init__(
        self,
        opt_level=2,
        required_pass=None,
        disabled_pass=None,
        instruments=None,
        config=None,
    ):
        required = list(required_pass) if required_pass else []
        if not isinstance(required, list | tuple):
            raise TypeError("required_pass is expected to be the type of " + "list/tuple/set.")

        disabled = list(disabled_pass) if disabled_pass else []
        if not isinstance(disabled, list | tuple):
            raise TypeError("disabled_pass is expected to be the type of " + "list/tuple/set.")

        instruments = list(instruments) if instruments else []
        if not isinstance(instruments, list | tuple):
            raise TypeError("instruments is expected to be the type of " + "list/tuple/set.")

        config = config if config else None
        self.__init_handle_by_constructor__(
            _ffi_transform_api.PassContext,
            opt_level,
            required,
            disabled,
            instruments,
            config,
        )

    def __enter__(self):
        _ffi_transform_api.EnterPassContext(self)
        return self

    def __exit__(self, ptype, value, trace):
        _ffi_transform_api.ExitPassContext(self)

    def override_instruments(self, instruments):
        """Override instruments within this PassContext.

        If there are existing instruments, their ``exit_pass_ctx`` callbacks are called.
        Then switching to new instruments and calling new ``enter_pass_ctx`` callbacks.

        instruments : Sequence[PassInstrument]
            The list of pass instrument implementations.
        """
        _ffi_transform_api.OverrideInstruments(self, instruments)

    @staticmethod
    def current():
        """Return the current pass context."""
        return _ffi_transform_api.GetCurrentPassContext()

    @staticmethod
    def list_configs():
        """List all registered `PassContext` configuration names and metadata.

        Returns
        -------
        configs : Dict[str, Dict[str, str]]

        """
        return _ffi_transform_api.ListConfigs()


@tvm_ffi.register_object("transform.Pass")
class Pass(tvm_ffi.Object):
    """Base interface for callable IRModule transformations and analyses.

    Obtain a concrete pass from a pass factory or :func:`module_pass`, then
    call ``after = pass_object(before)``. The :attr:`info` property exposes
    its name and optimization level. The base class defines no transformation
    of its own.
    """

    __slots__ = ("__dict__",)

    @property
    def info(self):
        """Get the pass meta."""
        return _ffi_transform_api.Info(self)

    def __call__(self, mod):
        """Execute the pass on a module.

        Parameters
        ----------
        mod : tvm_tirx.IRModule
            The module that a certain optimization is performed on.

        Returns
        -------
        mod : tvm_tirx.IRModule
            The updated module after applying this pass.
        """
        return _ffi_transform_api.RunPass(self, mod)


@tvm_ffi.register_object("transform.ModulePass")
class ModulePass(Pass):
    """A pass whose callback receives and returns a complete IRModule.

    Create instances with :func:`module_pass`, which accepts a function or a
    class defining ``transform_module``. Call ``after = pass_object(before)``
    to execute the callback under the current PassContext. Its callback,
    rather than this wrapper class, defines how the IR changes; see
    :func:`module_pass` for a TVMScript before/after example.
    """


@tvm_ffi.register_object("transform.Sequential")
class Sequential(Pass):
    """A pass that works on a sequence of pass objects. Multiple passes can be
    executed sequentially using this class.

    Passes run in the given order, subject to the optimization level and
    the required/disabled pass controls in PassContext.

    Parameters
    ----------
    passes : Optional[List[Pass]]
        A sequence of passes candidate for optimization.

    opt_level : Optional[int]
        The optimization level of this sequential pass.
        The opt_level of a default sequential pass is set to 0.
        Child passes use the active PassContext's optimization level.

    name : Optional[str]
        The name of the sequential pass.
    """

    def __init__(self, passes=None, opt_level=0, name="sequential"):
        passes = passes if passes else []
        if not isinstance(passes, list | tuple):
            raise TypeError("passes must be a list of Pass objects.")

        self.__init_handle_by_constructor__(_ffi_transform_api.Sequential, passes, opt_level, name)


def _wrap_class_module_pass(pass_cls, pass_info):
    """Wrap a python class as function pass"""

    class PyModulePass(ModulePass):
        """Internal wrapper class to create a class instance."""

        def __init__(self, *args, **kwargs):
            inst = pass_cls(*args, **kwargs)

            # it is important not to capture self to
            # avoid a cyclic dependency
            def _pass_func(mod, ctx):
                return inst.transform_module(mod, ctx)

            self.__init_handle_by_constructor__(
                _ffi_transform_api.MakeModulePass, _pass_func, pass_info
            )
            self._inst = inst

        def __getattr__(self, name):
            # fall back to instance attribute if there is not any
            return self._inst.__getattribute__(name)

    functools.update_wrapper(PyModulePass.__init__, pass_cls.__init__)
    PyModulePass.__name__ = pass_cls.__name__
    PyModulePass.__doc__ = pass_cls.__doc__
    PyModulePass.__module__ = pass_cls.__module__
    return PyModulePass


def module_pass(pass_func=None, opt_level=None, name=None):
    """Decorate a module pass.

    This function returns a callback when pass_func is provided.
    Otherwise, it serves a decorator function.

    pass_func can also be a class type with a method transform_module.
    This function will create a decorated ModulePass using transform_module
    as the pass function.

    Parameters
    ----------
    pass_func : Optional[Callable[(Module, PassContext) ->Module]]
        The transformation function or class.

    opt_level : int
        The optimization level of this module pass.

    name : Optional[str]
        The name of the module pass. The name could be empty. In this case, the
        name of the optimization function will be used as the pass name.

    Returns
    -------
    create_module_pass : Union[Callable, ModulePass]
        A decorator will be returned if pass_func is not provided,
        otherwise return the decorated result.
        The returned decorator has two behaviors depending on the input:
        A new ModulePass will be returned when we decorate a pass function.
        A new ModulePass class will be returned when we decorate a class type.
    """
    if opt_level is None:
        raise ValueError("Please provide opt_level for the module pass.")

    def create_module_pass(pass_arg):
        """Internal function that creates a module pass"""
        fname = name if name else pass_arg.__name__
        info = PassInfo(opt_level, fname)
        if inspect.isclass(pass_arg):
            return _wrap_class_module_pass(pass_arg, info)
        if not callable(pass_arg):
            raise TypeError("pass_func must be a callable for Module pass")
        return _ffi_transform_api.MakeModulePass(pass_arg, info)

    if pass_func:
        return create_module_pass(pass_func)
    return create_module_pass


def PrintIR(header=""):
    """Log the header and module IR, returning the module unchanged.

    Parameters
    ----------
    header : str
        The header to be displayed along with the dump.

    Returns
    -------
    fpass : Pass
        A module pass with optimization level zero.

    Examples
    --------
    The observation pass returns the module unchanged.

    .. code-block:: python

        import tvm_tirx as tvm
        from tvm_tirx.script import ir as I, tirx as T

        @I.ir_module
        class Before:
            @T.function(private=True)
            def main():
                T.evaluate(0)

        @I.ir_module
        class Expected:
            @T.function(private=True)
            def main():
                T.evaluate(0)

        def run_pass():
            return tvm.ir.transform.PrintIR("before lowering")(Before)

        tvm.ir.assert_structural_equal(run_pass(), Expected)
    """
    return _ffi_transform_api.PrintIR(header)
