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

import tvm.runtime

from . import _ffi_transform_api


@tvm_ffi.register_object("transform.PassInfo")
class PassInfo(tvm.runtime.Object):
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
class PassInstrument(tvm.runtime.Object):
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
        mod : tvm.ir.module.IRModule

            The module on which an optimization pass is being run.

        info : tvm.transform.PassInfo

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
        mod : tvm.ir.module.IRModule

            The module on which an optimization pass is being run.

        info : tvm.transform.PassInfo

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
        mod : tvm.ir.module.IRModule

            The module on which an optimization pass is being run.

        info : tvm.transform.PassInfo

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
    pi_class : class
        Instrument class. See example below.

    Examples
    --------

    .. code-block:: python

        @tvm.transform.pass_instrument
        class SkipPass:
            def __init__(self, skip_pass_name):
                self.skip_pass_name = skip_pass_name

            # Uncomment to customize
            # def enter_pass_ctx(self):
            #    pass

            # Uncomment to customize
            # def exit_pass_ctx(self):
            #    pass

            # If pass name contains keyword, skip it by return False. (return True: not skip)
            def should_run(self, mod, pass_info)
                if self.skip_pass_name in pass_info.name:
                    return False
                return True

            # Uncomment to customize
            # def run_before_pass(self, mod, pass_info):
            #    pass

            # Uncomment to customize
            # def run_after_pass(self, mod, pass_info):
            #    pass

        skip_annotate = SkipPass("AnnotateSpans")
        with tvm.transform.PassContext(instruments=[skip_annotate]):
            tvm.compile(mod, "llvm")
    """

    def create_pass_instrument(pi_cls):
        if not inspect.isclass(pi_cls):
            raise TypeError("pi_cls must be a class")

        return _wrap_class_pass_instrument(pi_cls)

    if pi_cls:
        return create_pass_instrument(pi_cls)
    return create_pass_instrument


@tvm_ffi.register_object("transform.PassContext")
class PassContext(tvm.runtime.Object):
    """The basis where a TVM optimization/analysis runs on.
    Each pass context contains a number of auxiliary information that is used
    to help an optimization pass. Such information includes the error reporter
    to record the errors of during the optimization, etc.

    opt_level : Optional[int]
        The optimization level of this pass.

    required_pass : Optional[Union[List[str], Set[str], Tuple[str]]]
        The list of passes that are required by a certain pass.

    disabled_pass : Optional[Union[List[str], Set[str], Tuple[str]]]
        The list of passes that are disabled.

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
class Pass(tvm.runtime.Object):
    """The base class of all passes. All methods here are just simple wrappers
    that are implemented in the backend. They are defined for users to
    conveniently interact with the base class.
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
        mod : tvm.IRModule
            The module that a certain optimization is performed on.

        Returns
        -------
        mod : tvm.IRModule
            The updated module after applying this pass.
        """
        return _ffi_transform_api.RunPass(self, mod)


@tvm_ffi.register_object("transform.ModulePass")
class ModulePass(Pass):
    """A pass that works on tvm.IRModule. Users don't need to interact with
    this class directly. Instead, a module pass should be created through
    `module_pass`, because the design of the `module_pass` API is flexible
    enough to handle the creation of a module pass in different manners. In
    addition, all members of a module pass can be accessed from the base class.
    The same rule applies to FunctionPass as well.
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
        Note that some of the passes within the Sequantial may still not be executed
        if their opt_level is higher than the provided opt_level.

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

    Examples
    --------
    The following code block decorates a module pass class.

    .. code-block:: python

        @tvm.transform.module_pass
        class CustomPipeline:
            def __init__(self, enable_fold):
                self.enable_fold = enable_fold
                self.const_fold = relax.transform.FoldConstant()

            def transform_module(self, mod, ctx):
                if self.enable_fold:
                    mod = self.const_fold(mod, ctx)
                return mod

        # create an instance of customized pipeline
        pipeline = CustomPipeline(enable_fold=False)
        assert isinstance(pipeline, transform.ModulePass)
        # run the pipeline.
        output_module = pipeline(input_module)

    The following code creates a module pass by decorating
    a user defined transform function.

    .. code-block:: python

        @tvm.transform.module_pass(opt_level=2)
        def transform(mod, ctx):
            return relax.transform.FoldConstant(mod)

        module_pass = transform
        assert isinstance(module_pass, transform.ModulePass)
        assert module_pass.info.opt_level == 2

        # Given a module m, the optimization could be invoked as the follwoing:
        updated_mod = module_pass(m)
        # Now a function abs should be added to the module m.
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
    """A special trace pass that prints the header and IR.

    Parameters
    ----------
    header : str
        The header to be displayed along with the dump.

    Returns
    --------
    The pass
    """
    return _ffi_transform_api.PrintIR(header)
