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
import inspect
import functools

import tvm._ffi
import tvm.runtime

from . import _ffi_transform_api


@tvm._ffi.register_object("transform.PassInfo")
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

    required : List[str]
        The list of passes that are required by a certain pass.
    """

    def __init__(self, opt_level, name, required=None, traceable=False):
        self.__init_handle_by_constructor__(
            _ffi_transform_api.PassInfo, opt_level, name, required, traceable
        )


@tvm._ffi.register_object("transform.PassContext")
class PassContext(tvm.runtime.Object):
    """The basis where a Relay optimization/analysis runs on.
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

    trace: Optional[relax.tuning.Trace]
        Initial trace for trace mode.

    trace_stack: Optional[List[relax.tuning_api.Trace]]
        Initial trace stack for trace mode.

    make_traceable: Optional[List[str]]
        List of passes to make traceable.

    num_evals: int
        initial number of evaluations conducted in the pipeline.

    tuning_api_database: Optional[relax.tuning_api.JSONDatabase]
    """

    def __init__(
        self,
        opt_level=2,
        required_pass=None,
        disabled_pass=None,
        instruments=None,
        config=None,
        trace=None,
        trace_stack=None,
        make_traceable=None,
        num_evals=0,
        tuning_api_database=None,
    ):
        required = list(required_pass) if required_pass else []
        if not isinstance(required, (list, tuple)):
            raise TypeError("required_pass is expected to be the type of " + "list/tuple/set.")

        disabled = list(disabled_pass) if disabled_pass else []
        if not isinstance(disabled, (list, tuple)):
            raise TypeError("disabled_pass is expected to be the type of " + "list/tuple/set.")

        instruments = list(instruments) if instruments else []
        if not isinstance(instruments, (list, tuple)):
            raise TypeError("instruments is expected to be the type of " + "list/tuple/set.")

        # Convert to Map<String, bool>
        # TODO(sunggg): Replace this to Set equivalent if exists
        make_traceable = {name: True for name in make_traceable} if make_traceable else None

        if not trace_stack:
            trace_stack = [trace] if trace else []

        config = config if config else None
        self.__init_handle_by_constructor__(
            _ffi_transform_api.PassContext,
            opt_level,
            required,
            disabled,
            instruments,
            config,
            trace_stack,
            make_traceable,
            num_evals,
            tuning_api_database,
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

    def push_trace(self, trace):
        """Push a trace into the stack."""
        return _ffi_transform_api.PushTrace(self, trace)

    def pop_trace(self, return_current=True):
        """Pop a topmost trace from the stack.
        Returns
        -------
        Trace : Optional[relax.tuning.Trace]
        """
        if return_current:
            cur_trace = self.get_current_trace()
            _ffi_transform_api.PopTrace(self)
            return cur_trace

        return _ffi_transform_api.PopTrace(self)

    def get_trace_stack(self):
        """Get the current trace stack."""
        return _ffi_transform_api.GetTraceStack(self)

    def get_trace_stack_size(self):
        """Get the size of current stack."""
        return _ffi_transform_api.GetTraceStackSize(self)

    def get_current_trace(self):
        """Get the trace on the top of the stack."""
        return _ffi_transform_api.GetCurrentTrace(self)

    def set_num_evals(self, num: int):
        """Set the number of evaluations conducted in the pipeline."""
        return _ffi_transform_api.SetNumEvals(self, num)

    def inc_num_evals(self, num: int):
        """Increment the number of evaluations conducted in the pipeline."""
        return _ffi_transform_api.IncNumEvals(self, num)

    def get_tuning_api_database(self):
        """Get tuning api database."""
        return _ffi_transform_api.GetTuningAPIDatabase(self)


@tvm._ffi.register_object("transform.Pass")
class Pass(tvm.runtime.Object):
    """The base class of all passes. All methods here are just simple wrappers
    that are implemented in the backend. They are defined for users to
    conveniently interact with the base class.
    """

    @property
    def info(self):
        """Get the pass meta."""
        return _ffi_transform_api.Info(self)

    def __call__(self, mod):
        """Execute the pass. Note that for sequential pass, the dependency among
        different passes will be resolved in the backend.

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


@tvm._ffi.register_object("transform.ModulePass")
class ModulePass(Pass):
    """A pass that works on tvm.IRModule. Users don't need to interact with
    this class directly. Instead, a module pass should be created through
    `module_pass`, because the design of the `module_pass` API is flexible
    enough to handle the creation of a module pass in different manners. In
    addition, all members of a module pass can be accessed from the base class.
    The same rule applies to FunctionPass as well.
    """


@tvm._ffi.register_object("transform.Sequential")
class Sequential(Pass):
    """A pass that works on a sequence of pass objects. Multiple passes can be
    executed sequentially using this class.

    Note that users can also provide a series of passes that they don't want to
    apply when running a sequential pass. Pass dependency will be resolved in
    the backend as well.

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

    required : Optional[List[str]]
        The list of passes that the sequential pass is dependent on.
    """

    def __init__(self, passes=None, opt_level=0, name="sequential", required=None, traceable=False):
        passes = passes if passes else []
        if not isinstance(passes, (list, tuple)):
            raise TypeError("passes must be a list of Pass objects.")

        required = required if required else []
        if not isinstance(required, (list, tuple)):
            raise TypeError("Required is expected to be the type of list/tuple.")

        self.__init_handle_by_constructor__(
            _ffi_transform_api.Sequential, passes, opt_level, name, required, traceable
        )


def _wrap_class_module_pass(pass_cls, pass_info):
    """Wrap a python class as function pass"""

    class PyModulePass(ModulePass):
        """Internal wrapper class to create a class instance."""

        def __init__(self, *args, **kwargs):
            # initialize handle in cass pass_cls creation failed.fg
            self.handle = None
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


def module_pass(pass_func=None, opt_level=None, name=None, required=None, traceable=False):
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

    required : Optional[List[str]]
        The list of passes that the module pass is dependent on.

    traceable: Boolean
        Boolean variable whether the module pass is traceable

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

        @relay.transform.module_pass
        class CustomPipeline:
            def __init__(self, enable_fold):
                self.enable_fold = enable_fold
                self.cse = relay.transform.EliminateCommonSubexpr()
                self.const_fold = relay.transform.FoldConstant()

            def transform_module(self, mod, ctx):
                mod = self.cse(mod, ctx)
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

        @relay.transform.module_pass(opt_level=2)
        def transform(mod, ctx):
            tp = relay.TensorType((10,), "float32")
            x = relay.var("x", tp)
            gv = relay.GlobalVar("var")
            func = relay.Function([x], relay.abs(x))
            new_mod = tvm.IRModule({gv: func})
            new_mod.update(mod)
            return new_mod

        module_pass = transform
        assert isinstance(module_pass, transform.ModulePass)
        assert module_pass.info.opt_level == 2

        # Given a module m, the optimization could be invoked as the follwoing:
        updated_mod = module_pass(m)
        # Now a function abs should be added to the module m.
    """
    if opt_level is None:
        raise ValueError("Please provide opt_level for the module pass.")

    required = required if required else []
    if not isinstance(required, (list, tuple)):
        raise TypeError("Required is expected to be the type of " + "list/tuple.")

    def create_module_pass(pass_arg):
        """Internal function that creates a module pass"""
        fname = name if name else pass_arg.__name__
        info = PassInfo(opt_level, fname, required, traceable)
        if inspect.isclass(pass_arg):
            return _wrap_class_module_pass(pass_arg, info)
        if not callable(pass_arg):
            raise TypeError("pass_func must be a callable for Module pass")
        return _ffi_transform_api.MakeModulePass(pass_arg, info)

    if pass_func:
        return create_module_pass(pass_func)
    return create_module_pass


def PrintIR(header="", show_meta_data=False):
    """A special trace pass that prints the header and IR.

    Parameters
    ----------
    header : str
        The header to be displayed along with the dump.

    show_meta_data : bool
        A boolean flag to indicate if meta data should be printed.

    Returns
    --------
    The pass
    """
    return _ffi_transform_api.PrintIR(header, show_meta_data)
