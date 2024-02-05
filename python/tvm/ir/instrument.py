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
"""Common pass instrumentation across IR variants."""
import inspect
import functools

import tvm._ffi
import tvm.runtime

from . import _ffi_instrument_api


@tvm._ffi.register_object("instrument.PassInstrument")
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
        self.handle = None
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
            _ffi_instrument_api.PassInstrument,
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
            self.handle = None
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

        @tvm.instrument.pass_instrument
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
            tvm.relay.build(mod, "llvm")
    """

    def create_pass_instrument(pi_cls):
        if not inspect.isclass(pi_cls):
            raise TypeError("pi_cls must be a class")

        return _wrap_class_pass_instrument(pi_cls)

    if pi_cls:
        return create_pass_instrument(pi_cls)
    return create_pass_instrument


@tvm._ffi.register_object("instrument.PassInstrument")
class PassTimingInstrument(tvm.runtime.Object):
    """A wrapper to create a passes time instrument that implemented in C++"""

    def __init__(self):
        self.__init_handle_by_constructor__(_ffi_instrument_api.MakePassTimingInstrument)

    @staticmethod
    def render():
        """Retrieve rendered time profile result
        Returns
        -------
        string : string
            The rendered string result of time profiles

        Examples
        --------

        .. code-block:: python

            timing_inst = PassTimingInstrument()
            with tvm.transform.PassContext(instruments=[timing_inst]):
                relay_mod = relay.transform.InferType()(relay_mod)
                relay_mod = relay.transform.FoldScaleAxis()(relay_mod)
                # before exiting the context, get profile results.
                profiles = timing_inst.render()
        """
        return _ffi_instrument_api.RenderTimePassProfiles()


@pass_instrument
class PassPrintingInstrument:
    """A pass instrument to print if before or
    print ir after each element of a named pass."""

    def __init__(self, print_before_pass_names, print_after_pass_names):
        self.print_before_pass_names = print_before_pass_names
        self.print_after_pass_names = print_after_pass_names

    def run_before_pass(self, mod, pass_info):
        if pass_info.name in self.print_before_pass_names:
            print(f"Print IR before: {pass_info.name}\n{mod}\n\n")

    def run_after_pass(self, mod, pass_info):
        if pass_info.name in self.print_after_pass_names:
            print(f"Print IR after: {pass_info.name}\n{mod}\n\n")


@pass_instrument
class PrintAfterAll:
    """Print the name of the pass, the IR, only after passes execute."""

    def run_after_pass(self, mod, info):
        print(f"After Running Pass: {info}")
        print(mod)


@pass_instrument
class PrintBeforeAll:
    """Print the name of the pass, the IR, only before passes execute."""

    def run_before_pass(self, mod, info):
        print(f"Before Running Pass: {info}")
        print(mod)
