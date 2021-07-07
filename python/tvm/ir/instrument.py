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

    Users don't need to interact with this class directly.
    Instead, a `PassInstrument` instance should be created through
    :py:func:`pass_instrument`
    """


def _wrap_class_pass_instrument(pi_cls):
    """Wrap a python class as pass instrument"""

    class PyPassInstrument(PassInstrument):
        """Internal wrapper class to create a class instance."""

        def __init__(self, *args, **kwargs):
            # initialize handle in case pi_cls creation failed.
            self.handle = None
            inst = pi_cls(*args, **kwargs)

            # check method declartion within class, if found, wrap it.
            def create_method(method):
                if hasattr(inst, method) and inspect.ismethod(getattr(inst, method)):

                    def func(*args):
                        return getattr(inst, method)(*args)

                    func.__name__ = "_" + method
                    return func
                return None

            # create runtime pass instrument object
            # reister instance's enter_pass_ctx,exit_pass_ctx, should_run, run_before_pass and
            # run_after_pass methods to it if present.
            self.__init_handle_by_constructor__(
                _ffi_instrument_api.PassInstrument,
                pi_cls.__name__,
                create_method("enter_pass_ctx"),
                create_method("exit_pass_ctx"),
                create_method("should_run"),
                create_method("run_before_pass"),
                create_method("run_after_pass"),
            )

            self._inst = inst

        def __getattr__(self, name):
            # fall back to instance attribute if there is not any
            return self._inst.__getattribute__(name)

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
