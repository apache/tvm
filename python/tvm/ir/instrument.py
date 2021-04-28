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
import tvm._ffi
import tvm.runtime

from . import _ffi_instrument_api


@tvm._ffi.register_object("instrument.PassInstrument")
class PassInstrument(tvm.runtime.Object):
    """A pass instrument implementation.

    Parameters
    ----------
    name : str
        The name for this instrument implementation.

    Examples
    --------

    .. code-block:: python
        pi = tvm.instrument.PassInstrument("print-before-after")

        @pi.register_set_up
        def set_up():
          pass

        @pi.register_tear_down
        def tear_down():
          pass

        @pi.register_run_before_pass
        def run_before_pass(mod, info):
          print("Before pass: " + info.name)
          print(mod)
          return True

        @pi.register_run_after_pass
        def run_after_pass(mod, info):
          print("After pass: " + info.name)
          print(mod)


    See Also
    --------
    instrument.PassInstrumentor
    """

    def __init__(self, name):
        self.__init_handle_by_constructor__(_ffi_instrument_api.PassInstrument, name)

    def register_set_up(self, callback):
        _ffi_instrument_api.RegisterSetUpCallback(self, callback)

    def register_tear_down(self, callback):
        _ffi_instrument_api.RegisterTearDownCallback(self, callback)

    def register_run_before_pass(self, callback):
        _ffi_instrument_api.RegisterRunBeforePassCallback(self, callback)

    def register_run_after_pass(self, callback):
        _ffi_instrument_api.RegisterRunAfterPassCallback(self, callback)


@tvm._ffi.register_object("instrument.PassInstrumentor")
class PassInstrumentor(tvm.runtime.Object):
    """A pass instrumentor collects a set of pass instrument implementations.

    Parameters
    ----------
    pass_instruments : List[tvm.instrument.PassInstrument]
        List of instrumentation to run within pass context

    Examples
    --------
    .. code-block:: python

        passes_mem = #... Impl of memory instrument
        passes_time = tvm.instrument.PassesTimeInstrument()

        with tvm.transform.PassContext(
            pass_instrumentor=tvm.instrument.PassInstrumentor([passes_mem, passes_time])):
            tvm.relay.build(mod, 'llvm')

            print(passes_time.rendor())

    See Also
    -------
    instrument.PassInstrument
    instrument.PassesTimeInstrument
    """

    def __init__(self, pass_instruments):
        self.__init_handle_by_constructor__(_ffi_instrument_api.PassInstrumentor, pass_instruments)


@tvm._ffi.register_object("instrument.PassInstrument")
class PassesTimeInstrument(tvm.runtime.Object):
    """A wrapper to create a passes time instrument that implemented in C++"""

    def __init__(self):
        self.__init_handle_by_constructor__(_ffi_instrument_api.MakePassesTimeInstrument)

    @staticmethod
    def render():
        """Retrieve rendered time profile result
        Returns
        -------
        string : string
            The rendered string result of time profiles
        """
        return _ffi_instrument_api.RenderTimePassProfiles()
