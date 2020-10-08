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

# pylint: disable=invalid-name, unused-import
"""Invoke python signal handlers after interrupted C syscalls."""

import atexit
import ctypes
import logging
import sys
from . import get_global_func


_LOG = logging.getLogger(__name__)


_CAUGHT_SIGNAL_EXCEPTION = None


def signal_decorator(func):
     def handler(*args):
          try:
               return func(*args)
          except BaseException as e:
               _CAUGHT_SIGNAL_EXCEPTION = e
               return

     return handler


def syscall_interrupted_callback():
     # NOTE: This function is pretty subtle especially on Python 3.7 and below.
     # Its purpose is to ensure that Python signal handlers (i.e. set with signal.signal) are
     # invoked when the TVM C code is blocked on a syscall. But, since this function is written in
     # Python, and there is some scaffolding around even calling this function from C (see
     # packed_func.py), it's very likely that the signal handlers were actually invoked by the time
     # we got here. Nevertheless, call PyErr_CheckSignals anyway just to be sure. Note that
     # ordinarily this would be called from C code, but we want the TVM library to be language-
     # agnostic.
     #
     # Now for the subtle part: in Python 3.7 and below, errors raised from signal handlers invoked
     # in this ctypes callback context are merely printed to stderr and ignored. This means that
     # e.g. raising KeyboardInterrupt from a SIGINT handler is meaningless when used to interrupt
     # TVM system calls that use this mechanism. Therefore, here, mimic the behavior of Python 3.7-
     # by swallowing any errors found in PyErr_CheckSignals and forcing the use of signal_decorator.
     print('syscall interrupted CB', sys.exc_info())
     if _CAUGHT_SIGNAL_EXCEPTION is not None:
          exc = _CAUGHT_SIGNAL_EXCEPTION
          _CAUGHT_SIGNAL_EXCEPTION = None
          raise exc

     try:
          rv = ctypes.pythonapi.PyErr_CheckSignals()
          sys.stdout.write('rv %r\n' % (rv,))
     except BaseException as e:
          _LOG.error('swallowing error from PyErr_CheckSignals', exc_info=True)

     if rv != 0:
          exc_type = sys.exc_info()[0]
          raise exc_type
     return None


def packedfunc_excepthook(unraisable):
     print('PF EXCEPTHOOK!')


def init_syscall_interrupted():
    get_global_func("tvm.runtime.SetSyscallInterruptedCallback")(syscall_interrupted_callback)
    sys.unraisablehook = packedfunc_excepthook


def syscall_interrupted_atexit():
    get_global_func("tvm.runtime.SetSyscallInterruptedCallback")(None)


atexit.register(syscall_interrupted_atexit)
