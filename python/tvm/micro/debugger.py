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

"""Defines functions for controlling debuggers for micro TVM binaries."""

import atexit
import abc
import errno
import logging
import os
import signal
import subprocess
import sys
import time
import termios
import threading

import psutil

from .._ffi import register_func
from . import class_factory
from . import transport


_LOG = logging.getLogger(__name__)


class Debugger(metaclass=abc.ABCMeta):
    """An interface for controlling micro TVM debuggers."""

    def __init__(self):
        self.on_terminate_callbacks = []

    @abc.abstractmethod
    def start(self):
        """Start the debugger, but do not block on it.

        The runtime will continue to be driven in the background.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def stop(self):
        """Terminate the debugger."""
        raise NotImplementedError()

    def _run_on_terminate_callbacks(self):
        for cb in self.on_terminate_callbacks:
            try:
                cb()
            except Exception as e:
                _LOG.warn('on_terminate_callback raised exception', exc_info=True)


class GdbDebugger(Debugger):
    """Handles launching, suspending signals, and potentially dealing with terminal issues."""

    # Number of seconds to wait in stop() for a graceful shutdown. After this time has elapsed,
    # the debugger is kill()'d.
    _GRACEFUL_SHUTDOWN_TIMEOUT_SEC = 5.0

    # The instance of GdbDebugger that's currently started.
    _STARTED_INSTANCE = None

    @classmethod
    def _stop_all(cls):
        if cls._STARTED_INSTANCE:
            cls._STARTED_INSTANCE.stop()

    def __init__(self):
        super(GdbDebugger, self).__init__()
        self._is_running = False

    @abc.abstractmethod
    def popen_kwargs(self):
        raise NotImplementedError()

    @classmethod
    def _sigint_handler(cls, signum, stack_frame):
        print('sigint handler')
        if cls._STARTED_INSTANCE is not None:
            try:
                print('kill subp', cls._STARTED_INSTANCE.child_pgid)
                os.killpg(cls._STARTED_INSTANCE.child_pgid, signal.SIGINT)
                return
            except ProcessLookupError as e:
                pass

        raise KeyboardInterrupt()

    def start(self):
        assert not self._is_running
        assert not self._STARTED_INSTANCE

        kwargs = self.popen_kwargs()
        self.did_start_new_session = (
            kwargs.setdefault('start_new_session', True))

        self.old_termios = termios.tcgetattr(sys.stdin.fileno())
        self.old_sigint_handler = signal.signal(signal.SIGINT, self._sigint_handler)
        self.popen = subprocess.Popen(**kwargs)
        self._is_running = True
        self.__class__._STARTED_INSTANCE = self
        try:
            self.child_pgid = os.getpgid(self.popen.pid)
        except Exception as e:
            self.stop()
            raise

    def _wait_til_pgexit(self):
        end_time = time.monotonic() + self._GRACEFUL_SHUTDOWN_TIMEOUT_SEC
        while time.monotonic() < end_time:
            try:
                ret = os.waitid(os.P_PGID, self.child_pgid, os.WEXITED | os.WNOHANG)
            except OSError as e:
                if e.errno == errno.EINVAL or e.errno == errno.ECHILD:
                    return True

                elif e.errno == errno.EINTR:
                    pass

                else:
                    raise

            time.sleep(0.1)

        return False

    def stop(self):
        assert self._is_running
        signal.signal(signal.SIGINT, self.old_sigint_handler)
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSAFLUSH, self.old_termios)

        try:
            os.killpg(self.child_pgid, signal.SIGTERM)
            if not self._wait_til_pgexit():
                os.killpg(self.child_pgid, signal.SIGKILL)
        except ProcessLookupError:
            _LOG.warn('error', exc_info=True)
            pass

        self._STARTED_INSTANCE = None
        self._is_running = False
        self._run_on_terminate_callbacks()


atexit.register(GdbDebugger._stop_all)


class GdbTransportDebugger(GdbDebugger):
    """A debugger that uses a single GDB subprocess as both the transport and the debugger.

    Opens pipes for the target's stdin and stdout, launches GDB and configures GDB's target
    arguments to read and write from the pipes using /dev/fd.
    """

    def __init__(self, args, **popen_kw):
        super(GdbTransportDebugger, self).__init__()
        self.args = args
        self.popen_kw = popen_kw

    def popen_kwargs(self):
        stdin_read, stdin_write = os.pipe()
        stdout_read, stdout_write = os.pipe()

        os.set_inheritable(stdin_read, True)
        os.set_inheritable(stdout_write, True)

        sysname = os.uname()[0]
        if sysname == "Darwin":
            args = [
                "lldb",
                "-O",
                f"target create {self.args[0]}",
                "-O",
                f"settings set target.input-path /dev/fd/{stdin_read}",
                "-O",
                f"settings set target.output-path /dev/fd/{stdout_write}",
            ]
            if len(self.args) > 1:
                args.extend(
                    ["-O", "settings set target.run-args {}".format(" ".join(self.args[1:]))]
                )
        elif sysname == "Linux":
            args = (
                ["gdb", "--args"] + self.args + ["</dev/fd/{stdin_read}", ">/dev/fd/{stdout_write}"]
            )
        else:
            raise NotImplementedError(f"System {sysname} is not yet supported")

        self.fd_transport = fd.FdTransport(stdout_read, stdin_write)
        self.fd_transport.open()

        return {
            "args": args,
            "pass_fds": [stdin_read, stdout_write],
        }

    def _wait_for_process_death(self):
        self.popen.wait()
        self.fd_transport.close()

    def start(self):
        to_return = super(GdbTransportDebugger, self).Start()
        threading.Thread(target=self._wait_for_process_death, daemon=True).start()
        return to_return

    def stop(self):
        self.fd_transport.close()
        super(GdbTransportDebugger, self).Stop()

    class _Transport(transport.Transport):
        def __init__(self, gdb_transport_debugger):
            self.gdb_transport_debugger = gdb_transport_debugger

        def timeouts(self):
            return transport.debug_transport_timeouts()

        def open(self):
            pass  # Pipes opened by parent class.

        def write(self, data, timeout_sec):
            return self.gdb_transport_debugger.fd_transport.write(data, timeout_sec)

        def read(self, n, timeout_sec):
            return self.gdb_transport_debugger.fd_transport.read(n, timeout_sec)

        def close(self):
            pass  # Pipes closed by parent class.

    def transport(self):
        return self._Transport(self)


class GdbRemoteDebugger(GdbDebugger):
    """A Debugger that invokes GDB and attaches to a remote GDBserver-based target."""

    def __init__(
        self, gdb_binary, remote_hostport, debug_binary, wrapping_context_manager=None, **popen_kw
    ):
        super(GdbRemoteDebugger, self).__init__()
        self.gdb_binary = gdb_binary
        self.remote_hostport = remote_hostport
        self.debug_binary = debug_binary
        self.wrapping_context_manager = wrapping_context_manager
        self.popen_kw = popen_kw

    def popen_kwargs(self):
        kwargs = {
            "args": [
                self.gdb_binary,
                "-iex",
                f"file {self.debug_binary}",
                "-iex",
                f"target remote {self.remote_hostport}",
            ],
        }
        kwargs.update(self.popen_kw)

        return kwargs

    def start(self):
        if self.wrapping_context_manager is not None:
            self.wrapping_context_manager.__enter__()
        super(GdbRemoteDebugger, self).start()

    def stop(self):
        try:
            super(GdbRemoteDebugger, self).stop()
        finally:
            if self.wrapping_context_manager is not None:
                self.wrapping_context_manager.__exit__(None, None, None)


GLOBAL_DEBUGGER = None


class DebuggerFactory(class_factory.ClassFactory):

  SUPERCLASS = Debugger


def launch_debugger(debugger_factory, *args, **kw):
  global GLOBAL_DEBUGGER
  if GLOBAL_DEBUGGER is not None:
    stop_debugger()

  GLOBAL_DEBUGGER = debugger_factory.instantiate()
  GLOBAL_DEBUGGER.start()


@register_func("tvm.micro.debugger.launch_debugger")
def _launch_debugger(debugger_factory_json):
  launch_debugger(DebuggerFactory.from_json(debugger_factory_json))


@register_func("tvm.micro.debugger.stop_debugger")
def stop_debugger():
  global GLOBAL_DEBUGGER
  if GLOBAL_DEBUGGER is not None:
    try:
      GLOBAL_DEBUGGER.stop()
    finally:
      GLOBAL_DEBUGGER = None


class RpcDebugger(Debugger):

    def __init__(self, rpc_session, factory, wrapping_context_manager=None):
        super(RpcDebugger, self).__init__()
        self._factory = factory
        self.launch_debugger = rpc_session.get_function('tvm.micro.debugger.launch_debugger')
        self.stop_debugger = rpc_session.get_function('tvm.micro.debugger.stop_debugger')
        self.wrapping_context_manager = wrapping_context_manager

    def start(self):
        if self.wrapping_context_manager is not None:
            self.wrapping_context_manager.__enter__()

        try:
            self.launch_debugger(self._factory.to_json)
        except Exception as e:
            if self.wrapping_context_manager is not None:
                self.wrapping_context_manager.__exit__(None, None, None)
            raise

        try:
            input('Press [Enter] when debugger is set')
        except Exception as e:
            self.stop()
            raise

        self._is_running = True

    def stop(self):
      try:
        self.stop_debugger()
        self._run_on_terminate_callbacks()
      finally:
        if self.wrapping_context_manager is not None:
          self.wrapping_context_manager.__exit__(None, None, None)
