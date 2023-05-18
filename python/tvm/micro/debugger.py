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
# pylint: disable=consider-using-with

"""Defines functions for controlling debuggers for micro TVM binaries."""

import atexit
import abc
import errno
import logging
import os
import shlex
import signal
import subprocess
import sys
import termios
import threading
import time

import psutil

from .._ffi import register_func
from . import class_factory
from . import transport
from .transport.file_descriptor import FdTransport


_LOG = logging.getLogger(__name__)


class Debugger(metaclass=abc.ABCMeta):
    """An interface for controlling micro TVM debuggers."""

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
        self._is_running_lock = threading.RLock()
        self._child_exited_event = threading.Event()
        self._signals_reset_event = threading.Event()

    @abc.abstractmethod
    def popen_kwargs(self):
        raise NotImplementedError()

    def _internal_stop(self):
        if not self._is_running:
            return

        os.kill(os.getpid(), signal.SIGUSR1)
        self._signals_reset_event.wait()
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSAFLUSH, self.old_termios)

        try:
            children = psutil.Process(self.popen.pid).children(recursive=True)
            for c in children:
                c.terminate()
                _, alive = psutil.wait_procs(children, timeout=self._GRACEFUL_SHUTDOWN_TIMEOUT_SEC)
                for a in alive:
                    a.kill()
        except psutil.NoSuchProcess:
            pass
        finally:
            self.__class__._STARTED_INSTANCE = None
            self._is_running = False
            self._child_exited_event.set()

    def _wait_for_child(self):
        self.popen.wait()
        with self._is_running_lock:
            self._internal_stop()

    @classmethod
    def _sigusr1_handler(cls, signum, stack_frame):  # pylint: disable=unused-argument
        assert (
            cls._STARTED_INSTANCE is not None
        ), "overridden sigusr1 handler should not be invoked when GDB not started"
        signal.signal(signal.SIGINT, cls._STARTED_INSTANCE.old_sigint_handler)
        signal.signal(signal.SIGUSR1, cls._STARTED_INSTANCE.old_sigusr1_handler)
        cls._STARTED_INSTANCE._signals_reset_event.set()

    @classmethod
    def _sigint_handler(cls, signum, stack_frame):  # pylint: disable=unused-argument
        assert (
            cls._STARTED_INSTANCE is not None
        ), "overridden sigint handler should not be invoked when GDB not started"
        with cls._STARTED_INSTANCE._is_running_lock:
            exists = cls._STARTED_INSTANCE._is_running
        if exists:
            try:
                os.killpg(cls._STARTED_INSTANCE.child_pgid, signal.SIGINT)
            except ProcessLookupError:
                pass

    def start(self):
        with self._is_running_lock:
            assert not self._is_running
            assert not self._STARTED_INSTANCE

            kwargs = self.popen_kwargs()
            self.did_start_new_session = kwargs.setdefault("start_new_session", True)

            self.old_termios = termios.tcgetattr(sys.stdin.fileno())
            self.popen = subprocess.Popen(**kwargs)
            self._is_running = True
            self.old_sigint_handler = signal.signal(signal.SIGINT, self._sigint_handler)
            self.old_sigusr1_handler = signal.signal(signal.SIGUSR1, self._sigusr1_handler)
            self.__class__._STARTED_INSTANCE = self
            try:
                self.child_pgid = os.getpgid(self.popen.pid)
            except Exception:
                self.stop()
                raise
            with self._is_running_lock:
                self._is_child_alive = True
            t = threading.Thread(target=self._wait_for_child)
            t.daemon = True
            t.start()

    def stop(self):
        self._child_exited_event.wait()


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
            args = [
                "gdb",
                "-ex",
                f"file {self.args[0]}",
                "-ex",
                (
                    f"set args {' '.join(shlex.quote(a) for a in self.args[1:])} "
                    f"</dev/fd/{stdin_read} >/dev/fd/{stdout_write}"
                ),
            ]
        else:
            raise NotImplementedError(f"System {sysname} is not yet supported")

        self.fd_transport = FdTransport(
            stdout_read, stdin_write, transport.debug_transport_timeouts()
        )
        self.fd_transport.open()

        return {
            "args": args,
            "pass_fds": [stdin_read, stdout_write],
        }

    def _internal_stop(self):
        self.fd_transport.close()
        super(GdbTransportDebugger, self)._internal_stop()

    class _Transport(transport.Transport):
        def __init__(self, gdb_transport_debugger):
            self.gdb_transport_debugger = gdb_transport_debugger

        def timeouts(self):
            return transport.debug_transport_timeouts()

        def open(self):
            pass  # Pipes opened by parent class.

        def write(self, data, timeout_sec):
            end_time = time.monotonic() + timeout_sec if timeout_sec is not None else None
            while True:
                try:
                    return self.gdb_transport_debugger.fd_transport.write(data, timeout_sec)
                except OSError as exc:
                    # NOTE: this error sometimes happens when writes are initiated before the child
                    # process launches.
                    if exc.errno == errno.EAGAIN:
                        if end_time is None or time.monotonic() < end_time:
                            time.sleep(0.1)  # sleep to avoid excessive CPU usage
                            continue

                    raise exc

            raise base.IoTimeoutError()

        def read(self, n, timeout_sec):
            end_time = time.monotonic() + timeout_sec if timeout_sec is not None else None
            while True:
                try:
                    return self.gdb_transport_debugger.fd_transport.read(n, timeout_sec)
                except OSError as exc:
                    # NOTE: this error sometimes happens when reads are initiated before the child
                    # process launches.
                    if exc.errno == errno.EAGAIN:
                        if end_time is None or time.monotonic() < end_time:
                            time.sleep(0.1)  # sleep to avoid excessive CPU usage
                            continue

                    raise exc

            raise base.IoTimeoutError()

        def close(self):
            pass  # Pipes closed by parent class (DebugWrapperTransport calls stop() next).

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

    GLOBAL_DEBUGGER = debugger_factory.instantiate(*args, **kw)
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
    """A Debugger instance that launches the actual debugger on a remote TVM RPC server."""

    def __init__(self, rpc_session, factory, wrapping_context_manager=None):
        super(RpcDebugger, self).__init__()
        self._factory = factory
        self.launch_debugger = rpc_session.get_function("tvm.micro.debugger.launch_debugger")
        self.stop_debugger = rpc_session.get_function("tvm.micro.debugger.stop_debugger")
        self.wrapping_context_manager = wrapping_context_manager

    def start(self):
        if self.wrapping_context_manager is not None:
            self.wrapping_context_manager.__enter__()

        try:
            self.launch_debugger(self._factory.to_json)
        except Exception:
            if self.wrapping_context_manager is not None:
                self.wrapping_context_manager.__exit__(None, None, None)
            raise

        try:
            input("Press [Enter] when debugger is set")
        except Exception:
            self.stop()
            raise

        self._is_running = True

    def stop(self):
        try:
            self.stop_debugger()
        finally:
            if self.wrapping_context_manager is not None:
                self.wrapping_context_manager.__exit__(None, None, None)
