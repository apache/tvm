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

import abc
import os
import signal
import subprocess
import threading

from . import transport


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


class GdbDebugger(Debugger):
    """Handles launching, suspending signals, and potentially dealing with terminal issues."""

    @abc.abstractmethod
    def popen_kwargs(self):
        raise NotImplementedError()

    def _wait_restore_signal(self):
        self.popen.wait()
        if not self.did_terminate.is_set():
            for callback in self.on_terminate_callbacks:
                try:
                    callback()
                except Exception:  # pylint: disable=broad-except
                    logging.warn("on_terminate_callback raised exception", exc_info=True)

    def start(self):
        kwargs = self.popen_kwargs()
        self.did_terminate = threading.Event()
        self.old_signal = signal.signal(signal.SIGINT, signal.SIG_IGN)
        self.popen = subprocess.Popen(**kwargs)
        threading.Thread(target=self._wait_restore_signal).start()

    def stop(self):
        self.did_terminate.set()
        self.popen.terminate()
        signal.signal(signal.SIGINT, self.old_signal)


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
        to_return = super(GdbTransportDebugger, self).start()
        threading.Thread(target=self._wait_for_process_death, daemon=True).start()
        return to_return

    def stop(self):
        self.fd_transport.close()
        super(GdbTransportDebugger, self).stop()

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
