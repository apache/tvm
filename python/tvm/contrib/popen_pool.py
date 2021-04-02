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
# pylint: disable=invalid-name
"""Multiprocessing via Popen.

This module provides a multi-processing pool backed by Popen.
with additional timeout support.
"""
import os
import sys
import struct
import threading
import subprocess
import concurrent.futures
from enum import IntEnum
from collections import namedtuple
import pickle


def kill_child_processes(pid):
    """Kill all child processes recursively for a given pid.

    Parameters
    ----------
    pid : int
        The given parameter id.
    """
    # pylint: disable=import-outside-toplevel
    import psutil

    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    for process in parent.children(recursive=True):
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass


class StatusKind(IntEnum):
    """Running and return value status."""

    RUNNING = 0
    COMPLETE = 1
    EXCEPTION = 2
    TIMEOUT = 3


class MapResult(namedtuple("MapResult", ["status", "value"])):
    """Result of map_with_error_catching.

    Parameters
    ----------
    status : StatusKind
        The status of the result.

    value : Any
        The result value.
    """

    __slots__ = []


class PopenWorker:
    """A subprocess worker via Popen.

    PopenWorker provides a low-level
    API to interact with a separate process via Popen.
    """

    def __init__(self):
        self._proc = None

    def __del__(self):
        try:
            self.kill()
        except ImportError:
            pass

    def kill(self):
        """Kill the current running process and cleanup.

        Note
        ----
        The worker can start a new process when send is called again.
        """
        if self._proc is not None:
            # allow gracefully shutdown
            try:
                self._writer.close()
            except IOError:
                pass
            try:
                self._reader.close()
            except IOError:
                pass
            # kill all child processes recurisvely
            try:
                kill_child_processes(self._proc.pid)
            except TypeError:
                pass
            try:
                self._proc.kill()
            except OSError:
                pass
            self._proc = None

    def _start(self):
        """Start a new subprocess if nothing is available"""
        if self._proc is not None:
            return

        # connect subprocess with a pair of pipes
        main_read, worker_write = os.pipe()
        worker_read, main_write = os.pipe()

        cmd = [sys.executable, "-m", "tvm.exec.popen_worker"]
        if sys.platform == "win32":
            # pylint: disable=import-outside-toplevel
            import msvcrt

            worker_read_handle = msvcrt.get_osfhandle(worker_read)
            worker_write_handle = msvcrt.get_osfhandle(worker_write)
            os.set_handle_inheritable(worker_read_handle, True)
            os.set_handle_inheritable(worker_write_handle, True)
            cmd += [str(worker_read_handle), str(worker_write_handle)]
            self._proc = subprocess.Popen(cmd, close_fds=False)
        else:
            cmd += [str(worker_read), str(worker_write)]
            self._proc = subprocess.Popen(cmd, pass_fds=(worker_read, worker_write))

        # close worker side of the pipe
        os.close(worker_read)
        os.close(worker_write)
        self._reader = os.fdopen(main_read, "rb")
        self._writer = os.fdopen(main_write, "wb")

    def join(self):
        """Join the current process worker before it terminates"""
        if self._proc:
            self._proc.wait()

    def send(self, fn, args=(), kwargs=None, timeout=None):
        """Send a new function task fn(*args, **kwargs) to the subprocess.

        Parameters
        ----------
        fn : function
            The function to be invoked.

        args : list
            Positional argument.

        kwargs : dict
            Keyword arguments

        timeout : float
            Timeout value when executing the function

        Note
        ----
        The caller must call recv before calling the next send in
        order to make sure the timeout and child process exit
        won't affect the later requests.
        """
        # use cloud pickle
        # pylint: disable=import-outside-toplevel
        import cloudpickle

        if self._proc is None:
            self._start()
        kwargs = {} if not kwargs else kwargs
        data = cloudpickle.dumps((fn, args, kwargs, timeout), protocol=pickle.HIGHEST_PROTOCOL)
        try:
            self._writer.write(struct.pack("<i", len(data)))
            self._writer.write(data)
            self._writer.flush()
        except IOError:
            pass

    def _child_process_error(self):
        """Raise a child process error."""
        # kill and lazily restart the process in the next send.
        self.kill()
        return ChildProcessError("Subprocess terminated")

    def recv(self):
        """Receive the result of the last send.

        Returns
        -------
        result: object
            The result of the last send.

        Raises
        ------
        ChildProcessError: if the child process exited abnormally.
        TimeoutError: if timeout happens
        Exception: if other exception happens during the execution.
        """
        # pylint: disable=import-outside-toplevel
        import cloudpickle

        try:
            len_data = self._reader.read(4)
        except IOError:
            raise self._child_process_error()

        if len(len_data) == 0:
            raise self._child_process_error()

        try:
            recv_bytes = struct.unpack("<i", len_data)[0]
            status, value = cloudpickle.loads(self._reader.read(recv_bytes))
        except IOError:
            raise self._child_process_error()

        if status == StatusKind.COMPLETE:
            return value
        if status == StatusKind.EXCEPTION:
            raise value
        assert status == StatusKind.TIMEOUT
        # kill and lazily restart the process in the next send.
        self.kill()
        raise TimeoutError()


class PopenPoolExecutor:
    """An parallel executor backed by Popen processes.

    Parameters
    ----------
    max_worker : int
        Maximum number of workers

    timeout : float
        Timeout value for each function submit.
    """

    def __init__(self, max_workers, timeout=None):
        # Use an internal thread pool to send to popen workers
        self._threadpool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._timeout = timeout
        self._worker_map = {}
        self._lock = threading.Lock()

    def __del__(self):
        self._lock.acquire()
        for worker in self._worker_map.values():
            try:
                worker.kill()
            except ImportError:
                pass
        self._lock.release()
        self._threadpool.shutdown()

    def _worker_run(self, fn, args, kwargs):
        """Internal thread runner."""
        self._lock.acquire()
        tid = threading.get_ident()
        if tid not in self._worker_map:
            proc = PopenWorker()
            self._worker_map[tid] = proc
        else:
            proc = self._worker_map[tid]
        self._lock.release()

        proc.send(fn, args, kwargs, self._timeout)
        return proc.recv()

    def _worker_run_with_error_catching(self, fn, args, kwargs) -> MapResult:
        # pylint: disable=broad-except
        try:
            return MapResult(status=StatusKind.COMPLETE, value=self._worker_run(fn, args, kwargs))
        except TimeoutError as exception:
            return MapResult(status=StatusKind.TIMEOUT, value=exception)
        except Exception as exception:
            return MapResult(status=StatusKind.EXCEPTION, value=exception)

    def submit(self, fn, *args, **kwargs) -> concurrent.futures.Future:
        """Submit a new function job to the pool

        Parameters
        ----------
        fn : function
            The function to be invoked.

        args : list
            Positional argument.

        kwargs : dict
            Keyword arguments

        Returns
        -------
        future : concurrent.futures.Future
            A future that can be used to access the result.
        """
        # pylint: disable=unnecessary-lambda
        worker = lambda *args: self._worker_run(*args)
        return self._threadpool.submit(worker, fn, args, kwargs)

    def map_with_error_catching(self, fn, iterator):
        """Same as map, but catches exceptions and return them instead.

        Parameters
        ----------
        fn : function
            The function to be invoked.

        iterator : Iterator
            Input iterator.

        Returns
        -------
        out_iter : Iterator[MapResult]
            The result iterator.
        """
        worker = lambda x: self._worker_run_with_error_catching(fn, (x,), None)
        return self._threadpool.map(worker, iterator)
