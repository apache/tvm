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
"""Pipe worker for multi-processing."""
import os
import subprocess
import sys

import psutil

from tvm._ffi import register_func
from tvm.runtime import ShapeTuple


class DiscoPopenWorker:
    """A subprocess worker via Popen.

    PopenWorker provides a low-level
    API to interact with a separate process via Popen.

    Parameters
    ----------
    worker_id : int
        The worker id of the current worker.

    num_workers : int
        The total number of workers.

    stdout: Union[None, int, IO[Any]]
        The standard output streams handler specified for the popen process.

    stderr: Union[None, int, IO[Any]]
        The standard error streams handler specified for the popen process.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        worker_id: int,
        num_workers: int,
        entrypoint: str = "tvm.exec.disco_worker",
        stdout=None,
        stderr=None,
    ):
        self.worker_id = worker_id
        self.num_workers = num_workers
        self.entrypoint = entrypoint
        self._proc = None
        self._stdout = stdout
        self._stderr = stderr

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
            # kill all child processes recursively
            try:
                _kill_child_processes(self._proc.pid)
            except TypeError:
                pass
            try:
                self._proc.kill()
            except OSError:
                pass

            # Join the child process to avoid zombie processes
            self.join(timeout=1.0)
            self._proc = None

    def join(self, timeout=None):
        """Join the current process worker before it terminates.

        Parameters
        ----------
        timeout: Optional[number]
            Timeout value, block at most timeout seconds if it
            is a positive number.
        """
        if self._proc:
            try:
                self._proc.wait(timeout)
            except subprocess.TimeoutExpired:
                pass

    def start(self):
        """Start a new subprocess if nothing is available"""
        if self._proc is not None:
            return None, None

        # connect subprocess with a pair of pipes
        main_read, worker_write = os.pipe()
        worker_read, main_write = os.pipe()

        cmd = [
            sys.executable,
            "-m",
            self.entrypoint,
            str(self.worker_id),
            str(self.num_workers),
        ]
        if sys.platform == "win32":
            import msvcrt  # pylint: disable=import-error,import-outside-toplevel

            worker_read_handle = msvcrt.get_osfhandle(worker_read)
            worker_write_handle = msvcrt.get_osfhandle(worker_write)
            os.set_handle_inheritable(worker_read_handle, True)
            os.set_handle_inheritable(worker_write_handle, True)
            cmd += [str(worker_read_handle), str(worker_write_handle)]
            self._proc = subprocess.Popen(
                cmd,
                close_fds=False,
                stdout=self._stdout,
                stderr=self._stderr,
            )
        else:
            cmd += [str(worker_read), str(worker_write)]
            self._proc = subprocess.Popen(  # pylint: disable=consider-using-with
                cmd,
                pass_fds=(worker_read, worker_write),
                stdout=self._stdout,
                stderr=self._stderr,
            )

        # close worker side of the pipe
        os.close(worker_read)
        os.close(worker_write)
        return main_read, main_write


def _kill_child_processes(pid):
    """Kill all child processes recursively for a given pid.

    Parameters
    ----------
    pid : int
        The given parameter id.
    """
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        return

    for process in children:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass


@register_func("runtime.disco.create_process_pool")
def _create_process_pool(num_workers: int, entrypoint: str):
    """Create a process pool where the workers' are [1, num_workers)."""
    pool = [DiscoPopenWorker(i, num_workers, entrypoint) for i in range(1, num_workers)]

    def result_func(worker_id: int):
        nonlocal pool
        if worker_id != 0:
            read_fd, write_fd = pool[worker_id - 1].start()
            return ShapeTuple([read_fd, write_fd])
        del pool
        return None

    return result_func
