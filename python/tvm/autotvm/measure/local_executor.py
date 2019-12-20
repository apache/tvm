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
"""Local based implementation of the executor using multiprocessing"""

import signal
import os

if os.name == 'nt':
    import queue as thread_queue
    import threading
    # Pathos uses dill, which can pickle things like functions
    from pathos.helpers import ProcessPool
    # On Windows, there is no fork(), a 'multiprocessing.Process'
    # or ProcessPool has to 'build up' the script from scratch, we
    # set these environment variables so each python.exe process
    # does not allocate unneeded threads
    os.environ['OMP_NUM_THREADS'] = "1"
    os.environ['TVM_NUM_THREADS'] = "1"
    # numpy seems to honor this
    os.environ['MKL_NUM_THREADS'] = "1"

    # Since there is no fork() on Windows, to mitigate performance impact
    # we will use a process pool for executers, vs the *nix based systems
    # that will fork() a new process for each executor
    EXECUTOR_POOL = None

#pylint: disable=wrong-import-position
#pylint: disable=ungrouped-imports
from multiprocessing import Process, Queue, cpu_count
try:
    from queue import Empty
except ImportError:
    from Queue import Empty

try:
    import psutil
except ImportError:
    psutil = None

from . import executor
#pylint: enable=ungrouped-imports
#pylint: enable=wrong-import-position

def kill_child_processes(parent_pid, sig=signal.SIGTERM):
    """kill all child processes recursively"""
    try:
        parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return
    children = parent.children(recursive=True)
    for process in children:
        try:
            process.send_signal(sig)
        except psutil.NoSuchProcess:
            return

def _execute_func(func, queue, args, kwargs):
    """execute function and return the result or exception to a queue"""
    try:
        res = func(*args, **kwargs)
    except Exception as exc:  # pylint: disable=broad-except
        res = exc
    queue.put(res)


def call_with_timeout(queue, timeout, func, args, kwargs):
    """A wrapper to support timeout of a function call"""

    # start a new process for timeout (cannot use thread because we have c function)
    p = Process(target=_execute_func, args=(func, queue, args, kwargs))
    p.start()
    p.join(timeout=timeout)

    queue.put(executor.TimeoutError())

    kill_child_processes(p.pid)
    p.terminate()
    p.join()

if os.name == 'nt':
    def call_from_pool(func, args, kwargs, timeout, env): # pylint: disable=unused-argument
        """A wrapper to support timeout of a function call for a pool process"""

        # Restore environment variables from parent
        for key, val in env.items():
            os.environ[key] = val

        queue = thread_queue.Queue(2)

        # We use a thread here for Windows, because starting up a new Process can be heavy
        # This isn't as clean as the *nix implementation, which can kill a process that
        # has timed out
        thread = threading.Thread(target=_execute_func, args=(func, queue, args, kwargs))
        thread.start()
        thread.join()
        queue.put(executor.TimeoutError())

        res = queue.get()
        return res

class LocalFuture(executor.Future):
    """Local wrapper for the future

    Parameters
    ----------
    process: multiprocessing.Process
        process for running this task
    queue: multiprocessing.Queue
        queue for receiving the result of this task
    """
    def __init__(self, process, queue):
        self._done = False
        self._process = process
        self._queue = queue

    def done(self):
        self._done = self._done or not self._queue.empty()
        return self._done

    def get(self, timeout=None):
        try:
            res = self._queue.get(block=True, timeout=timeout)
        except Empty:
            raise executor.TimeoutError()
        if self._process.is_alive():
            kill_child_processes(self._process.pid)
            self._process.terminate()
        self._process.join()
        self._queue.close()
        self._queue.join_thread()
        self._done = True
        del self._queue
        del self._process
        return res


class LocalFutureNoFork(executor.Future):
    """Local wrapper for the future.
    This is a none-fork version of LocalFuture.
    Use this for the runtime that does not support fork (like cudnn)
    """
    def __init__(self, result):
        self._result = result

    def done(self):
        return True

    def get(self, timeout=None):
        return self._result

if os.name == 'nt':
    class LocalFuturePool(executor.Future):
        """Local wrapper for the future using a Process pool

        Parameters
        ----------
        thread: threading.Thread
            Thread for running this task
        pool_results: result from Pool.apply_async
            queue for receiving the result of this task
        """
        def __init__(self, pool_results):
            self._done = False
            self._pool_results = pool_results

        def done(self):
            return self._done

        def get(self, timeout=None):
            try:
                res = self._pool_results.get()
            except Empty:
                raise executor.TimeoutError()
            self._done = True
            return res

class LocalExecutor(executor.Executor):
    """Local executor that runs workers on the same machine with multiprocessing.

    Parameters
    ----------
    timeout: float, optional
        timeout of a job. If time is out. A TimeoutError will be returned (not raised)
    do_fork: bool, optional
        For some runtime systems that do not support fork after initialization
        (e.g. cuda runtime, cudnn). Set this to False if you have used these runtime
        before submitting jobs.
    """
    def __init__(self, timeout=None, do_fork=True):
        self.timeout = timeout or executor.Executor.DEFAULT_TIMEOUT
        self.do_fork = do_fork

        if self.do_fork:
            if not psutil:
                raise RuntimeError("Python package psutil is missing. "
                                   "please try `pip install psutil`")

    def submit(self, func, *args, **kwargs):
        if not self.do_fork:
            return LocalFutureNoFork(func(*args, **kwargs))

        if os.name != 'nt':
            queue = Queue(2)
            process = Process(target=call_with_timeout,
                              args=(queue, self.timeout, func, args, kwargs))
            process.start()
            return LocalFuture(process, queue)

        global EXECUTOR_POOL

        if EXECUTOR_POOL is None:
            # We use a static pool for executor processes because Process.start(entry)
            # is so slow on Windows, we lose a lot of parallelism.
            # Right now cpu_count() is used, which isn't optimal from a user configuration
            # perspective, but is reasonable at this time.
            EXECUTOR_POOL = ProcessPool(cpu_count() * 2)

        # Windows seemed to be missing some valuable environ variables
        # on the pool's process side.  We might be able to get away with
        # just sending the PATH variable, but for now, we just clone our env
        return LocalFuturePool(EXECUTOR_POOL.apply_async(call_from_pool,
                                                         (func, args, kwargs,
                                                          self.timeout, os.environ.copy())))
