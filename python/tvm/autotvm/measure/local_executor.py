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

try:
    import psutil
except ImportError:
    psutil = None

from . import executor
from ..env import GLOBAL_SCOPE
from ...contrib.popen_pool import PopenPoolExecutor


def kill_child_processes(parent_pid, sig=signal.SIGTERM):
    """kill all child processes recursively"""
    try:
        parent = psutil.Process(parent_pid)
        children = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        return
    for process in children:
        try:
            process.send_signal(sig)
        except psutil.NoSuchProcess:
            return


def _popen_initializer(global_scope):
    global GLOBAL_SCOPE
    GLOBAL_SCOPE = global_scope


class LocalFuture(executor.Future):
    """Local wrapper for the future
    Parameters
    ----------
    future: concurrent.futures.Future
        A future returned by PopenPoolExecutor.
    """

    def __init__(self, future):
        self._done = False
        self._future = future

    def done(self):
        return self._future.done()

    def get(self, timeout=None):
        return self._future.result(timeout)


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
                raise RuntimeError(
                    "Python package psutil is missing. " "please try `pip install psutil`"
                )

    def submit(self, func, *args, **kwargs):
        if not self.do_fork:
            return LocalFutureNoFork(func(*args, **kwargs))

        pool = PopenPoolExecutor(
            timeout=self.timeout, initializer=_popen_initializer, initargs=(GLOBAL_SCOPE,)
        )
        return LocalFuture(pool.submit(func, args, kwargs))
