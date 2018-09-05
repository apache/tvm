"""Local based implementation of the executor using multiprocessing"""

import signal

from multiprocessing import Process, Queue
try:
    from queue import Empty
except ImportError:
    from Queue import Empty

try:
    import psutil
except ImportError:
    psutil = None

from . import executor


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

        queue = Queue(2)
        process = Process(target=call_with_timeout,
                          args=(queue, self.timeout, func, args, kwargs))
        process.start()
        return LocalFuture(process, queue)
