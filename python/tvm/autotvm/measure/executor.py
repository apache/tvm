""" Abstraction for asynchronous job execution """

class Executor(object):
    """
    Base abstract executor interface for asynchronous job submission.
    Allows submit asynchronous jobs and returns the Future object.
    """
    # timeout for jobs that may hang
    DEFAULT_TIMEOUT = 60

    def submit(self, func, *args, **kwargs):
        """
        Pass task (function, arguments) to the Executor.

        Parameters
        ----------
        func : callable
            function to be run by a worker
        args : list or tuple, optional
            arguments passed to the function
        kwargs : dict, optional
            The keyword arguments

        Returns
        -------
        future : Future
            Future object wrapping the task which can be used to
            collect the task's result.
        """
        raise NotImplementedError()


class Future(object):
    """
    Base class of the future object.
    The implementations can return object of subclass of this.
    This objects encapsulates the asynchronous execution of task
    submitted to another thread, or another worker for execution.

    Future objects store the state of tasks--can be polled for
    result or a blocking call to retrieve the result can be used.
    """
    def done(self):
        """
        Return True if job was successfully cancelled or finished running.
        """
        raise NotImplementedError()

    def get(self, timeout=None):
        """
        Get the result. This will block until the result is available.

        Parameters
        ----------
        timeout : int or float, optional
            Maximum number of seconds to wait before it timeouts.
            If not specified, it means we block until the result is available.

        Returns
        -------
        result : Any
            The result returned by the submitted function.

        Raises
        ------
        TimeoutError : if the result call timeouts.
        """
        raise NotImplementedError()

class FutureError(RuntimeError):
    """Base error class of all future events"""
    pass

# pylint:disable=redefined-builtin
class TimeoutError(FutureError):
    """Error raised when a task is timeout."""
    pass

class ExecutionError(FutureError):
    """
    Error raised when future execution crashes or failed.
    """
    pass
