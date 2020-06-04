"""Common utilities"""
import multiprocessing
import multiprocessing.pool
import queue
import signal
import threading
import os

import numpy as np

try:
    import psutil
except ImportError:
    psutil = None

from .. import rpc as _rpc
from tvm.tir import expr
from tvm.tir.transform import Simplify
from tvm.ir.transform import Sequential


def get_func_name(func):
    """Get name of a function

    Parameters
    ----------
    func: Function
        The function
    Returns
    -------
    name: str
        The name
    """

    return func.func_name if hasattr(func, 'func_name') else func.__name__


def get_const_int(exp):
    """Verifies expr is integer and get the constant value.

    Parameters
    ----------
    exp : tvm.Expr or int
        The input expression.

    Returns
    -------
    out_value : int
        The output.
    """
    if isinstance(exp, int):
        return exp
    if not isinstance(exp, (expr.IntImm)):
        opt = Sequential([Simplify()])
        exp = opt(exp)
    if not isinstance(exp, (expr.IntImm)):
        raise ValueError("Expect value to be constant int")
    return exp.value


def get_const_tuple(in_tuple):
    """Verifies input tuple is IntImm, returns tuple of int.

    Parameters
    ----------
    in_tuple : tuple of Expr
        The input.

    Returns
    -------
    out_tuple : tuple of int
        The output.
    """
    return tuple(get_const_int(x) for x in in_tuple)


def to_str_round(x, decimal=6):
    """Convert object to str and round float numbers"""
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)) or isinstance(x, np.ndarray):
        return "[" + ", ".join([to_str_round(y, decimal=decimal)
                                for y in x]) + "]"
    if isinstance(x, dict):
        return str({k: eval(to_str_round(v)) for k, v in x.items()})
    if isinstance(x, int):
        return str(x)
    if isinstance(x, (np.float32, np.float64, float)):
        format_str = "%%.%df" % decimal
        return format_str % x
    raise ValueError("Invalid value: " + str(x) + "\ttype: " + str(type(x)))


def array_mean(arr):
    """Mean function for tvm array (Array<Expr>)"""
    return sum(x.value for x in arr) / len(arr)


class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess


class NoDaemonPool(multiprocessing.pool.Pool):
    """A no daemon pool version of multiprocessing.Pool.
    This allows us to start new processings inside the worker function"""

    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super().__init__(*args, **kwargs)


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


def call_func_with_timeout(timeout, func, args=(), kwargs=None):
    """Call a function with timeout"""
    def func_wrapper(que):
        if kwargs:
            que.put(func(*args, **kwargs))
        else:
            que.put(func(*args))

    que = multiprocessing.Queue(2)
    process = multiprocessing.Process(target=func_wrapper, args=(que,))
    process.start()
    process.join(timeout)

    try:
        res = que.get(block=False)
    except queue.Empty:
        res = TimeoutError()

    # clean queue and process
    kill_child_processes(process.pid)
    process.terminate()
    process.join()
    que.close()
    que.join_thread()
    del process
    del que

    return res


def request_remote(device_key, host=None, port=None, priority=1, timeout=60):
    """Request a remote session

    Parameters
    ----------
    device_key: string
        The device key of registered device in tracker
    host: host, optional
        The host address of rpc tracker.
        If is none, will use environment variable "TVM_TRACKER_HOST"
    port: int, optional
        The port of rpc tracker.
        If is none, will use environment variable "TVM_TRACKER_PORT"
    priority: int, optional
        The priority of this request, larger is more prior
    timeout: float, optional
        The timeout of this session (units: second)

    Returns
    ------
    session: RPCSession
    """
    # connect to the tracker
    host = host or os.environ['TVM_TRACKER_HOST']
    port = port or int(os.environ['TVM_TRACKER_PORT'])

    tracker = _rpc.connect_tracker(host, port)
    remote = tracker.request(device_key, priority=priority,
                             session_timeout=timeout)
    return remote


def check_remote(device_key, host=None, port=None, priority=100, timeout=10):
    """
    Check the availability of a remote device

    Parameters
    ----------
    device_key: string
        device key of registered device in tracker
    host: host, optional
        The host address of rpc tracker.
        If is none, will use environment variable "TVM_TRACKER_HOST"
    port: int, optional
        The port address of rpc tracker.
        If is none, will use environment variable "TVM_TRACKER_PORT"
    priority: int, optional
        The priority of this request, larger is more prior
    timeout: float, optional
        The timeout of this check (units: seconds).

    Returns
    -------
    available: bool
        True if can find available device
    """

    def _check():
        remote = request_remote(device_key, host, port, priority)

    t = threading.Thread(target=_check, )
    t.start()
    t.join(timeout)
    return not t.is_alive()
