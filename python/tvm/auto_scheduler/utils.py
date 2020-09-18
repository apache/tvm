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

""" Common utilities for auto_scheduler. """

from typing import Hashable
import multiprocessing
import multiprocessing.pool
import queue
import signal
import threading
import os

try:
    import psutil
except ImportError:
    raise ImportError("psutil not found, try `pip install psutil` to fix this")

from tvm import rpc
from tvm.tir import expr
from tvm.tir.transform import Simplify
from tvm.ir.transform import Sequential
from ..te import Tensor, placeholder


def get_func_name(func):
    """Get name of a function.

    Parameters
    ----------
    func: Function
        The input function.

    Returns
    -------
    name: str
        The function name.
    """
    return func.func_name if hasattr(func, "func_name") else func.__qualname__


def get_const_int(exp):
    """Verifies expr is integer and get the constant value.

    Parameters
    ----------
    exp : Union[tvm.tir.expr, int]
        The input expression.

    Returns
    -------
    out_value : int
        The output.
    """
    if isinstance(exp, int):
        return exp
    if not isinstance(exp, expr.IntImm):
        opt = Sequential([Simplify()])
        exp = opt(exp)
    if not isinstance(exp, expr.IntImm):
        raise ValueError("Expect value to be constant int")
    return exp.value


def get_const_tuple(in_tuple):
    """Verifies input tuple is IntImm, returns tuple of int.

    Parameters
    ----------
    in_tuple : Tuple[tvm.tir.expr]
        The input.

    Returns
    -------
    out_tuple : Tuple[int]
        The output.
    """
    return tuple(get_const_int(x) for x in in_tuple)


def list_to_tuple(x):
    """ Convert a list to a tuple recursively. """
    assert isinstance(x, list)
    return tuple(list_to_tuple(y) if isinstance(y, list) else y for y in x)


def serialize_args(args):
    """
    Serialize arguments of a function to a hashable and jsonable tuple.
    Currently this is mainly used for tvm.tensor.Tensor
    """
    ret = []
    for t in args:
        if isinstance(t, Tensor):
            t = ("TENSOR", get_const_tuple(t.shape), t.dtype)
        elif isinstance(t, list):
            t = list_to_tuple(t)

        assert isinstance(t, Hashable), str(t) + " is not hashable"
        ret.append(t)

    return tuple(ret)


def deserialize_args(args):
    """The inverse function of :code:`serialize_args`"""
    ret = []
    for t in args:
        if isinstance(t, (tuple, list)) and t[0] == "TENSOR":
            ret.append(placeholder(shape=t[1], dtype=t[2]))
        else:
            ret.append(t)
    return ret


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
    This allows us to start new processes inside the worker function"""

    def __init__(self, *args, **kwargs):
        kwargs["context"] = NoDaemonContext()
        super().__init__(*args, **kwargs)

    def __reduce__(self):
        pass


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
    """Request a remote session.

    Parameters
    ----------
    device_key : str
        The device key of registered device in tracker.
    host : Optional[str]
        The host address of rpc tracker.
        If is none, will use environment variable "TVM_TRACKER_HOST".
    port : Optional[int]
        The port of rpc tracker.
        If is none, will use environment variable "TVM_TRACKER_PORT".
    priority : int = 1
        The priority of this request, larger is more prior.
    timeout : int = 60
        The timeout of this session in second.

    Returns
    -------
    remote : RPCSession
        The connected remote RPCSession.
    """
    # connect to the tracker
    host = host or os.environ["TVM_TRACKER_HOST"]
    port = port or int(os.environ["TVM_TRACKER_PORT"])

    tracker = rpc.connect_tracker(host, port)
    remote = tracker.request(device_key, priority=priority, session_timeout=timeout)
    return remote


def check_remote(device_key, host=None, port=None, priority=100, timeout=10):
    """
    Check the availability of a remote device.

    Parameters
    ----------
    device_key: str
        device key of registered device in tracker.
    host: Optional[str]
        The host address of rpc tracker.
        If is none, will use environment variable "TVM_TRACKER_HOST".
    port: Optional[int]
        The port address of rpc tracker.
        If is none, will use environment variable "TVM_TRACKER_PORT".
    priority: int = 100
        The priority of this request, larger is more prior.
    timeout: int = 10
        The timeout of this check in seconds.

    Returns
    -------
    available: bool
        True if can find available device.
    """

    def _check():
        request_remote(device_key, host, port, priority)

    t = threading.Thread(
        target=_check,
    )
    t.start()
    t.join(timeout)
    return not t.is_alive()
