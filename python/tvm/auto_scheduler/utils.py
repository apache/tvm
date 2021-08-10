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

""" Common utilities for auto_scheduler. """

from typing import Hashable
import json
import signal
import threading
import traceback
import os

import numpy as np

try:
    import psutil
except ImportError:
    psutil = None

import tvm
from tvm import rpc
from tvm.tir import expr
from tvm.tir.transform import Simplify
from tvm.ir.transform import Sequential
from ..te import Tensor, placeholder


def decode_workload_key(workload_key):
    """Decode the workload key from a string to the name and arguments. The wokrload key
    is expected to be a list of "[func_name/hash, args ...]" in a JSON string. If not,
    then simply return the workload key as the name without arguments.

    Parameters
    ----------
    workload_key: str
        The workload key in string. Format: "[func_name/hash, args ...]".

    Returns
    -------
    name: str
        The workload function name or the DAG hash.
    args: Optional[Tuple[Any, ...]]
        The flatten arguments in a tuple, or None if the workload key format is not decodeable.
    """

    def flatten_list(inp):
        ret = []
        for elt in inp:
            if isinstance(elt, list):
                ret += flatten_list(elt)
            else:
                ret.append(elt)
        return ret

    try:
        key_list = json.loads(workload_key)
        if isinstance(key_list, list) and len(key_list) >= 1:
            return key_list[0], tuple(flatten_list(key_list[1:]))
    except json.decoder.JSONDecodeError:
        pass
    return workload_key, None


def calc_workload_dis_factor(target_workload_pair, workload_pair):
    """Calculate the distance factor of the workload to the target workload.
    If two workloads are not compatible at all (i.e., different compute DAG or function),
    then the distance factor is "inf". Otherwise, we calculate the factor by traversing
    the workload arguments, which are the arguments of the compute function,
    or the output shapes for the ComputeDAG. The factor is calculated by the following rules:

    1. For non-zero integer values: `product(target_arg / candidate_arg)`.
    2. For non-integer or zero values: "inf" if not equal else 1.

    As a result, factor=1 is the optimal when two workloads are identical.

    Parameters
    ----------
    target_workload_pair: Tuple[str, Optional[Tuple[Any, ...]]]
        The target workload pair: (hash, argument tuple).

    workload_pair: Tuple[str, Optional[Tuple[Any, ...]]]
        The candidate workload pair: (hash, argument tuple).

    Returns
    -------
    dis_f: float
        The distance factor.
    """
    target_key, target_args = target_workload_pair
    target_args = target_args if target_args is not None else []
    key, args = workload_pair
    args = args if args is not None else []

    # Not even the same func/DAG.
    if key != target_key or len(target_args) != len(args):
        return float("inf")

    dis_f = 1
    for target_arg, arg in zip(target_args, args):
        if isinstance(target_arg, int):
            if target_arg == 0 or arg == 0:
                if target_arg != arg:
                    return float("inf")
            elif target_arg % arg != 0:
                return float("inf")
            else:
                dis_f *= target_arg / arg
        elif target_arg != arg:
            return float("inf")
    return dis_f


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
    out_tuple : Tuple[Union[int,tvm.tir.Var,tvm.tir.Any]]
        The output tuple of int. The dynamic shape variables (Var or Any) will be preserved.
    """
    ret = []
    for elem in in_tuple:
        if isinstance(elem, (tvm.tir.Var, tvm.tir.expr.Any)):
            ret.append(elem)
        else:
            ret.append(get_const_int(elem))
    return tuple(ret)


def list_to_tuple(x):
    """Convert a list to a tuple recursively."""
    assert isinstance(x, list)
    return tuple(list_to_tuple(y) if isinstance(y, list) else y for y in x)


def serialize_args(args):
    """
    Serialize arguments of a function to a hashable and jsonable tuple.
    Currently this is mainly used for tvm.tensor.Tensor
    """
    ret = []
    if args is None:
        return tuple(ret)

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


def kill_child_processes(parent_pid, sig=signal.SIGTERM):
    """kill all child processes recursively"""
    if not psutil:
        raise ImportError("psutil not found, try `pip install psutil` to fix this")

    try:
        parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return

    try:
        children = parent.children(recursive=True)
        for process in children:
            process.send_signal(sig)
    except psutil.NoSuchProcess:
        return


# The maximum length of traceback information
MAX_TRACEBACK_INFO_LEN = 512


def make_traceback_info():
    """Get the error message from traceback."""
    info = str(traceback.format_exc())
    if len(info) > MAX_TRACEBACK_INFO_LEN:
        info = (
            info[: MAX_TRACEBACK_INFO_LEN // 2] + "\n...\n" + info[-MAX_TRACEBACK_INFO_LEN // 2 :]
        )
    return info


class PropagatingThread(threading.Thread):
    """A thread that propagates the exception to the main thread"""

    def run(self):
        self.exc = None
        try:
            self.ret = self._target(*self._args, **self._kwargs)
        except Exception as e:  # pylint: disable=broad-except
            self.exc = e

    def join(self, timeout=None):
        super(PropagatingThread, self).join(timeout)
        if self.exc:
            raise self.exc
        return self.ret


def call_func_with_thread(func, args, kwargs):
    """Call a function within a new thread"""
    res = []

    def wrapper():
        res.append(func(*args, **kwargs))

    t = PropagatingThread(target=wrapper)
    t.start()
    t.join()
    return res[0]


def call_func_with_timeout(
    worker, timeout, func, args=(), kwargs=None
):  # pylint: disable=unused-argument
    """Call a function with timeout"""
    worker.send(func, args, kwargs, timeout)
    try:
        res = worker.recv()
    except Exception:  # pylint: disable=broad-except
        res = Exception(make_traceback_info())

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


def array_mean(arr):
    """Compute mean of the elments in a TVM Array<PrimExpr>

    Parameters
    ----------
    arr: Array
        A TVM Array<PrimExpr>

    Returns
    -------
    mean: float
        The mean of the elements in the array
    """
    return sum(x.value for x in arr) / len(arr)


def to_str_round(x, decimal=6):
    """Convert an object to str and round float numbers

    Parameters
    ----------
    x: Union[str, list, int, float, np.ndarray]
        The input object
    decimal: int
        The precision of decimal fraction

    Returns
    -------
    ret: str
        The string format of these objects
    """
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple, np.ndarray)):
        return "[" + ", ".join([to_str_round(y, decimal=decimal) for y in x]) + "]"
    if isinstance(x, dict):
        return str({k: to_str_round(v) for k, v in x.items()})
    if isinstance(x, int):
        return str(x)
    if isinstance(x, (np.float32, np.float64, float)):
        format_str = "%%.%df" % decimal
        return format_str % x
    raise ValueError("Invalid value: " + str(x) + "\ttype: " + str(type(x)))
