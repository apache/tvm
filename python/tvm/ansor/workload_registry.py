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

"""
Workload registration and serialization.

We use a json string to represent a workload (a compute dag).
The format of the string is `[func_name, [args...]]`.
The dag should be the return value of this `func_name(*args)`.

Rationale: The workload is actually a compute dag defined by tvm dsl. But serializing compute dags
and matching them efficiently is not easy. Therefore, we use the above string to encode a compute
dag.
These strings are efficient for serialization/matching and wont' be too long.
When we need the dag, we decode the string and call the function, which will return the dag.
"""

from typing import Hashable
import pickle
import json
import hashlib

import tvm._ffi
from ..te import Tensor, PlaceholderOp, ComputeOp, placeholder
from .utils import get_const_tuple
from .compute_dag import ComputeDAG

WORKLOAD_FUNC_REGISTRY = {}


def register_workload_func(func):
    """Register a workload generation function
    The input function should take hashable and jsonable arguments
    (int, float, tuple of int, tvm.tensor.Tensor, ...) and return a list of tvm.tensor.Tensor.

    Examples
    --------
    @register_workload_func
    def matmul(N, M, K):
        A = te.placeholder((N, K), name='A')
        B = te.placeholder((K, M), name='B')
        k = te.reduce_axis((0, K), name='k')
        C = te.compute((N, M), lambda i, j: tvm.sum(A[i][k] * B[k][j], axis=[k]), name='C')
        return [A, B, C]
    """
    func_name = func.__name__
    if func_name in WORKLOAD_FUNC_REGISTRY:
        raise RuntimeError('%s has been registered already' % func_name)
    WORKLOAD_FUNC_REGISTRY[func_name] = func
    return func


def compute_dag_hash(dag):
    """ Get hash value for a ComputeDAG.

    Parameters
    ----------
    dag : ComputeDAG
        The target ComputeDAG.

    Returns
    -------
    hash_value : Str
        The hash value of this ComputeDAG in hex digest.
    """
    # todo: implement this more carefully and move this to c++ as a member function of ComputeDAG
    str_key = ''
    for op in dag.ops:
        t = op.output(0)
        if isinstance(op, PlaceholderOp):
            str_key += 'placeholder,'
            str_key += str(get_const_tuple(t.shape)) + ','
            str_key += t.dtype + ';'
        elif isinstance(op, ComputeOp):
            str_key += str(t.op.body) + ','
            str_key += str(get_const_tuple(t.shape)) + ','
            str_key += t.dtype + ';'
        else:
            raise ValueError("Invalid op: " + op)

    str_key = str_key.encode(encoding='utf-8')
    return hashlib.md5(str_key).hexdigest()


def register_workload_bufs(bufs):
    """ Directly register buffers of a workload and return the workload_key.

    The buffers can be looked up with workload_key_to_tensors by the workload_key.

    Parameters
    ----------
    bufs : List[Tensor]
        A list of Tensors for the target compute declaration.

    Returns
    -------
    workload_key : Str
        A workload key mapping to the registered compute declaration.
    """
    dag = ComputeDAG(bufs)
    key = compute_dag_hash(dag)
    WORKLOAD_FUNC_REGISTRY[key] = bufs
    return json.dumps((key,))


def list_to_tuple(x):
    """Convert a list to a tuple recursively"""
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
            t = ('TENSOR', get_const_tuple(t.shape), t.dtype)
        elif isinstance(t, list):
            t = list_to_tuple(t)

        assert isinstance(t, Hashable), str(t) + " is not hashable"
        ret.append(t)

    return tuple(ret)


def deserialize_args(args):
    """The inverse function of :code:`serialize_args`"""
    ret = []
    for t in args:
        if isinstance(t, (tuple, list)) and t[0] == 'TENSOR':
            ret.append(placeholder(shape=t[1], dtype=t[2]))
        else:
            ret.append(t)
    return ret


@tvm._ffi.register_func("ansor.workload_key_to_tensors")
def workload_key_to_tensors(workload_key):
    """ Decode a workload key to the input/output tensors.

    Parameters
    ----------
    workload_key : Str
        The target workload key.

    Returns
    -------
    tensors : List[Tensor]
        The registered compute declaration Tensors.
    """
    workload = json.loads(workload_key)
    name = workload[0]
    lookup = WORKLOAD_FUNC_REGISTRY[name]

    if callable(lookup):
        args = deserialize_args(workload[1:])
        return lookup(*args)
    return lookup


@ tvm._ffi.register_func("ansor.workload_key_to_dag")
def workload_key_to_dag(workload_key):
    """ Decode a workload key to a compute dag.

    Parameters
    ----------
    workload_key : Str
        The target workload key.

    Returns
    -------
    dag : ComputeDAG
        ComputeDAG to the registered compute declaration.
    """
    tensors = workload_key_to_tensors(workload_key)
    return ComputeDAG(tensors)


def make_workload_key_func(func, args):
    """ make a workload key from function and arguments.

    Parameters
    ----------
    func : Function
        The target function that returns the compute declaration Tensors.
    args : Args
        The args of the target function.

    Returns
    -------
    workload_key : Str
        The workload key of the target function.
    """
    args = serialize_args(args)

    if callable(func):
        func_name = func.__name__
    elif isinstance(func, str):
        func_name = func
    else:
        raise ValueError("Invalid function: " + str(func))

    assert func_name in WORKLOAD_FUNC_REGISTRY, \
        "%s is not registered. Please register it with register_auto_scheduler_workload_func" % func

    return json.dumps((func_name,) + args)


def make_workload_key_bufs(bufs):
    """ make a workload key from bufs.

    Parameters
    ----------
    bufs : List[Tensor]
        A list of Tensors for the target compute declaration.

    Returns
    -------
    workload_key : Str
        A workload key mapping to the registered compute declaration.
    """
    dag = ComputeDAG(bufs)
    key = compute_dag_hash(dag)
    return json.dumps((key,))


def dump_workload_func_registry(filename):
    """ Dump workload function registry to a pickle binary file.

    Parameters
    ----------
    filename : Str
        The filename to dump workload function registry to.
    """
    global WORKLOAD_FUNC_REGISTRY

    pickle.dump(WORKLOAD_FUNC_REGISTRY, open(filename, 'wb'))


def load_workload_func_registry(filename):
    """ Load workload function registry from a pickle binary file.

    Parameters
    ----------
    filename : Str
        The filename to load workload function registry from.
    """
    global WORKLOAD_FUNC_REGISTRY

    WORKLOAD_FUNC_REGISTRY = pickle.load(open(filename, 'rb'))
