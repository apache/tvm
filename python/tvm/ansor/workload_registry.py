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

import pickle
import json

import tvm._ffi
from .utils import serialize_args, deserialize_args

WORKLOAD_FUNC_REGISTRY = {}


def register_workload_by_func(func):
    """ Register a workload by generation function.

    The input function should take hashable and jsonable arguments
    (int, float, tuple of int, tvm.tensor.Tensor, ...) and return a list of tvm.tensor.Tensor.

    Examples
    --------
    @ansor.register_workload_by_func
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


def make_workload_key_by_func(func, args):
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

    if not func_name in WORKLOAD_FUNC_REGISTRY:
        raise ValueError("%s is not registered. "  % func,
                         "Please register it with @ansor.register_workload_by_func")

    return json.dumps((func_name,) + args)


def decode_workload_key_to_func_args(workload_key):
    """ Decode a workload key to the registerd function name and its corresponding args.

    Parameters
    ----------
    workload_key : str
        The target workload key.

    Returns
    -------
    name : str
        The function name of this workload key.
    args : List[Tensor]
        The args of the generation function.
    """
    workload = json.loads(workload_key)
    if not workload[0] in WORKLOAD_FUNC_REGISTRY:
        raise ValueError("%s is not registered. " % workload[0] +
                         "Please register it with @ansor.register_workload_by_func")
    return workload[0], deserialize_args(workload[1:])


@tvm._ffi.register_func("ansor.workload_key_to_tensors")
def workload_key_to_tensors(workload_key):
    """ Get the input/output tensors from the workload key.

    This method is usually used to create a ComputeDAG by workload key.

    Parameters
    ----------
    workload_key : str
        The target workload key.

    Returns
    -------
    tensors : List[Tensor]
        The registered compute declaration Tensors.
    """
    name, args = decode_workload_key_to_func_args(workload_key)
    lookup = WORKLOAD_FUNC_REGISTRY[name]
    assert callable(lookup)
    return lookup(*args)


def dump_workload_func_registry(filename):
    """ Dump workload function registry to a pickle binary file.

    Parameters
    ----------
    filename : str
        The filename to dump workload function registry to.
    """
    global WORKLOAD_FUNC_REGISTRY

    pickle.dump(WORKLOAD_FUNC_REGISTRY, open(filename, 'wb'))


def load_workload_func_registry(filename):
    """ Load workload function registry from a pickle binary file.

    Parameters
    ----------
    filename : str
        The filename to load workload function registry from.
    """
    global WORKLOAD_FUNC_REGISTRY

    WORKLOAD_FUNC_REGISTRY = pickle.load(open(filename, 'rb'))
