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

We use a json string to represent a workload (a computation graph).
The format of the string is `[func_name, [args...]]`.
The dag should be the return value of this `func_name(*args)`.

Rationale: The workload is actually a compute dag defined by tvm dsl. But serializing compute dags
and matching them efficiently is not easy. Therefore, we use the above string to encode a compute
dag.
These strings are efficient for serialization/matching and won't be too long.
When we need the dag, we decode the string and call the function, which will return the dag.
"""

import pickle
import json

import tvm._ffi
from .utils import serialize_args, deserialize_args, get_func_name

WORKLOAD_FUNC_REGISTRY = {}


def register_workload(func_name, f=None, override=False):
    """Register a function that generates a certain workload.

    The input function should take hashable and jsonable arguments
    (int, float, tuple of int, tvm.tensor.Tensor, ...) and return a list of tvm.tensor.Tensor.

    Parameters
    ----------
    func_name : Union[Function, str]
        The generation function that returns the compute declaration Tensors or its function name.
    f : Optional[Function]
        The generation function to be registered.
    override : boolean = False
        Whether override existing entry.

    Examples
    --------
    @auto_scheduler.register_workload
    def matmul(N, M, K):
        A = te.placeholder((N, K), name='A')
        B = te.placeholder((K, M), name='B')
        k = te.reduce_axis((0, K), name='k')
        C = te.compute((N, M), lambda i, j: tvm.sum(A[i][k] * B[k][j], axis=[k]), name='C')
        return [A, B, C]
    """
    global WORKLOAD_FUNC_REGISTRY

    if callable(func_name):
        f = func_name
        func_name = get_func_name(f)
    if not isinstance(func_name, str):
        raise ValueError("expect string function name")

    def register(myf):
        """internal register function"""
        if func_name in WORKLOAD_FUNC_REGISTRY and not override:
            raise RuntimeError("%s has been registered already" % func_name)
        WORKLOAD_FUNC_REGISTRY[func_name] = myf
        return myf

    if f:
        return register(f)
    return register


def make_workload_key(func, args):
    """Make a workload key by function and arguments.

    Parameters
    ----------
    func : Union[Function, str]
        The function that returns the compute declaration Tensors.
        Can be the a function or the function name.
    args : Args
        The args of the function.

    Returns
    -------
    workload_key : str
        The workload key of the function.
    """
    global WORKLOAD_FUNC_REGISTRY

    if callable(func):
        func_name = get_func_name(func)
    elif isinstance(func, str):
        func_name = func
    else:
        raise ValueError(
            "Invalid function: "
            + str(func)
            + " . `make_workload_key` expects a callable function or its function name"
        )

    if not func_name in WORKLOAD_FUNC_REGISTRY:
        raise ValueError(
            "%s is not registered. " % func,
            "Please register it with @auto_scheduler.register_workload",
        )

    args = serialize_args(args)

    return json.dumps((func_name,) + args)


def decode_workload_key_to_func_args(workload_key):
    """Decode a workload key to the registerd function name and its corresponding args.

    Parameters
    ----------
    workload_key : str
        The input workload key.

    Returns
    -------
    name : str
        The function name of this workload key.
    args : List[Tensor]
        The args of the generation function.
    """
    global WORKLOAD_FUNC_REGISTRY

    workload = json.loads(workload_key)
    if not workload[0] in WORKLOAD_FUNC_REGISTRY:
        raise ValueError(
            "%s is not registered. " % workload[0]
            + "Please register it with @auto_scheduler.register_workload"
        )
    return workload[0], deserialize_args(workload[1:])


@tvm._ffi.register_func("auto_scheduler.workload_key_to_tensors")
def workload_key_to_tensors(workload_key):
    """Get the input/output tensors from the workload key.

    This method is usually used to create a ComputeDAG by workload key.

    Parameters
    ----------
    workload_key : str
        The input workload key.

    Returns
    -------
    tensors : List[Tensor]
        The registered compute declaration Tensors.
    """
    global WORKLOAD_FUNC_REGISTRY

    name, args = decode_workload_key_to_func_args(workload_key)
    lookup = WORKLOAD_FUNC_REGISTRY[name]
    assert callable(lookup)
    return lookup(*args)


def save_workload_func_registry(filename):
    """Dump workload function registry to a pickle binary file.

    Parameters
    ----------
    filename : str
        The filename to dump workload function registry to.
    """
    global WORKLOAD_FUNC_REGISTRY

    pickle.dump(WORKLOAD_FUNC_REGISTRY, open(filename, "wb"))


def load_workload_func_registry(filename):
    """Load workload function registry from a pickle binary file.

    Parameters
    ----------
    filename : str
        The filename to load workload function registry from.
    """
    global WORKLOAD_FUNC_REGISTRY

    WORKLOAD_FUNC_REGISTRY = pickle.load(open(filename, "rb"))
