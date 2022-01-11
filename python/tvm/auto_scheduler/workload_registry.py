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

import logging
import pickle
import json

import tvm._ffi
from tvm.runtime._ffi_node_api import LoadJSON, SaveJSON
from .utils import serialize_args, deserialize_args, get_func_name

logger = logging.getLogger("auto_scheduler")

# Global workload function and hash key registry
# It stores two types of workload:
# 1. User registered tasks. This type of workload is registered
#    by the decorator "register_workload"
# 2. Extracted tasks from a relay program. This type of workload is
#    registered by function "register_workload_tensors".
#
# For 1, the dictionary maps a function name to its function pointer
# For 2, the dictionary maps a hash key to a list of input/output tensors
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
        Whether to override existing entry.

    Examples
    --------
    .. code-block:: python

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


def register_workload_tensors(workload_key, tensors, override=True):
    """Register a workload by provding input/output tensors. Since this function is used
    when extracting/deserializing tasks, it expects duplicated registrations by default.

    Parameters
    ----------
    workload_key: str
        The wokrload key of the compute DAG in JSON string.
    tensors: List[Tensor]
        The input/output tensors of a compute DAG
    override : boolean = True
        Whether to override existing entry.

    Returns
    -------
    workload_key: str
        The wokrload key of the compute DAG in JSON string.
    """
    register_workload(workload_key, override=override)(tensors)
    return workload_key


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


@tvm._ffi.register_func("auto_scheduler.workload_key_to_tensors")
def workload_key_to_tensors(workload_key):
    """Get the input/output tensors from the workload key.

    This method is usually used to create a ComputeDAG by workload key.

    Parameters
    ----------
    workload_key : str
        The input workload key in JSON string. The format is either (func_name, arguments...)
        for compute functions, or (hash, shapes...) for ComputeDAG.

    Returns
    -------
    tensors : List[Tensor]
        The registered compute declaration Tensors.
    """
    global WORKLOAD_FUNC_REGISTRY

    # We register ComputeDAG with both hash and argumetns, which are fixed in ComputeDAG,
    # so we use an entire workload key to query the ComputeDAG.
    if workload_key in WORKLOAD_FUNC_REGISTRY:
        return WORKLOAD_FUNC_REGISTRY[workload_key]

    # We register compute function with only the function name since
    # it does not bind to specific arguments, so we use the function name to query
    # the function and call the function with arguments to get the tensors.
    workload = json.loads(workload_key)
    name = workload[0]
    value = WORKLOAD_FUNC_REGISTRY[name]
    assert callable(value)

    args = deserialize_args(workload[1:])
    return value(*args)


def serialize_workload_registry_entry(workload_key):
    """
    Serialize a workload registry entry.

    This is used when the start method of multiprocessing is spawn.
    We need to serialize the entry and register it in the new processes.

    Parameters
    ----------
    workload_key : str
        The workload key

    Returns
    -------
    data: Tuple
        The serialized pickable data
    """
    global WORKLOAD_FUNC_REGISTRY

    if workload_key in WORKLOAD_FUNC_REGISTRY:
        sname = workload_key
    else:
        workload = json.loads(workload_key)
        sname = workload[0]

    svalue = WORKLOAD_FUNC_REGISTRY[sname]
    if not callable(svalue):
        # pylint: disable=assignment-from-no-return
        svalue = SaveJSON(svalue)

    return sname, svalue


def deserialize_workload_registry_entry(data):
    """
    Deserialize a workload registry entry.
    This should be used along with :code:`serialize_workload_registry_entry`

    Parameters
    ----------
    data: Tuple
        The return value of :code:`serialize_workload_registry_entry`
    """
    global WORKLOAD_FUNC_REGISTRY

    name, value = data
    if name not in WORKLOAD_FUNC_REGISTRY:
        # pylint: disable=assignment-from-no-return
        if not callable(value):
            value = LoadJSON(value)
        WORKLOAD_FUNC_REGISTRY[name] = value


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
