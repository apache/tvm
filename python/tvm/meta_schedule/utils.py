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
"""Utilities for meta schedule"""
import json
import os
import shutil
from typing import Any, Callable, List, Optional, Union

import psutil  # type: ignore
import tvm
from tvm._ffi import get_global_func, register_func
from tvm.error import TVMError
from tvm.ir import Array, Map, IRModule
from tvm.rpc import RPCSession
from tvm.runtime import PackedFunc, String
from tvm.tir import FloatImm, IntImm


@register_func("meta_schedule.cpu_count")
def cpu_count(logical: bool = True) -> int:
    """Return the number of logical or physical CPUs in the system

    Parameters
    ----------
    logical : bool = True
        If True, return the number of logical CPUs, otherwise return the number of physical CPUs

    Returns
    -------
    cpu_count : int
        The number of logical or physical CPUs in the system

    Note
    ----
    The meta schedule search infra intentionally does not adopt the following convention in TVM:
    - C++ API `tvm::runtime::threading::MaxConcurrency()`
    - Environment variable `TVM_NUM_THREADS` or
    - Environment variable `OMP_NUM_THREADS`

    This is because these variables are dedicated to controlling
    the runtime behavior of generated kernels, instead of the host-side search.
    Setting these variables may interfere the host-side search with profiling of generated kernels
    when measuring locally.
    """
    return psutil.cpu_count(logical=logical) or 1


def get_global_func_with_default_on_worker(
    name: Union[None, str, Callable],
    default: Callable,
) -> Callable:
    """Get the registered global function on the worker process.

    Parameters
    ----------
    name : Union[None, str, Callable]
        If given a string, retrieve the function in TVM's global registry;
        If given a python function, return it as it is;
        Otherwise, return `default`.

    default : Callable
        The function to be returned if `name` is None.

    Returns
    -------
    result : Callable
        The retrieved global function or `default` if `name` is None
    """
    if name is None:
        return default
    if callable(name):
        return name
    try:
        return get_global_func(name)
    except TVMError as error:
        raise ValueError(
            "Function '{name}' is not registered on the worker process. "
            "The build function and export function should be registered in the worker process. "
            "Note that the worker process is only aware of functions registered in TVM package, "
            "if there are extra functions to be registered, "
            "please send the registration logic via initializer."
        ) from error


def get_global_func_on_rpc_session(
    session: RPCSession,
    name: str,
    extra_error_msg: Optional[str] = None,
) -> PackedFunc:
    """Get a PackedFunc from the global registry from an RPCSession.

    Parameters
    ----------
    session : RPCSession
        The RPCSession to be retrieved from
    name : str
        The name of the PackedFunc
    extra_error_msg : Optional[str]
        Extra information to provide in the error message

    Returns
    -------
    result : PackedFunc
        The result
    """
    try:
        result = session.get_function(name)
    except AttributeError as error:
        error_msg = f'Unable to find function "{name}" on the remote RPC server.'
        if extra_error_msg:
            error_msg = f"{error_msg} {extra_error_msg}"
        raise AttributeError(error_msg) from error
    return result


@register_func("meta_schedule.remove_build_dir")
def remove_build_dir(artifact_path: str) -> None:
    """Clean up the build directory"""
    shutil.rmtree(os.path.dirname(artifact_path))


def _json_de_tvm(obj: Any) -> Any:
    """Unpack a TVM nested container to a JSON object in python.

    Parameters
    ----------
    obj : Any
        The TVM nested container to be unpacked.

    Returns
    -------
    result : Any
        The unpacked json object.
    """
    if obj is None:
        return None
    if isinstance(obj, (int, float)):
        return obj
    if isinstance(obj, (IntImm, FloatImm)):
        return obj.value
    if isinstance(obj, (str, String)):
        return str(obj)
    if isinstance(obj, Array):
        return [_json_de_tvm(i) for i in obj]
    if isinstance(obj, Map):
        return {_json_de_tvm(k): _json_de_tvm(v) for k, v in obj.items()}
    raise TypeError("Not supported type: " + str(type(obj)))


@register_func("meta_schedule.json_obj2str")
def json_obj2str(json_obj: Any) -> str:
    json_obj = _json_de_tvm(json_obj)
    return json.dumps(json_obj)


@register_func("meta_schedule.batch_json_str2obj")
def batch_json_str2obj(json_strs: List[str]) -> List[Any]:
    """Covert a list of JSON strings to a list of json objects.
    Parameters
    ----------
    json_strs : List[str]
        The list of JSON strings
    Returns
    -------
    result : List[Any]
        The list of json objects
    """
    return [
        json.loads(json_str)
        for json_str in map(str.strip, json_strs)
        if json_str and (not json_str.startswith("#")) and (not json_str.startswith("//"))
    ]


def structural_hash(mod: IRModule) -> str:
    """Get the structural hash of a module.

    Parameters
    ----------
    mod : IRModule
        The module to be hashed.

    Returns
    -------
    result : str
        The structural hash of the module.
    """
    shash = tvm.ir.structural_hash(mod)
    if shash < 0:
        # Workaround because `structural_hash` returns a size_t, i.e., unsigned integer
        # but ffi can't handle unsigned integers properly so it's parsed into a negative number
        shash += 1 << 64
    return str(shash)


def check_override(
    derived_class: Any, base_class: Any, required: bool = True, func_name: str = None
) -> Callable:
    """Check if the derived class has overridden the base class's method.

    Parameters
    ----------
    derived_class : Any
        The derived class.
    base_class : Any
        The base class of derived class.
    required : bool
        If the method override is required.
    func_name : str
        Name of the method. Default value None, which would be set to substring of the given
        function, e.g. `f_generate`->`generate`.

    Returns
    -------
    func : Callable
        Raise NotImplementedError if the function is required and not overridden. If the
        function is not overridden return None, other return the overridden function.
    """

    def inner(func: Callable):

        if func_name is None:
            method = func.__name__[2:]
        else:
            method = func_name

        if getattr(derived_class, method) is getattr(base_class, method):
            if required:
                raise NotImplementedError(f"{derived_class}'s {method} method is not implemented!")
            return None
        return func

    return inner
