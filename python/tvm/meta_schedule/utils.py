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
import ctypes
import os
import shutil
from typing import Any, Callable, List, Optional, Union

import numpy as np  # type: ignore
import psutil  # type: ignore
from tvm._ffi import get_global_func, register_func
from tvm.error import TVMError
from tvm.ir import Array, IRModule, Map
from tvm.rpc import RPCSession
from tvm.runtime import PackedFunc, String
from tvm.tir import FloatImm, IntImm


def derived_object(cls: type) -> type:
    """A decorator to register derived subclasses for TVM objects.

    Parameters
    ----------
    cls : type
        The derived class to be registered.

    Returns
    -------
    cls : type
        The decorated TVM object.

    Example
    -------
    .. code-block:: python

        @register_object("meta_schedule.PyRunner")
        class _PyRunner(meta_schedule.Runner):
            def __init__(self, f_run: Callable = None):
                self.__init_handle_by_constructor__(_ffi_api.RunnerPyRunner, f_run)

        class PyRunner:
            _tvm_metadata = {
                "cls": _PyRunner,
                "methods": ["run"]
            }
            def run(self, runner_inputs):
                raise NotImplementedError

        @derived_object
        class LocalRunner(PyRunner):
            def run(self, runner_inputs):
                ...
    """

    import functools  # pylint: disable=import-outside-toplevel
    import weakref  # pylint: disable=import-outside-toplevel

    def _extract(inst: type, name: str):
        """Extract function from intrinsic class."""

        def method(*args, **kwargs):
            return getattr(inst, name)(*args, **kwargs)

        if getattr(base, name) is getattr(cls, name) and name != "__str__":
            # for task scheduler return None means calling default function
            # otherwise it will trigger a TVMError of method not implemented
            # on the c++ side when you call the method, __str__ not required
            return None
        return method

    assert isinstance(cls.__base__, type)
    assert hasattr(
        cls, "_tvm_metadata"
    ), "Please use the user-facing method overriding class, i.e., PyRunner."

    base = cls.__base__
    metadata = getattr(base, "_tvm_metadata")
    fields = metadata.get("fields", [])
    methods = metadata.get("methods", [])

    class TVMDerivedObject(metadata["cls"]):  # type: ignore
        """The derived object to avoid cyclic dependency."""

        def __init__(self, *args, **kwargs):
            """Constructor."""
            self.handle = None
            self._inst = cls(*args, **kwargs)

            super().__init__(
                # the constructor's parameters, builder, runner, etc.
                *[getattr(self._inst, name) for name in fields],
                # the function methods, init_with_tune_context, build, run, etc.
                *[_extract(self._inst, name) for name in methods],
            )

            # for task scheduler hybrid funcs in c++ & python side
            # using weakref to avoid cyclic dependency
            self._inst._outer = weakref.ref(self)

        def __getattr__(self, name: str):
            """Bridge the attribute function."""
            try:
                return self._inst.__getattribute__(name)
            except AttributeError:
                return super(TVMDerivedObject, self).__getattr__(name)

        def __setattr__(self, name, value):
            if name not in ["_inst", "key", "handle"]:
                self._inst.__setattr__(name, value)
            else:
                super(TVMDerivedObject, self).__setattr__(name, value)

    functools.update_wrapper(TVMDerivedObject.__init__, cls.__init__)  # type: ignore
    TVMDerivedObject.__name__ = cls.__name__
    TVMDerivedObject.__doc__ = cls.__doc__
    TVMDerivedObject.__module__ = cls.__module__
    return TVMDerivedObject


@register_func("meta_schedule.cpu_count")
def _cpu_count_impl(logical: bool = True) -> int:
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
    return _cpu_count_impl(logical)


@register_func("meta_schedule.using_ipython")
def _using_ipython() -> bool:
    """Return whether the current process is running in an IPython shell.

    Returns
    -------
    result : bool
        Whether the current process is running in an IPython shell.
    """
    try:
        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"  # type: ignore
    except NameError:
        return False


@register_func("meta_schedule.print_interactive_table")
def print_interactive_table(data: str) -> None:
    """Print the dataframe interactive table in notebook.

    Parameters
    ----------
    data : str
        The serialized performance table from MetaSchedule table printer.
    """
    import pandas as pd  # type: ignore # pylint: disable=import-outside-toplevel
    from IPython.display import display  # type: ignore # pylint: disable=import-outside-toplevel

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_colwidth", None)
    parsed = [
        x.split("|")[1:] for x in list(filter(lambda x: set(x) != {"-"}, data.strip().split("\n")))
    ]
    display(
        pd.DataFrame(
            parsed[1:],
            columns=parsed[0],
        )
    )


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


def shash2hex(mod: IRModule) -> str:
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
    func = get_global_func("meta_schedule._SHash2Hex")
    return str(func(mod))


def _get_default_str(obj: Any) -> str:
    return (
        # pylint: disable=protected-access
        f"meta_schedule.{obj.__class__.__name__}"
        + f"({_to_hex_address(obj._outer().handle)})"  # type: ignore
        # pylint: enable=protected-access
    )


def _to_hex_address(handle: ctypes.c_void_p) -> str:
    """Get the hexadecimal address of a handle.
    Parameters
    ----------
    handle : ctypes.c_void_p
        The handle to be converted.
    Returns
    -------
    result : str
        The hexadecimal address of the handle.
    """
    return hex(ctypes.cast(handle, ctypes.c_void_p).value)


def fork_seed(seed: Optional[int], n: int) -> List[int]:
    # fmt: off
    return np.random.RandomState(seed=seed).randint(1, 2 ** 30, size=n).tolist()
    # fmt: on
