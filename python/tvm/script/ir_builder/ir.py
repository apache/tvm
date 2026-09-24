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
"""Package tvm.script.ir_builder.ir"""

import inspect
from typing import TypeVar

from tvm.ir import BaseFunc, GlobalInfo, GlobalVar
from tvm.runtime import Object as tvm_Object

from . import _ffi_api
from .base import IRBuilder
from .frame import IRModuleFrame

T = TypeVar("T")


def meta_var(value: T) -> T:
    """Return a Python metadata value without binding, naming or relocating it.

    Parameters
    ----------
    value : T
        Any host or IR object, including an unpackable sequence.

    Returns
    -------
    T
        The exact input object. No frame is required, no IR is emitted, and
        existing names and source locations are retained.

    .. code:: python

        # Source and generated Python (the value retains its identity)
        value = I.meta_var(existing_value)
        a, b = I.meta_var((left, right))
    """
    return value


def ir_module() -> IRModuleFrame:
    """Start a ir_module frame.

    Returns
    -------
    frame: IRModuleFrame
        The constructed frame.
    """
    return _ffi_api.IRModule()  # type: ignore[attr-defined] # pylint: disable=no-member


def decl_function(func_name: str, func_signature: BaseFunc) -> GlobalVar:
    """Declare a Function without given the specific function implementation.

    Parameters
    ----------
    func_name : str
        The function unique name.

    func_signature: BaseFunc
        A Function w/o body, which used to specify the function signature
        (i.e. func params and func return type/shape).

    Note
    ----
    It is usually used in cross-function call. And we can specify the function by `DefFunction`

    Returns
    -------
    gv : GlobalVar
        The corresponding GlobalVar.
    """
    if not isinstance(func_signature, BaseFunc):
        raise ValueError(
            "decl_function expects an instance of BaseFunc, "
            f"but {func_signature} is of type {type(func_signature)}"
        )
    return _ffi_api.DeclFunction(  # type: ignore[attr-defined] # pylint: disable=no-member
        func_name, func_signature
    )


def def_function(func_name: str, func: BaseFunc) -> None:
    """Define the function which is declared before.
    Parameters
    ----------
    func_name : str
        The function unique name.
    func: BaseFunc
        The given function implementation
    """
    return _ffi_api.DefFunction(func_name, func)  # type: ignore[attr-defined] # pylint: disable=no-member


def module_attrs(attrs: dict[str, tvm_Object], allow_overwrite=False) -> None:
    """Specify the attrs of the ir_module frame.
    Parameters
    ----------
    attrs: Dict[str, Object]
        The module attrs.
    allow_overwrite: bool
        Whether allow overwrite the existing attrs.
    """
    return _ffi_api.ModuleAttrs(attrs, allow_overwrite)  # type: ignore[attr-defined] # pylint: disable=no-member


def module_get_attr(attr_key: str) -> tvm_Object | None:
    """Get the specified attr of the ir_module frame.
    Parameters
    ----------
    attr_key: str
        The key of the attr to be retrieved.
    Returns
    -------
    attr: Optional[Object]
        The specified module attr or None if not found.
    """
    return _ffi_api.ModuleGetAttr(attr_key)  # type: ignore[attr-defined] # pylint: disable=no-member


def module_set_attr(
    attr_key: str, attr_value: tvm_Object | None, allow_overwrite: bool = False
) -> None:
    """Set the specified attr of the ir_module frame.
    Parameters
    ----------
    attr_key: str
        The key of the attr to be set.
    attr_value: Optional[Object]
        The value of the attr to be set.
    allow_overwrite: bool
        Whether allow overwrite the existing attr.
    """
    return _ffi_api.ModuleSetAttr(attr_key, attr_value, allow_overwrite)  # type: ignore[attr-defined] # pylint: disable=no-member


def module_global_infos(global_infos: dict[str, list[GlobalInfo]]) -> None:
    """Specify the global infos of the ir_module frame.

    Parameters
    ----------
    global_infos: Dict[str, List[GlobalInfo]]
        The module global infos.
    """
    if IRBuilder.is_in_scope():
        return _ffi_api.ModuleGlobalInfos(global_infos)
    # Keep native argument validation even before Python has applied the module decorator.
    _ffi_api.ModuleGlobalInfos(global_infos)
    frame = inspect.currentframe().f_back
    try:
        if _is_class_frame(frame):
            previous = frame.f_locals.get("__tvm_script_global_infos__")
            if previous:
                raise ValueError(f"Duplicate module global_infos, previous one is:\n{previous}")
            frame.f_locals["__tvm_script_global_infos__"] = {
                name: tuple(values) for name, values in global_infos.items()
            }
    finally:
        del frame


def _is_class_frame(frame):
    return (
        not frame.f_code.co_flags & inspect.CO_NEWLOCALS
        and frame.f_locals is not frame.f_globals
        and "__module__" in frame.f_locals
        and frame.f_locals.get("__qualname__", "").split(".")[-1] == frame.f_code.co_name
    )


def _class_global_infos():
    # Only a still-executing class owns eager signature context. Never retain its frame,
    # and do not fall through an inner class to an unrelated outer module declaration.
    frame = inspect.currentframe().f_back
    try:
        while frame is not None:
            if _is_class_frame(frame):
                return frame.f_locals.get("__tvm_script_global_infos__", {})
            frame = frame.f_back
    finally:
        del frame
    return {}


def lookup_global_info(name: str, index: int) -> GlobalInfo:
    """Resolve a concrete global info in a builder or an executing module class."""
    if IRBuilder.is_in_scope():
        for frame in reversed(IRBuilder.current().frames):
            if isinstance(frame, IRModuleFrame):
                return frame.global_infos[name][index]
        raise ValueError("The GlobalInfos in the IRModule is not defined.")
    infos = _class_global_infos()
    if not infos:
        raise ValueError("The GlobalInfos in the IRModule is not defined.")
    return infos[name][index]


def lookup_name(name: str) -> bool:
    """Check if a global variable with the given name exists.
    Parameters
    ----------
    name: str
        The name of the global variable.

    Returns
    -------
    res : bool
        True if the global variable exists, False otherwise.
    """
    return _ffi_api.LookupName(name)  # type: ignore[attr-defined] # pylint: disable=no-member
