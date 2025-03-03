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
"""Package tvm.script.ir_builder.ir.ir"""

from typing import Dict, List, Optional

from tvm.ir import BaseFunc, GlobalVar, GlobalInfo, VDevice, DummyGlobalInfo
from tvm.runtime import Object as tvm_Object


from . import _ffi_api
from .frame import IRModuleFrame


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


def module_attrs(attrs: Dict[str, tvm_Object], allow_overwrite=False) -> None:
    """Specify the attrs of the ir_module frame.
    Parameters
    ----------
    attrs: Dict[str, Object]
        The module attrs.
    allow_overwrite: bool
        Whether allow overwrite the existing attrs.
    """
    return _ffi_api.ModuleAttrs(attrs, allow_overwrite)  # type: ignore[attr-defined] # pylint: disable=no-member


def current_ir_module() -> IRModuleFrame:
    """Get the current ir_module frame.
    Returns
    -------
    frame: IRModuleFrame
        The current frame.
    """
    return _ffi_api.CurrentIRModule()  # type: ignore[attr-defined] # pylint: disable=no-member


def module_get_attrs() -> Dict[str, tvm_Object]:
    """Get the attrs of the ir_module frame.
    Returns
    -------
    attrs: Dict[str, Object]
        The module attrs.
    """
    return _ffi_api.ModuleGetAttrs()  # type: ignore[attr-defined] # pylint: disable=no-member


def module_get_attr(attr_key: str) -> Optional[tvm_Object]:
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
    attr_key: str, attr_value: Optional[tvm_Object], allow_overwrite: bool = False
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


def module_global_infos(global_infos: Dict[str, List[GlobalInfo]]) -> None:
    """Specify the global infos of the ir_module frame.
    Parameters
    ----------
    global_infos: Dict[str, List[GlobalInfo]]
        The module global infos.
    """
    return _ffi_api.ModuleGlobalInfos(global_infos)  # type: ignore[attr-defined] # pylint: disable=no-member


############################### GlobalInfo ###############################


def dummy_global_info() -> DummyGlobalInfo:
    """Create a dummy global info expression.
    Returns
    -------
    res : DummyGlobalInfo
        The result dummy global info.
    """
    return DummyGlobalInfo()  # type: ignore[attr-defined] # pylint: disable=no-member


def vdevice(target=None, vdevice_id: int = 0, memory_scope: str = "global") -> VDevice:
    """Create a virtual device global info.
    Parameters
    ----------
    target
        The target.
    vdevice_id: int
        The virtual device index.
    memory_scope: str
        The memory scope, default is "global"

    Returns
    -------
    res : VDevice
        The result virtual device.
    """
    return VDevice(target, vdevice_id, memory_scope)  # type: ignore[attr-defined] # pylint: disable=no-member


def lookup_vdevice(target_kind: str = None, device_index: int = -1) -> VDevice:
    """Retrieve a virtual device from the globalinfo vdevice list.
    Parameters
    ----------
    target_kind: str
        The target device kind, for example 'llvm' or 'cuda'.
    device_index: int
        The virtual device index.

    Returns
    -------
    res : VDevice
        The result virtual device.
    """
    return _ffi_api.LookupVDevice(target_kind, device_index)  # type: ignore[attr-defined] # pylint: disable=no-member
