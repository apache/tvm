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
# pylint: disable=invalid-name, no-member
"""VM build logics"""
from typing import List, Optional, Union, Dict, Any

import tvm
from tvm import relax

from tvm.contrib import utils as _utils

from tvm.ir.module import IRModule
from tvm.tir.function import PrimFunc

from . import _ffi_api


class Executable:
    """The executable object emitted by the VM compiler or the ExecBuilder."""

    def __init__(self, mod: tvm.runtime.Module):
        self.mod = mod
        self._stats = self.mod["stats"]
        self._as_text = self.mod["as_text"]
        self._as_python = self.mod["as_python"]

    def stats(self) -> str:
        """print the detailed statistics of the executable."""
        return self._stats()

    def as_text(self) -> str:
        """print the instructions as text format."""
        return self._as_text()

    def as_python(self) -> str:
        """print the instructions as python program."""
        return self._as_python()

    def jit(self, fcompile=None, addons=None, **kwargs) -> tvm.runtime.Module:
        """Just-in-time compile and link the modules.

        The Executable returned by relax.build may not be directly
        runnable as they may contain cuda source files and objects that
        are yet to be compiled and linked.
        This function helps to create a runtime.Module for these cases.

        Parameters
        ----------
        fcompile : function(target, file_list, kwargs), optional
            The compilation function to use create the final library object during

        kwargs : dict, optional
            Additional arguments passed to fcompile

        Returns
        -------
        rt_mod: tvm.runtime.Module
            A runnable runtime module that can be passed to VirtualMachine.

        Examples
        --------
        .. code:: python

            ex = relax.build(mod, target)
            # build a runnable module using nvcc to link everything
            rt_mod = ex.jit()
            vm = tvm.relax.VirtualMachine(rt_mod, tvm.cuda())
        """
        # TODO(tvm-team): Update runtime.Module interfac
        # to query these properties as bitmask.
        def _not_runnable(x):
            return x.type_key in ("c", "static_library")

        # pylint:disable = protected-access
        not_runnable_list = self.mod._collect_from_import_tree(_not_runnable)

        # everything is runnable, directly return mod.
        if len(not_runnable_list) == 0:
            return self.mod

        # found source module, or other not runnable modules
        # need to be export and load
        # TODO(tvm-team): Support runnable but not exportable module.
        # by collecting the link and allow export_library skip those modules.
        workspace_dir = _utils.tempdir()
        dso_path = workspace_dir.relpath("exported.so")
        self.mod.export_library(dso_path, fcompile=fcompile, addons=addons, **kwargs)
        return tvm.runtime.load_module(dso_path)

    def export_library(
        self,
        file_name: str,
        fcompile: Optional[Union[str, callable]] = None,
        workspace_dir: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """Export the executable to a library which can then be loaded back.

        Parameters
        ----------
        file_name : str
            The name of the shared library.

        fcompile : function(target, file_list, kwargs), optional
            The compilation function to use create the final library object during

        workspace_dir : str, optional
            The path of the directory used to create the intermediate
            artifacts when exporting the module.
            If this is not provided a temporary dir will be created.

        kwargs : dict, optional
            Additional arguments passed to fcompile

        Returns
        -------
        result of fcompile()  : unknown, optional
            If the compilation function returns an artifact it would be returned via
            export_library, if any.

        Examples
        --------
        .. code:: python

            ex = relax.build(mod, target)
            # export the library
            ex.export_library("exported.so")

            # load it back for future uses.
            rt_mod = tvm.runtime.load_module("exported.so")
            vm = tvm.relax.VirtualMachine(rt_mod, tvm.cuda())
        """
        return self.mod.export_library(
            file_name=file_name, fcompile=fcompile, workspace_dir=workspace_dir, **kwargs
        )


def _vmcodegen(
    builder: "relax.ExecBuilder",
    mod: tvm.IRModule,
    exec_mode: str = "bytecode",
) -> tvm.IRModule:
    """Running VM codegen.

    Parameters
    ----------
    builder: relax.ExecBuilder
        ExecBuilder to collect the vm executable.

    mod: IRModule
        The input IRModule to be built.

    exec_mode: {"bytecode", "compiled"}
        The execution mode.

    Return
    ------
    leftover: IRModule
        Left over IRModule that may contain extra functions.
    """

    if exec_mode == "bytecode":
        return _ffi_api.VMCodeGen(builder, mod)  # type:ignore
    if exec_mode == "compiled":
        return _ffi_api.VMTIRCodeGen(builder, mod)  # type: ignore
    raise ValueError("Unknown exec_mode %s" % exec_mode)


def _autodetect_system_lib_req(target: tvm.target.Target, system_lib):
    """Automatically detect system lib requirement"""
    host = target if target.host is None else target.host
    if system_lib is None:
        system_lib = False
        if "wasm" in host.attrs.get("mtriple", ""):
            system_lib = True
    if system_lib:
        # use packed-func to avoid relay dep.
        return tvm.get_global_func("relay.backend.CreateRuntime")("cpp", {"system-lib": system_lib})
    return None


def _vmlink(
    builder: "relax.ExecBuilder",
    target: Union[str, tvm.target.Target],
    tir_mod: Optional[tvm.IRModule] = None,
    ext_libs: List[tvm.runtime.Module] = None,
    params: Optional[Dict[str, list]] = None,
    *,
    system_lib: Optional[bool] = None,
):
    """
    Internal codegen function to make executable.

    This function is only used for unit-testing purpoes.

    Use build instead.

    Parameters
    ----------
    builder: relax.ExecBuilder
        Builder used to collect executables.

    target : Union[str, tvm.target.Target]
        A build target which can have optional host side compilation target.

    tir_mod: IRModule
        The input TIR IRModule to be linked together.

    ext_libs:  List[tvm.runtime.Module]
        List of compiled external modules.

    params: Optional[Dict[str, list]]
        Extra parameter mappings.

    Returns
    -------
    ex: tvm.relax.Executable
        An executable that can be loaded by virtual machine.
    """
    if isinstance(target, str):
        target = tvm.target.Target(target)
    if params is None:
        params = {}
    if ext_libs is None:
        ext_libs = []
    lib = None
    if tir_mod is not None:
        lib = tvm.build(
            tir_mod, target=target, runtime=_autodetect_system_lib_req(target, system_lib)
        )
    return Executable(_ffi_api.VMLink(builder, target, lib, ext_libs, params))  # type: ignore


def build(
    mod: tvm.IRModule,
    target: Union[str, tvm.target.Target],
    params: Optional[Dict[str, list]] = None,
    exec_mode: str = "bytecode",
    *,
    system_lib: Optional[bool] = None,
) -> Executable:
    """
    Build an IRModule to VM executable.

    Parameters
    ----------
    mod: IRModule
        The input IRModule to be built.

    target : Union[str, tvm.target.Target]
        A build target which can have optional host side compilation target.

        When TVM compiles device specific program such as CUDA,
        we also need host(CPU) side code to interact with the driver
        to setup the dimensions and parameters correctly.
        host is used to specify the host side codegen target.
        By default, llvm is used if it is enabled,
        otherwise a stackvm interpreter is used.

    params: Optional[Dict[str, list]]
        Parameters for the input IRModule that will be bound.

    exec_mode: {"bytecode", "compiled"}
        The execution mode.

    system_lib: Optional[bool]
        Whether to build system lib that is being packed statically and
        auto registers generated functions to the system.
        By default auto detects based on the target.

    Returns
    -------
    ex: tvm.relax.Executable
        An executable that can be loaded by virtual machine.

    Example
    -------

    .. code-block:: python
        class InputModule:
            @R.function
            def foo(x: Tensor((3, 4), "float32"), y: Tensor((3, 4), "float32")):
                z = R.add(x, y)
                return z

        mod = InputModule
        target = tvm.target.Target("llvm", host="llvm")
        ex = relax.build(mod, target)
    """
    if isinstance(target, str):
        target = tvm.target.Target(target)

    passes = []
    passes.append(relax.transform.RewriteDataflowReshape())
    passes.append(relax.transform.ToNonDataflow())
    passes.append(relax.transform.RemovePurityChecking())
    passes.append(relax.transform.CallTIRRewrite())
    passes.append(relax.transform.StaticPlanBlockMemory())

    if tvm.transform.PassContext.current().config.get("relax.backend.use_cuda_graph", False):
        passes.append(relax.transform.RewriteCUDAGraph())

    passes.append(relax.transform.VMBuiltinLower())
    passes.append(relax.transform.VMShapeLower())
    passes.append(relax.transform.AttachGlobalSymbol())
    seq = tvm.transform.Sequential(passes)
    new_mod = seq(mod)

    # Extract external runtime modules if exist.
    attrs = dict(mod.attrs) if mod.attrs else {}

    ext_libs = attrs.get("external_mods", [])
    constants = attrs.get("const_name_to_constant", {})

    if params is not None:
        params.update(dict(constants))
    else:
        params = constants

    # builder collects the executable
    builder = relax.ExecBuilder()
    leftover_mod = _vmcodegen(builder, new_mod, exec_mode=exec_mode)
    tir_mod = _filter_tir(leftover_mod)
    return _vmlink(builder, target, tir_mod, ext_libs, params, system_lib=system_lib)


def _filter_tir(mod: tvm.IRModule) -> tvm.IRModule:
    tir_mod = IRModule({})
    tir_mod = tir_mod.with_attrs(mod.attrs)
    for gv in mod.get_global_vars():
        if isinstance(mod[gv], PrimFunc):
            tir_mod[gv] = mod[gv]
    return tir_mod
