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
"""tvm.contrib.msc.core.codegen.codegen"""

import os
import subprocess
from typing import Dict, List, Optional

import tvm
from tvm.contrib.msc.core.utils.namespace import MSCFramework
from tvm.contrib.msc.plugin import _ffi_api
from tvm.contrib.msc.core import utils as msc_utils
from .sources import get_plugin_sources


class BasePluginCodeGen(object):
    """Manager class to generate codes and build plugin

    Parameters
    ----------
    workspace: MSCDirectory
        The workspace folder.
    codegen_config: dict<string, string>
        The config to generate code.
    cpp_print_config: dict<string, string>
        The config to print cpp code.
    py_print_config: dict<string, string>
        The config to print python code.
    extern_sources: dict<string, string>
        The depend source files.
    extern_libs: dict<string, string>
        The depend lib files.
    on_debug: bool
        Whether to debug the building.
    """

    def __init__(
        self,
        workspace: msc_utils.MSCDirectory,
        codegen_config: Optional[Dict[str, str]] = None,
        cpp_print_config: Optional[Dict[str, str]] = None,
        py_print_config: Optional[Dict[str, str]] = None,
        extern_sources: Dict[str, str] = None,
        extern_libs: Dict[str, str] = None,
        on_debug: bool = False,
    ):
        self._codegen_config = msc_utils.copy_dict(codegen_config)
        self._cpp_print_config = msc_utils.dump_dict(cpp_print_config)
        self._py_print_config = msc_utils.dump_dict(py_print_config)
        self._build_folder = workspace.create_dir(
            "source_" + self.framework, keep_history=on_debug, cleanup=not on_debug
        )
        self._output_folder = workspace.create_dir(self.framework)
        self._extern_sources = extern_sources or {}
        self._extern_libs = extern_libs or {}
        self.setup()

    def setup(self):
        """Set up the codegen"""

        self._lib_folder = self._output_folder.create_dir("lib")
        self._manager_folder = self._output_folder
        self._libs = [os.path.basename(l) for l in self._extern_libs.values()]
        self._libs.extend([os.path.basename(l) for l in self._lib_folder.listdir()])
        self._project_name = "msc_{}_plugin".format(self.framework)
        self._codegen_config.update(
            {
                "install_dir": self._output_folder.path,
                "project_name": self._project_name,
                "version": msc_utils.get_version(self.framework),
            }
        )

    def libs_built(self) -> bool:
        """Check if the libs are built

        Returns
        -------
        libs_built: bool
            Whether libs are built.
        """

        return any(self._project_name in f for f in self._lib_folder.listdir())

    def build_libs(self) -> List[str]:
        """Generate source and build the lib

        Returns
        -------
        paths: list<str>
            The lib file paths.
        """

        codegen_config = msc_utils.dump_dict(self._codegen_config)
        sources = self.source_getter(codegen_config, self._cpp_print_config, "build")
        with self._build_folder as folder:
            # add depends
            with folder.create_dir("src") as src_folder:
                for name, file in self._extern_sources.items():
                    src_folder.copy(file, name)
                for name, source in get_plugin_sources().items():
                    src_folder.add_file(name, source)
                for name, source in sources.items():
                    if name == "CMakeLists.txt":
                        folder.add_file(name, source)
                    else:
                        src_folder.add_file(name, source)
            with folder.create_dir("build"):
                command = "cmake ../ && make"
                with open("codegen.log", "w") as log_f:
                    process = subprocess.Popen(command, stdout=log_f, stderr=log_f, shell=True)
                process.wait()
                assert (
                    process.returncode == 0
                ), "Failed to build plugin under {}, check codegen.log for detail".format(
                    os.getcwd()
                )
            self._libs.extend([os.path.basename(l) for l in self._lib_folder.listdir()])
        return self._lib_folder.listdir(as_abs=True)

    def manager_built(self) -> bool:
        """Check if the manager are built

        Returns
        -------
        manager_built: bool
            Whether manager is built.
        """

        return os.path.isfile(self._manager_folder.relpath("manager.py"))

    def build_manager(self, ops_info: dict) -> List[str]:
        """Generate manager source for plugin

        Parameters
        ----------
        ops_info: dict
            The info of ops.

        Returns
        -------
        paths: list<str>
            The manager file paths.
        """

        self._codegen_config["libs"] = self._libs
        self._codegen_config["ops_info"] = {n: msc_utils.dump_dict(i) for n, i in ops_info.items()}
        codegen_config = msc_utils.dump_dict(self._codegen_config)
        sources = self.source_getter(codegen_config, self._py_print_config, "manager")
        manager_files = []
        with self._manager_folder as folder:
            for name, source in sources.items():
                manager_files.append(folder.add_file(name, source))
        return manager_files

    @property
    def source_getter(self):
        raise NotImplementedError("source_getter is not supported for Base codegen")

    @property
    def need_manager(self):
        return True

    @property
    def framework(self):
        return MSCFramework.MSC

    @property
    def output_folder(self):
        return self._output_folder

    @property
    def lib_folder(self):
        return self._lib_folder

    @property
    def manager_folder(self):
        return self._manager_folder


class TVMPluginCodegen(BasePluginCodeGen):
    """Plugin codegen for tvm"""

    def setup(self):
        """Set up the codegen"""

        super().setup()
        tvm_root = os.path.dirname(os.path.dirname(tvm.__path__[0]))
        self._codegen_config.update(
            {"need_convert": False, "with_runtime": True, "tvm_root": tvm_root}
        )

    @property
    def source_getter(self):
        return _ffi_api.GetTVMPluginSources

    @property
    def framework(self):
        return MSCFramework.TVM


class TorchPluginCodegen(BasePluginCodeGen):
    """Plugin codegen for torch"""

    def setup(self):
        """Set up the codegen"""
        # pylint: disable=import-outside-toplevel
        import torch.utils

        super().setup()
        self._codegen_config.update(
            {
                "need_convert": True,
                "with_runtime": False,
                "torch_prefix": torch.utils.cmake_prefix_path,
            }
        )

    @property
    def source_getter(self):
        return _ffi_api.GetTorchPluginSources

    @property
    def framework(self):
        return MSCFramework.TORCH


class TensorRTPluginCodegen(BasePluginCodeGen):
    """Plugin codegen for tensorrt"""

    def setup(self):
        """Set up the codegen"""
        # pylint: disable=import-outside-toplevel
        from tvm.contrib.msc.framework.tensorrt import _ffi_api as _trt_api

        super().setup()
        self._codegen_config.update(
            {
                "need_convert": False,
                "with_runtime": False,
                "tensorrt_root": _trt_api.GetTensorRTRoot(),
            }
        )

    @property
    def source_getter(self):
        return _ffi_api.GetTensorRTPluginSources

    @property
    def framework(self):
        return MSCFramework.TENSORRT


def get_codegen(
    framework: str,
    workspace: msc_utils.MSCDirectory,
    codegen_config: Optional[Dict[str, str]] = None,
    cpp_print_config: Optional[Dict[str, str]] = None,
    py_print_config: Optional[Dict[str, str]] = None,
    extern_sources: Dict[str, str] = None,
    extern_libs: Dict[str, str] = None,
    on_debug: bool = False,
):
    """Create codegen for framework

    Parameters
    ----------
    framework: str
        THe framework for the plugin.
    workspace: MSCDirectory
        The workspace folder.
    codegen_config: dict<string, string>
        The config to generate code.
    cpp_print_config: dict<string, string>
        The config to print cpp code.
    py_print_config: dict<string, string>
        The config to print python code.
    extern_sources: dict<string, string>
        The depend source files.
    extern_libs: dict<string, string>
        The depend lib files.
    on_debug: bool
        Whether to debug the building.
    """

    codegen_cls = None
    if framework == MSCFramework.TVM:
        codegen_cls = TVMPluginCodegen
    elif framework == MSCFramework.TORCH:
        codegen_cls = TorchPluginCodegen
    elif framework == MSCFramework.TENSORRT:
        codegen_cls = TensorRTPluginCodegen
    else:
        raise NotImplementedError(
            "framework {} is not support for plugin codegen".format(framework)
        )
    return codegen_cls(
        workspace,
        codegen_config=codegen_config,
        cpp_print_config=cpp_print_config,
        py_print_config=py_print_config,
        extern_sources=extern_sources,
        extern_libs=extern_libs,
        on_debug=on_debug,
    )
