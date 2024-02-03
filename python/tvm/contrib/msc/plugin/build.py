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
"""tvm.contrib.msc.plugin.build"""

import os
import sys
import subprocess

from typing import List, Dict, Any, Optional
from tvm.contrib.msc.core import utils as msc_utils
from tvm.contrib.msc.plugin.codegen import get_codegen
from .register import register_plugin


def _build_plugins(
    plugins: Dict[str, dict],
    frameworks: List[str],
    workspace: msc_utils.MSCDirectory = None,
    codegen_config: Optional[Dict[str, str]] = None,
    cpp_print_config: Optional[Dict[str, str]] = None,
    py_print_config: Optional[Dict[str, str]] = None,
    externs_dir: msc_utils.MSCDirectory = None,
    on_debug: bool = False,
):
    """Build the plugins

    Parameters
    ----------
    plugins: dict<str, dict>
        The plugins define.
    frameworks: list<str>
        The frameworks for plugin.
    workspace: MSCDirectory
        The workspace folder.
    codegen_config: dict<string, string>
        The config to generate code.
    cpp_print_config: dict<string, string>
        The config to print cpp code.
    py_print_config: dict<string, string>
        The config to print python code.
    externs_dir: MSCDirectory
        The extern sources folder.
    on_debug: bool
        Whether to debug the building.
    """

    workspace = workspace or msc_utils.msc_dir("msc_plugin")

    # register the plugins
    extern_sources, extern_libs, ops_info = {}, {}, {}
    for name, plugin in plugins.items():
        sources, libs, info = register_plugin(name, plugin, externs_dir)
        extern_sources.update(sources)
        extern_libs.update(libs)
        ops_info[name] = info
    # build plugins for frameworks
    codegens = {}
    for framework in frameworks:
        codegen = get_codegen(
            framework,
            workspace,
            codegen_config,
            cpp_print_config=cpp_print_config,
            py_print_config=py_print_config,
            extern_sources=extern_sources,
            extern_libs=extern_libs,
            on_debug=on_debug,
        )
        if not codegen.libs_built():
            codegen.build_libs()
        if codegen.need_manager and not codegen.manager_built():
            codegen.build_manager(ops_info)
        codegens[framework] = codegen
    return codegens


def build_plugins(
    plugins: Dict[str, dict],
    frameworks: List[str],
    workspace: msc_utils.MSCDirectory = None,
    codegen_config: Optional[Dict[str, str]] = None,
    cpp_print_config: Optional[Dict[str, str]] = None,
    py_print_config: Optional[Dict[str, str]] = None,
    externs_dir: msc_utils.MSCDirectory = None,
    on_debug: bool = False,
) -> Dict[str, Any]:
    """Build the plugins and load plugin manager

    Parameters
    ----------
    plugins: dict<str, dict>
        The plugins define.
    frameworks: list<str>
        The frameworks for plugin.
    workspace: MSCDirectory
        The workspace folder.
    codegen_config: dict<string, string>
        The config to generate code.
    cpp_print_config: dict<string, string>
        The config to print cpp code.
    py_print_config: dict<string, string>
        The config to print python code.
    externs_dir: MSCDirectory
        The extern sources folder.
    on_debug: bool
        Whether to debug the building.

    Returns
    -------
    managers: dict<string, PluginManager>
        The plugin managers.
    """

    codegens = _build_plugins(
        plugins,
        frameworks,
        workspace,
        codegen_config=codegen_config,
        cpp_print_config=cpp_print_config,
        py_print_config=py_print_config,
        externs_dir=externs_dir,
        on_debug=on_debug,
    )
    managers = {}
    for name, codegen in codegens.items():
        manager_file = codegen.manager_folder.relpath("manager.py")
        manager_cls = msc_utils.load_callable(manager_file + ":PluginManager")
        managers[name] = manager_cls(codegen.output_folder.path)
    return managers


def pack_plugins(
    plugins: Dict[str, dict],
    frameworks: List[str],
    project_name: str = "msc_plugin",
    codegen_config: Optional[Dict[str, str]] = None,
    cpp_print_config: Optional[Dict[str, str]] = None,
    py_print_config: Optional[Dict[str, str]] = None,
    externs_dir: msc_utils.MSCDirectory = None,
    setup_config: Optional[Dict[str, str]] = None,
    on_debug: bool = False,
) -> str:
    """Build the plugins and build to wheel

    Parameters
    ----------
    plugins: dict<str, dict>
        The plugins define.
    frameworks: list<str>
        The frameworks for plugin.
    project_name: str
        The project name
    codegen_config: dict<string, string>
        The config to generate code.
    cpp_print_config: dict<string, string>
        The config to print cpp code.
    py_print_config: dict<string, string>
        The config to print python code.
    externs_dir: MSCDirectory
        The extern sources folder.
    setup_config: dict<string, string>
        The config to setup wheel.
    on_debug: bool
        Whether to debug the building.

    Returns
    -------
    wheel_path: str
        The file path of wheel.
    """

    project_dir = msc_utils.msc_dir(project_name)
    workspace = project_dir.create_dir(project_name)
    codegens = _build_plugins(
        plugins,
        frameworks,
        workspace,
        codegen_config=codegen_config,
        cpp_print_config=cpp_print_config,
        py_print_config=py_print_config,
        externs_dir=externs_dir,
        on_debug=on_debug,
    )
    # add init files
    init_code = """# Licensed to the Apache Software Foundation (ASF) under one
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

from .manager import *
"""
    with open(workspace.relpath("__init__.py"), "w") as f:
        f.write(init_code)
    for name in codegens:
        with open(workspace.create_dir(name).relpath("__init__.py"), "w") as f:
            f.write(init_code)

    # add setup file
    if setup_config:
        setup_config_str = "\n    " + "\n    ".join(
            ["{} = {},".format(k, v) for k, v in setup_config.items()]
        )
    else:
        setup_config_str = ""
    setup_code = """
import os
import shutil

from setuptools import find_packages, setup
from setuptools.dist import Distribution

project_name = "{0}"
data_files = []
for framework in [{2}]:
    for folder in ["lib", "include"]:
        src_path = os.path.join(project_name, framework, folder)
        data_files.append(
            (
                os.path.join(project_name, framework, folder),
                [os.path.join(src_path, f) for f in os.listdir(src_path)],
            ),
        )

class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

    def is_pure(self):
        return False

setup(
    name="{0}"{1},
    packages=find_packages(),
    distclass=BinaryDistribution,
    data_files=data_files
)

shutil.rmtree("build")
shutil.rmtree("{0}.egg-info")
""".format(
        project_name, setup_config_str, ",".join(['"{}"'.format(f) for f in frameworks])
    )
    with open(project_dir.relpath("setup.py"), "w") as f:
        f.write(setup_code)

    # build the wheel
    with project_dir:
        command = "{} setup.py bdist_wheel".format(sys.executable)
        with open("build.log", "w") as log_f:
            process = subprocess.Popen(command, stdout=log_f, stderr=log_f, shell=True)
        process.wait()
        assert (
            process.returncode == 0
        ), "Failed to build wheel under {}, check build.log for detail".format(os.getcwd())
    dist_dir = project_dir.create_dir("dist")
    files = list(dist_dir.listdir())
    assert len(files) == 1 and files[0].endswith(
        ".whl"
    ), "Failed to build wheel, no .whl found @ " + str(dist_dir.path)
    return dist_dir.relpath(files[0])
