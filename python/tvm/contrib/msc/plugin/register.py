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
"""tvm.contrib.msc.plugin.register"""

import os
from typing import Dict

import tvm
from tvm.contrib.msc.core import utils as msc_utils
from tvm.contrib.msc.core import _ffi_api


def register_plugin(
    name: str, plugin: dict, externs_dir: msc_utils.MSCDirectory = None
) -> Dict[str, str]:
    """Register a plugin

    Parameters
    ----------
    name: str
        The name of the plugin.
    plugin: dict
        The define of a plugin.
    externs_dir: MSCDirectory
        The extern sources folder.

    Returns
    -------
    depend_files: dict<str, str>
        The depend file paths.
    """

    plugin = {"name": name, **msc_utils.load_dict(plugin)}
    assert "externs" in plugin, "externs are needed to build plugin"
    # check device compute
    remove_externs = set()
    for extern in plugin["externs"]:
        if extern == "cuda_compute" and not tvm.cuda().exist:
            remove_externs.add(extern)
    if remove_externs:
        plugin["externs"] = {k: v for k, v in plugin["externs"].items() if k not in remove_externs}
    externs = plugin["externs"]

    def _check_file(info: dict, key: str) -> str:
        if key not in info:
            return None
        file_path = info[key]
        if os.path.abspath(file_path) != file_path:
            assert externs_dir, "externs_dir is need to find file " + str(file_path)
            file_path = externs_dir.relpath(file_path)
        assert os.path.isfile(file_path), "Can not find externs file " + str(file_path)
        info[key] = os.path.basename(file_path)
        return file_path

    # find depend files
    extern_sources, extern_libs = {}, {}
    for info in externs.values():
        for key in ["header", "source"]:
            file_path = _check_file(info, key)
            if file_path:
                extern_sources[os.path.basename(file_path)] = file_path
        file_path = _check_file(info, "lib")
        if file_path:
            extern_libs[os.path.basename(file_path)] = file_path
    _ffi_api.RegisterPlugin(name, msc_utils.dump_dict(plugin))
    # remove needless keys
    for key in ["support_dtypes", "externs"]:
        plugin.pop(key)
    plugin["inputs"] = [{"name": i["name"]} for i in plugin["inputs"]]
    plugin["outputs"] = [{"name": o["name"]} for o in plugin["outputs"]]
    return extern_sources, extern_libs, plugin
