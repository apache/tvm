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

"""Defines functions for exporting to Model Library Format."""

import datetime
import json
import os
import re
import tarfile

from ..contrib import utils
from ..relay.backend import executor_factory
from ..relay import param_dict

# This should be kept identical to runtime::symbol::tvm_module_main
MAIN_FUNC_NAME_STR = "__tvm_main__"


class UnsupportedInModelLibraryFormatError(Exception):
    """Raised when export_model_library_format does not support the given Module tree."""


def _populate_codegen_dir(mod, codegen_dir: str, module_name: str = None):
    """Populate the codegen sub-directory as part of a Model Library Format export.

    Parameters
    ----------
    mod : tvm.runtime.Module
        Module which should be written to codegen_dir.
    codegen_dir : str
        Path to the codegen directory on disk.
    module_name: Optional[str]
        Name used to prefix the generated source files

    """
    dso_modules = mod._collect_dso_modules()
    dso_module_handles = [m.handle.value for m in dso_modules]
    non_dso_modules = mod._collect_from_import_tree(lambda m: m not in dso_modules)
    if non_dso_modules:
        raise UnsupportedInModelLibraryFormatError(
            f"Don't know how to export non-c or non-llvm modules; found: {non_dso_modules!r}"
        )

    mod_indices = {"lib": 0, "src": 0}
    host_codegen_dir = os.path.join(codegen_dir, "host")
    lib_name = f"{module_name}_lib" if module_name else "lib"

    for dso_mod in dso_modules:
        if dso_mod.type_key == "c":
            index = mod_indices["src"]
            mod_indices["src"] += 1
            parent_dir = os.path.join(host_codegen_dir, "src")
            file_name = os.path.join(parent_dir, f"{lib_name}{index}.c")
        elif dso_mod.type_key == "llvm":
            index = mod_indices["lib"]
            mod_indices["lib"] += 1
            parent_dir = os.path.join(host_codegen_dir, "lib")
            file_name = os.path.join(parent_dir, f"{lib_name}{index}.o")
        else:
            assert (
                False
            ), f"do not expect module with type_key={mod.type_key} from _collect_dso_modules"

        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        dso_mod.save(file_name)


def _build_memory_map(mod):
    ret = dict()
    if isinstance(mod, executor_factory.GraphExecutorFactoryModule):
        ret["sids"] = _build_sid_map(mod.graph_json)
    ret["functions"] = _build_function_memory_map(mod.function_metadata)
    return ret


def _build_sid_map(graph_json):
    """Build a simpler storage id info map from graph JSON.

    Parameters
    ----------
    graph_json : str
        String representation of the graph_json created from tvm.relay.build().

    Returns
    -------
    list :
        A list with one entry per storage id describing that memory.
    """
    graph = json.loads(graph_json)
    seen_storage_ids = set()
    memory_map = []
    for node_id, storage_id in enumerate(graph["attrs"]["storage_id"][1]):
        if storage_id in seen_storage_ids:
            continue

        seen_storage_ids.add(storage_id)
        num_elements = 1
        for dim in graph["attrs"]["shape"][1][storage_id]:
            num_elements *= dim

        dltype = graph["attrs"]["dltype"][1][storage_id]
        m = re.match(r"^[a-zA-Z]+([0-9]+)$", dltype)
        assert m, f"Exported graph contains unknown dltype {dltype}"

        elem_bits = int(m.group(1))

        map_entry = {
            "storage_id": storage_id,
            "size_bytes": (num_elements * elem_bits + 7) // 8,
        }
        if node_id in graph["arg_nodes"]:
            map_entry["input_binding"] = graph["nodes"][node_id]["name"]

        memory_map.append(map_entry)

    return memory_map


def _build_function_memory_map(function_metadata):
    """Build a simple map that shows how much workspace is required to execute
    each primitive function. The main_func describes how much memory is required
    to execute the main control code.

    Parameters
    ----------
    function_metadata : Map<String, FunctionInfo>
        This contains all the compiled metadata on a function basis

    Returns
    -------
    dict :
        This will have two entries:
        1.) A list with one entry per function describing local memory it is using.
        2.) A global memory requirement if all functions are executed sequentially
    """
    device_max_workspace = dict()
    main_func_metadata = function_metadata[MAIN_FUNC_NAME_STR]
    num_targets = len(main_func_metadata.workspace_sizes.items())
    func_entries = []
    target_local_entries = dict()
    for i in range(num_targets):
        target = main_func_metadata.workspace_sizes.items()[i][0]
        device_max_workspace[target] = 0
        for func_name, finfo in function_metadata.items():
            if func_name == MAIN_FUNC_NAME_STR:
                continue
            target_local_entries[func_name] = list()

        for func_name, finfo in function_metadata.items():
            if func_name == MAIN_FUNC_NAME_STR:
                continue
            assert len(finfo.constant_sizes.items()) == num_targets
            assert len(finfo.io_sizes.items()) == num_targets
            target = finfo.workspace_sizes.items()[i][0]
            workspace_size = finfo.workspace_sizes.items()[i][1]
            target_entry = {
                "device": int(target.kind.device_type),
                "workspace_size_bytes": int(workspace_size),
            }
            target_local_entries[func_name].append(target_entry)
            if workspace_size > device_max_workspace[target]:
                device_max_workspace[target] = workspace_size

    for func_name, target_entries_ in target_local_entries.items():
        func_entry = {
            "function_name": str(func_name),
            "workspace": target_entries_,
        }
        func_entries.append(func_entry)

    target_main_entries = list()
    for i in range(num_targets):
        target = main_func_metadata.workspace_sizes.items()[i][0]
        main_func_local_workspace = main_func_metadata.workspace_sizes.items()[i][1]
        main_func_constants = main_func_metadata.constant_sizes.items()[i][1]
        main_func_io = main_func_metadata.io_sizes.items()[i][1]
        target_main_entries.append(
            {
                "device": int(target.kind.device_type),
                "workspace_size_bytes": int(device_max_workspace[target])
                + int(main_func_local_workspace),
                "constants_size_bytes": int(main_func_constants),
                "io_size_bytes": int(main_func_io),
            }
        )

    ret = {
        "operator_functions": func_entries,
        "main": target_main_entries,
    }
    return ret


def export_model_library_format(mod: executor_factory.ExecutorFactoryModule, file_name):
    """Export the build artifact in Model Library Format.

    This function creates a .tar archive containing the build artifacts in a standardized
    layout. It's intended to allow downstream automation to build TVM artifacts against the C
    runtime.

    Parameters
    ----------
    mod : tvm.relay.backend.executor_factory.ExecutorFactoryModule
        The return value of tvm.relay.build, which will be exported into Model Library Format.
    file_name : str
        Path to the .tar archive to generate.

    Returns
    -------
    file_name : str
        The path to the generated .tar archive.
    """
    tempdir = utils.tempdir()
    is_aot = isinstance(mod, executor_factory.AOTExecutorFactoryModule)
    runtime = ["aot"] if is_aot else ["graph"]

    metadata = {
        "version": 3,
        "model_name": mod.libmod_name,
        "export_datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%SZ"),
        "memory": _build_memory_map(mod),
        "target": {int(k): str(v) for k, v in mod.target.items()},
        "runtimes": runtime,
    }

    with open(tempdir.relpath("metadata.json"), "w") as json_f:
        json.dump(metadata, json_f, indent=2, sort_keys=True)

    codegen_dir_path = tempdir.relpath("codegen")
    os.mkdir(codegen_dir_path)
    _populate_codegen_dir(mod.lib, codegen_dir_path, mod.libmod_name)

    parameters_dir_path = tempdir.relpath("parameters")
    os.mkdir(parameters_dir_path)
    param_filename = os.path.join(parameters_dir_path, f"{mod.libmod_name}.params")
    with open(param_filename, "wb") as f:
        f.write(param_dict.save_param_dict(mod.params))

    with open(tempdir.relpath("relay.txt"), "w") as f:
        f.write(str(mod.ir_mod))

    if not is_aot:
        graph_config_dir_path = tempdir.relpath(os.path.join("runtime-config", "graph"))
        os.makedirs(graph_config_dir_path)
        with open(os.path.join(graph_config_dir_path, "graph.json"), "w") as f:
            f.write(mod.get_executor_config())

    with tarfile.open(file_name, "w") as tar_f:

        def reset(tarinfo):
            tarinfo.uid = tarinfo.gid = 0
            tarinfo.uname = tarinfo.gname = "root"
            return tarinfo

        tar_f.add(tempdir.temp_dir, arcname=".", filter=reset)

    return file_name
