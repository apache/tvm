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
from ..relay.backend import graph_executor_factory
from ..relay import param_dict


class UnsupportedInModelLibraryFormatError(Exception):
    """Raised when export_model_library_format does not support the given Module tree."""


def _populate_codegen_dir(mod, codegen_dir: str):
    """Populate the codegen sub-directory as part of a Model Library Format export.

    Parameters
    ----------
    mod : tvm.runtime.Module
        Module which should be written to codegen_dir.
    codegen_dir : str
        Path to the codegen directory on disk.
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
    for dso_mod in dso_modules:
        if dso_mod.type_key == "c":
            index = mod_indices["src"]
            mod_indices["src"] += 1
            parent_dir = os.path.join(host_codegen_dir, "src")
            file_name = os.path.join(parent_dir, f"lib{index}.c")
        elif dso_mod.type_key == "llvm":
            index = mod_indices["lib"]
            mod_indices["lib"] += 1
            parent_dir = os.path.join(host_codegen_dir, "lib")
            file_name = os.path.join(parent_dir, f"lib{index}.o")
        else:
            assert (
                False
            ), f"do not expect module with type_key={mod.type_key} from _collect_dso_modules"

        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        dso_mod.save(file_name)


def _build_memory_map(graph_json):
    """Build a simpler memory map from graph JSON.

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


def export_model_library_format(mod: graph_executor_factory.GraphExecutorFactoryModule, file_name):
    """Export the build artifact in Model Library Format.

    This function creates a .tar archive containing the build artifacts in a standardized
    layout. It's intended to allow downstream automation to build TVM artifacts against the C
    runtime.

    Parameters
    ----------
    mod : tvm.relay.backend.graph_executor_factory.GraphExecutorFactoryModule
        The return value of tvm.relay.build, which will be exported into Model Library Format.
    file_name : str
        Path to the .tar archive to generate.
    """
    tempdir = utils.tempdir()
    metadata = {
        "version": 1,
        "model_name": mod.libmod_name,
        "export_datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%SZ"),
        "memory": _build_memory_map(mod.graph_json),
        "target": {int(k): str(v) for k, v in mod.target.items()},
        "runtimes": ["graph"],
    }
    with open(tempdir.relpath("metadata.json"), "w") as json_f:
        json.dump(metadata, json_f, indent=2, sort_keys=True)

    codegen_dir_path = tempdir.relpath("codegen")
    os.mkdir(codegen_dir_path)
    _populate_codegen_dir(mod.lib, codegen_dir_path)

    parameters_dir_path = tempdir.relpath("parameters")
    os.mkdir(parameters_dir_path)
    param_filename = os.path.join(parameters_dir_path, f"{mod.libmod_name}.params")
    with open(param_filename, "wb") as f:
        f.write(param_dict.save_param_dict(mod.params))

    with open(tempdir.relpath("relay.txt"), "w") as f:
        f.write(str(mod.ir_mod))

    graph_config_dir_path = tempdir.relpath(os.path.join("runtime-config", "graph"))
    os.makedirs(graph_config_dir_path)
    with open(os.path.join(graph_config_dir_path, "graph.json"), "w") as f:
        f.write(mod.graph_json)

    with tarfile.open(file_name, "w") as tar_f:

        def reset(tarinfo):
            tarinfo.uid = tarinfo.gid = 0
            tarinfo.uname = tarinfo.gname = "root"
            return tarinfo

        tar_f.add(tempdir.temp_dir, arcname=".", filter=reset)
