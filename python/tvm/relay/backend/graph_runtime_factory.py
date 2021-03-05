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
"""Graph runtime factory."""
import datetime
import os
import json
import re
import tarfile
import warnings
from ...contrib import utils
from ..._ffi.base import string_types
from ..._ffi.registry import get_global_func
from ...runtime import ndarray
from .. import param_dict


class GraphRuntimeFactoryModule:
    """Graph runtime factory module.
    This is a module of graph runtime factory

    Parameters
    ----------
    graph_json_str : str
        The graph to be deployed in json format output by graph compiler.
        The graph can contain operator(tvm_op) that points to the name of
        PackedFunc in the libmod.
    target : tvm.Target
        The Target used to build this module.
    libmod : tvm.Module
        The module of the corresponding function
    libmod_name: str
        The name of module
    params : dict of str to NDArray
        The parameters of module
    """

    def __init__(self, ir_mod, target, graph_json_str, libmod, libmod_name, params):
        assert isinstance(graph_json_str, string_types)
        fcreate = get_global_func("tvm.graph_runtime_factory.create")
        args = []
        for k, v in params.items():
            args.append(k)
            args.append(ndarray.array(v))
        self.ir_mod = ir_mod
        self.target = target
        self.module = fcreate(graph_json_str, libmod, libmod_name, *args)
        self.graph_json = graph_json_str
        self.lib = libmod
        self.libmod_name = libmod_name
        self.params = params
        self.iter_cnt = 0

    def export_library(self, file_name, fcompile=None, addons=None, **kwargs):
        return self.module.export_library(file_name, fcompile, addons, **kwargs)

    def _build_memory_map(self):
        graph = json.loads(self.graph_json)

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

    def export_model_library_format(self, file_name):
        """Export the build artifact in Model Library Format.

        This function creates a .tar archive containing the build artifacts in a standardized
        layout. It's intended to allow downstream automation to build TVM artifacts against the C
        runtime.

        Parameters
        ----------
        file_name : str
            Path to the .tar archive to generate.
        """
        tempdir = utils.tempdir()
        metadata = {
            "version": 1,
            "model_name": self.libmod_name,
            "export_datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%SZ"),
            "memory": self._build_memory_map(),
            "target": {int(k): str(v) for k, v in self.target.items()},
            "runtimes": ["graph"],
        }
        with open(tempdir.relpath("metadata.json"), "w") as json_f:
            json.dump(metadata, json_f, indent=2, sort_keys=True)

        codegen_dir_path = tempdir.relpath("codegen")
        print("codegen_dir", codegen_dir_path)
        os.mkdir(codegen_dir_path)
        self.lib.export_model_library_format(codegen_dir_path)
        parameters_dir_path = tempdir.relpath("parameters")
        os.mkdir(parameters_dir_path)
        param_filename = os.path.join(parameters_dir_path, f"{self.libmod_name}.params")
        with open(param_filename, "wb") as f:
            f.write(param_dict.save_param_dict(self.params))
        with open(tempdir.relpath("relay.txt"), "w") as f:
            f.write(str(self.ir_mod))
        graph_config_dir_path = tempdir.relpath(os.path.join("runtime-config", "graph"))
        os.makedirs(graph_config_dir_path)
        with open(os.path.join(graph_config_dir_path, "graph.json"), "w") as f:
            f.write(self.graph_json)
        with tarfile.open(file_name, "w") as tar_f:

            def reset(tarinfo):
                tarinfo.uid = tarinfo.gid = 0
                tarinfo.uname = tarinfo.gname = "root"
                return tarinfo

            tar_f.add(tempdir.temp_dir, arcname=".", filter=reset)

    # Sometimes we want to get params explicitly.
    # For example, we want to save its params value to
    # an independent file.
    def get_params(self):
        return self.params

    def get_json(self):
        return self.graph_json

    def get_lib(self):
        return self.lib

    def __getitem__(self, item):
        return self.module.__getitem__(item)

    def __iter__(self):
        warnings.warn(
            "legacy graph runtime behavior of producing json / lib / params will be "
            "removed in the next release."
            " Please see documents of tvm.contrib.graph_runtime.GraphModule for the "
            " new recommended usage.",
            DeprecationWarning,
            2,
        )
        return self

    def __next__(self):
        if self.iter_cnt > 2:
            raise StopIteration

        objs = [self.graph_json, self.lib, self.params]
        obj = objs[self.iter_cnt]
        self.iter_cnt += 1
        return obj
