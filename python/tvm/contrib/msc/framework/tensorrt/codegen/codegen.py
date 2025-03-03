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
"""tvm.contrib.msc.framework.tensorrt.codegen.codegen"""

import os
import subprocess
from typing import Dict, Optional, List, Union, Any
import numpy as np

import tvm
from tvm.contrib.msc.core.ir import MSCGraph
from tvm.contrib.msc.core.codegen import CodeGen
from tvm.contrib.msc.core.utils import MSCFramework
from tvm.contrib.msc.core import utils as msc_utils
from tvm.contrib.msc.framework.tensorrt import _ffi_api
from .sources import get_trt_sources
from .utils import write_weight


def to_sub_tensorrt(
    graph: MSCGraph,
    weights: Dict[str, tvm.nd.array],
    codegen_config: Optional[Dict[str, str]] = None,
    print_config: Optional[Dict[str, str]] = None,
    build_folder: msc_utils.MSCDirectory = None,
    output_folder: msc_utils.MSCDirectory = None,
    plugin: Any = None,
) -> str:
    """Change MSCGraph to TensorRT engine file.

    Parameters
    ----------
    graph: tvm.contrib.msc.core.ir.MSCGraph
        The translated graph.
    weights: dict of <string:tvm.ndarray>
        The parameters of the IRModule.
    codegen_config: dict
        The config for codegen.
    print_config: dict
        The config for print.
    build_folder: MSCDirectory
        The folder for saving sources and datas.
    export_folder: MSCDirectory
        The folder for saving outputs.
    plugin: PluginManager
        The plugin manager.

    Returns
    -------
    engine: str
        The engine file.
    """

    codegen_config = msc_utils.copy_dict(codegen_config)
    codegen_config["version"] = msc_utils.get_version(MSCFramework.TENSORRT)
    if "tensorrt_root" not in codegen_config:
        codegen_config["tensorrt_root"] = _ffi_api.GetTensorRTRoot()
    build_folder = build_folder or msc_utils.msc_dir(keep_history=False, cleanup=True)
    output_folder = output_folder or msc_utils.msc_dir("msc_output")
    depends = {}
    if "range_file" in codegen_config:
        range_file = codegen_config["range_file"]
        codegen_config["range_file"] = os.path.basename(range_file)
        depends[codegen_config["range_file"]] = {"src": range_file, "copy_back": True}

    def _create_depends(folder: msc_utils.MSCDirectory) -> str:
        if weights:
            # gather weights
            engine_wts = {}
            for node in graph.get_nodes():
                for weight in node.get_weights().values():
                    engine_wts[weight.name] = weights[weight.name]
                if node.optype in ("nn.conv2d", "msc.linear"):
                    weight = node.weight_at("weight")
                    bias = np.zeros([weight.dim_at("O")], dtype=weight.dtype_name)
                    engine_wts[node.name + ".bias"] = bias
            # write weights file
            with open(folder.relpath(graph.name + ".wts"), "w") as f:
                f.write("{}\n".format(len(engine_wts)))
                for name, data in engine_wts.items():
                    write_weight(name, msc_utils.cast_array(data), f)
        # copy plugin
        if plugin:
            plugin.copy_libs("plugin_lib")
            plugin.copy_includes("plugin")
        # save utils sources
        with folder.create_dir("utils") as utils_folder:
            for name, source in get_trt_sources().items():
                utils_folder.add_file(name, source)
        # copy depends
        for path, info in depends.items():
            if os.path.exists(info["src"]):
                folder.copy(info["src"], path)

    def _build_engine(engine_name: str, folder: msc_utils.MSCDirectory) -> str:
        with open("engine.log", "w") as log_f:
            process = subprocess.Popen("./" + engine_name, stdout=log_f, stderr=log_f, shell=True)
        process.wait()
        assert (
            process.returncode == 0
        ), "Failed to test engine {} under {}, check engine.log for detail".format(
            engine_name, os.getcwd()
        )
        for path, info in depends.items():
            if info.get("copy_back", False) and os.path.exists(path):
                folder.copy(path, info["src"])
        return folder.move(engine_name + ".trt", output_folder.relpath(engine_name + ".trt"))

    with build_folder as folder:
        sub_folder = folder.create_dir(graph.name)
        if plugin:
            codegen_config["extern_libs"] = [
                sub_folder.create_dir("plugin_lib").relpath(f) for f in plugin.list_libs()
            ]
        codegen = CodeGen(
            graph,
            _ffi_api.GetTensorRTSources,
            codegen_config,
            print_config,
            sub_folder,
            code_format="cpp",
        )
        engine_file = codegen.load([], pre_load=_create_depends, post_load=_build_engine)
    return {
        "graph_json": graph.to_json(),
        "graph_name": graph.name,
        "engine": engine_file,
    }


def to_tensorrt(
    mod: tvm.IRModule,
    graphs: List[MSCGraph],
    weights: Dict[str, tvm.nd.array],
    codegen_configs: Optional[Union[Dict[str, str], List[Dict[str, str]]]] = None,
    print_configs: Optional[Union[Dict[str, str], List[Dict[str, str]]]] = None,
    extra_options: Optional[Union[Dict[str, str], List[Dict[str, str]]]] = None,
    build_folder: msc_utils.MSCDirectory = None,
    output_folder: msc_utils.MSCDirectory = None,
    plugin: Any = None,
) -> Dict[str, str]:
    """Change all MSCGraphs to TensorRT engine files.

    Parameters
    ----------
    mod: IRModule
        The IRModule of relax.
    graphs: list<graph>
        The translated graphs.
    weights: dict<str, tvn.nd.array>
        The weights.
    codegen_configs: dict or list<dict>
        The config for codegen.
    print_configs: dict ot list<dict>
        The config for print.
    extra_option: dict
        The extra option for sub engine.
    build_folder: MSCDirectory
        The folder for saving sources and datas.
    export_folder: MSCDirectory
        The folder for saving outputs.
    plugin: PluginManager
        The plugin manager.

    Returns
    -------
    mod: IRModule
        The translated mod with target func.
    """

    target_options = {}
    if not isinstance(codegen_configs, (list, tuple)):
        codegen_configs = [codegen_configs] * len(graphs)
    if not isinstance(print_configs, (list, tuple)):
        print_configs = [print_configs] * len(graphs)
    if not isinstance(extra_options, (list, tuple)):
        extra_options = [extra_options] * len(graphs)
    for idx, graph in enumerate(graphs):
        options = to_sub_tensorrt(
            graph,
            weights,
            codegen_configs[idx],
            print_configs[idx],
            build_folder,
            output_folder,
            plugin=plugin,
        )
        if extra_options[idx]:
            options.update(extra_options[idx])
        target_options[graph.name] = msc_utils.dump_dict(options)
    mod = tvm.transform.Sequential(
        [
            tvm.relax.transform.RunCodegen({"msc_tensorrt": target_options}),
        ]
    )(mod)
    return mod
