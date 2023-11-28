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
from typing import Dict, Optional, Tuple, List
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
    weights: Optional[Dict[str, tvm.nd.array]] = None,
    codegen_config: Optional[Dict[str, str]] = None,
    print_config: Optional[Dict[str, str]] = None,
    build_folder: msc_utils.MSCDirectory = None,
    output_folder: msc_utils.MSCDirectory = None,
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

    def _create_depends(folder: msc_utils.MSCDirectory) -> str:
        if weights:
            # fill fake weights
            runtime_weights = weights
            for node in graph.get_nodes():
                if node.optype in ("nn.conv2d", "msc.linear"):
                    weight = node.weight_at("weight")
                    bias = np.zeros([weight.dim_at("O")], dtype=weight.dtype_name)
                    runtime_weights[node.name + ".bias"] = bias
            # write weights file
            with open(folder.relpath(graph.name + ".wts"), "w") as f:
                f.write("{}\n".format(len(runtime_weights)))
                for name, data in runtime_weights.items():
                    if isinstance(data, np.ndarray):
                        write_weight(name, data, f)
                    else:
                        write_weight(name, data.asnumpy(), f)
        # save utils sources
        with folder.create_dir("utils") as utils_folder:
            for name, source in get_trt_sources().items():
                utils_folder.add_file(name, source)

    def _build_engine(engine_name: str, folder: msc_utils.MSCDirectory) -> str:
        with open("engine.log", "w") as log_f:
            process = subprocess.Popen("./" + engine_name, stdout=log_f, stderr=log_f, shell=True)
        process.wait()
        assert (
            process.returncode == 0
        ), "Failed to test engine {} under {}, check engine.log for detail".format(
            engine_name, os.getcwd()
        )
        return folder.move_file(engine_name + ".trt", output_folder.create_dir(graph.name))

    codegen = CodeGen(
        graph,
        _ffi_api.GetTensorRTSources,
        codegen_config,
        print_config,
        build_folder.create_dir(graph.name),
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
    graph_infos: List[Tuple[str, MSCGraph, Dict[str, tvm.nd.array]]],
    codegen_config: Optional[Dict[str, str]] = None,
    print_config: Optional[Dict[str, str]] = None,
    extra_option: Optional[Dict[str, str]] = None,
    build_folder: msc_utils.MSCDirectory = None,
    output_folder: msc_utils.MSCDirectory = None,
) -> Dict[str, str]:
    """Change all MSCGraphs to TensorRT engine files.

    Parameters
    ----------
    mod: IRModule
        The IRModule of relax.
    graph_infos: list<name, graph, name>
        The translated graph.
    codegen_config: dict
        The config for codegen.
    print_config: dict
        The config for print.
    extra_option: dict
        The extra option for sub engine.
    build_folder: MSCDirectory
        The folder for saving sources and datas.
    export_folder: MSCDirectory
        The folder for saving outputs.

    Returns
    -------
    mod: IRModule
        The translated mod with target func.
    """

    target_options = {}
    for graph, weights in graph_infos:
        options = to_sub_tensorrt(
            graph, weights, codegen_config, print_config, build_folder, output_folder
        )
        if extra_option:
            options.update(extra_option)
        target_options[graph.name] = msc_utils.dump_dict(options)
    mod = tvm.transform.Sequential(
        [
            tvm.relax.transform.RunCodegen({"msc_tensorrt": target_options}),
        ]
    )(mod)
    return mod
