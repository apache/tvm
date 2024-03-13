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
"""tvm.contrib.msc.framework.torch.codegen.codegen"""

from typing import Dict, Optional, Any
import torch

import tvm
from tvm.contrib.msc.core.ir import MSCGraph
from tvm.contrib.msc.core.codegen import CodeGen
from tvm.contrib.msc.core import utils as msc_utils
from tvm.contrib.msc.framework.torch import _ffi_api


def to_torch(
    graph: MSCGraph,
    weights: Optional[Dict[str, tvm.nd.array]] = None,
    codegen_config: Optional[Dict[str, str]] = None,
    print_config: Optional[Dict[str, str]] = None,
    build_folder: msc_utils.MSCDirectory = None,
    plugin: Any = None,
) -> torch.nn.Module:
    """Change MSCGraph to torch nn.Module.

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
        The folder for saving scripts and datas.
    plugin: PluginManager
        The plugin manager.

    Returns
    -------
    model: torch.nn.Module
        The torch.nn.Module.
    """

    def _save_weights(folder: msc_utils.MSCDirectory):
        if weights:
            state_dict = {}
            for name, data in weights.items():
                w_producer = graph.find_producer(name)
                if w_producer.optype == "constant" and w_producer.has_attr("scalar"):
                    continue
                w_tensor = graph.find_tensor(name)
                w_name = w_tensor.alias or name
                state_dict[w_name] = torch.from_numpy(data.asnumpy())
            torch.save(state_dict, folder.relpath(graph.name + ".pth"))

    def _bind_weights(model: torch.nn.Module, folder: msc_utils.MSCDirectory) -> torch.nn.Module:
        if weights:
            state_dict = torch.load(folder.relpath(graph.name + ".pth"))
            model.load_state_dict(state_dict)
        return model

    codegen = CodeGen(graph, _ffi_api.GetTorchSources, codegen_config, print_config, build_folder)
    model_args = [plugin] if plugin else []
    return codegen.load(model_args, pre_load=_save_weights, post_load=_bind_weights)
