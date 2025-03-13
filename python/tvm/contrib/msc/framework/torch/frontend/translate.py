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
"""tvm.contrib.msc.framework.torch.frontend.translate"""

from typing import Dict, Optional, Tuple, List, Union

import torch
import tvm
from tvm.relax.frontend.torch import from_fx
from tvm.contrib.msc.core.ir.graph import MSCGraph
from tvm.contrib.msc.core.frontend import from_relax, normalize_inputs


def set_weight_alias(graph: MSCGraph) -> MSCGraph:
    """Set weight with alias in MSCGraph.

    Parameters
    ----------
    graph: MSCGraph
        The graph.


    Returns
    -------
    graph: MSCGraph
        The graph with weight alias.
    """

    for node in graph.get_nodes():
        for ref, weight in node.get_weights().items():
            if node.optype == "constant":
                alias = node.name.replace(".", "_")
            elif node.optype in ("nn.batch_norm", "nn.layer_norm", "nn.group_norm"):
                if ref == "gamma":
                    alias = node.name.replace(".", "_") + ".weight"
                elif ref == "beta":
                    alias = node.name.replace(".", "_") + ".bias"
                elif ref == "mean":
                    alias = node.name.replace(".", "_") + ".running_mean"
                elif ref == "var":
                    alias = node.name.replace(".", "_") + ".running_var"
            else:
                alias = node.name.replace(".", "_") + "." + ref
            graph.set_tensor_alias(weight, alias)
    return graph


def from_torch(
    model: torch.nn.Module,
    input_info: List[Tuple[Tuple[int], str]],
    trans_config: Optional[Dict[str, str]] = None,
    build_config: Optional[Dict[str, str]] = None,
    as_msc: bool = True,
    custom_convert_map: dict = None,
) -> Tuple[Union[MSCGraph, tvm.IRModule], Dict[str, tvm.nd.array]]:
    """Change torch nn.Module to MSCGraph.

    Parameters
    ----------
    model: torch.nn.Module
        The torch module.
    input_info: list
        The input info in format [(shape, dtype)].
    input_names: list<str>
        The input names.
    trans_config: dict
        The config for transform IRModule.
    build_config: dict
        The config for build MSCGraph.
    opt_config: dict
        The config for optimize before translate.
    as_msc: bool
        Set to to return msc graph, otherwise relax mod
    custom_convert_map: dict
        The convert map for plugin
    build_folder: MSCDirectory
        The folder for saving scripts and datas.

    Returns
    -------
    graph/mod: tvm.contrib.msc.core.ir.MSCGraph/tvm.IRModule
        The translated graph/IRModule.
    weights: dict of <string:tvm.ndarray>
        The weights from the IRModule.
    """

    graph_model = torch.fx.symbolic_trace(model)
    input_info, params = normalize_inputs(input_info), None
    with torch.no_grad():
        relax_mod = from_fx(graph_model, input_info, custom_convert_map=custom_convert_map)
    if not as_msc:
        return relax_mod, params
    graph, weights = from_relax(relax_mod, trans_config=trans_config, build_config=build_config)
    return set_weight_alias(graph), weights
