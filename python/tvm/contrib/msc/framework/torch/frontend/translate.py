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

from typing import Dict, Optional, Tuple, List
import numpy as np

import torch
import tvm
from tvm.relax.frontend.torch import from_fx

from tvm.contrib.msc.core.ir.graph import MSCGraph
from tvm.contrib.msc.core.ir.translate import from_relax
from tvm.contrib.msc.core.codegen import relay_to_relax


def from_torch(
    model: torch.nn.Module,
    input_info: List[Tuple[Tuple[int], str]],
    input_names: List[str] = None,
    via_relax: bool = True,
    trans_config: Optional[Dict[str, str]] = None,
    build_config: Optional[Dict[str, str]] = None,
    opt_config: Optional[Dict[str, str]] = None,
) -> Tuple[MSCGraph, Dict[str, tvm.nd.array]]:
    """Change torch nn.Module to MSCGraph.

    Parameters
    ----------
    model: torch.nn.Module
        The torch module.
    input_info: list
        The input info in format [(shape, dtype)].
    input_names: list<str>
        The input names.
    via_relax: bool
        Whether translate torch to relax.
    trans_config: dict
        The config for transfrorm IRModule.
    build_config: dict
        The config for build MSCGraph.
    opt_config: dict
        The config for optimize the relay before translate.

    Returns
    -------
    graph: tvm.contrib.msc.core.ir.MSCGraph
        The translated graph.
    weights: dict of <string:tvm.ndarray>
        The weights from the IRModule.
    """

    if via_relax:
        graph_model, params = torch.fx.symbolic_trace(model), None
        with torch.no_grad():
            relax_mod = from_fx(graph_model, input_info)
    else:
        datas = [np.random.rand(*i[0]).astype(i[1]) for i in input_info]
        torch_datas = [torch.from_numpy(i) for i in datas]
        with torch.no_grad():
            scripted_model = torch.jit.trace(model, tuple(torch_datas)).eval()
        if input_names:
            assert len(input_names) == len(
                input_info
            ), "input_names {} length mismatch with input_info {}".format(input_names, input_info)
            shape_list = list(zip(input_names, input_info))
        else:
            shape_list = [("input" + str(idx), i_info) for idx, i_info in enumerate(input_info)]
        relay_mod, params = tvm.relay.frontend.from_pytorch(scripted_model, shape_list)
        relax_mod = relay_to_relax(relay_mod, params, trans_config, build_config, opt_config)
    graph, weights = from_relax(relax_mod, trans_config=trans_config, build_config=build_config)
    # set alias for weights
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
            weight.set_alias(alias)
    return graph, weights
