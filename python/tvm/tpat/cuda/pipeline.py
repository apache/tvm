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

import os
from typing import Tuple

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import onnxruntime as ort
from onnx import shape_inference

from tvm.tpat.cuda.kernel import Kernel
from tvm.tpat.cuda.template import StaticBatchPluginTemplate
from tvm.tpat.cuda.template_params import PluginTemplateParams

from .rewrite import rewrite
import copy


def _extract_target_onnx_node(model, tunning_node):
    """
    Extract target node from onnx graph
    """
    graph = gs.import_onnx(model)
    tensors = graph.tensors()

    tuning_node_inputs = [
        tensors[inp.name].to_variable(dtype=inp.dtype, shape=inp.shape)
        for inp in tunning_node.inputs
        if (inp.__class__ == gs.Variable and not inp.is_empty())
    ]
    tuning_node_outputs = [
        tensors[oup.name].to_variable(dtype=oup.dtype, shape=oup.shape)
        for oup in tunning_node.outputs
    ]
    tuning_input_shapes = [(inp.name, inp.shape, inp.dtype.name) for inp in graph.inputs]

    graph.inputs = tuning_node_inputs
    graph.outputs = tuning_node_outputs
    graph.cleanup()
    submodel = gs.export_onnx(graph)

    return graph, submodel, tuning_input_shapes


def pipeline(
    onnx_file: str, node_names: list[str], enable_tunning: bool, tunning_option: object, output_onnx: str
) -> Tuple[str, list[str]]:
    """Generate plugins for specified nodes in an ONNX model.

    This function is the entry point for generating plugins for specific nodes as requested by users.

    Parameters
    ----------
    onnx_file : str
        Path to the input ONNX file.
    node_names : list[str]
        Names of the nodes to be generated as TensorRT plugins.
    enable_tunning : bool
        Flag indicating whether tunning is enabled.
    tunning_option : object
        Tunning option provided for ms.relay_integration.tune_relay, you don't need to specify mod, params and target.
    output_onnx : str
        Path to the output ONNX file where the modified model will be saved.

    Returns
    -------
    Tuple[str, List[str]]
    A tuple containing the path to the output ONNX file and a list of generated plugin paths.
    """

    # 1. load onnx
    onnx_model = onnx.load(onnx_file)
    inferred_model = shape_inference.infer_shapes(onnx_model)
    graph = gs.import_onnx(inferred_model)

    # 2. retrieve all node which need to transform to plugins
    if node_names is None or len(node_names) == 0:
        return

    node_to_be_tunned = [node for node in graph.nodes if node.name in node_names]

    assert len(node_to_be_tunned) > 0, "The number of nodes to be tunned should larger than zero"

    # 3. generate plugins for each of them
    node_name_to_plugin_name = {}
    plugin_path = []
    for node in node_to_be_tunned:
        name = node.name
        plugin_name = "tpat_{}".format(name.replace("/", "_").replace(".", "_"))

        subgraph, submodel, shapes = _extract_target_onnx_node(inferred_model, node)

        kernel = Kernel(plugin_name, submodel, shapes, enable_tunning, tunning_option)
        kernel.run()

        ## 3.1 fill in template
        params = PluginTemplateParams(kernel, submodel, subgraph, node, name)
        template = StaticBatchPluginTemplate(params)
        lib = template.fill()

        plugin_path.append(lib)

        node_name_to_plugin_name[name] = plugin_name

    # 4. generate the modified onnx
    rewrite(inferred_model, node_to_be_tunned, node_name_to_plugin_name, output_onnx)

    return output_onnx, plugin_path
