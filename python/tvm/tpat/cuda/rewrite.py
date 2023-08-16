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

import onnx
import onnx_graphsurgeon as gs
from .type_mapping import onnx_type_mapping


def _handle_trt_not_support_type(
    graph,
    output_model_path,
    node_name_to_plugin_name,
    onnx_original_tensor_type,
):
    count = 0
    insert_cast_nodes = False

    for node in graph.nodes:
        if node.name in node_name_to_plugin_name:
            node.op = node_name_to_plugin_name[node.name]
            for i, inp in enumerate(node.inputs):
                if inp.is_empty():
                    node.inputs.remove(inp)
                    graph.cleanup()
                    continue
                if onnx_original_tensor_type[inp.name] in onnx_type_mapping:
                    cast_node = gs.Node(
                        op="Cast",
                        name="cast_to_int32_for_" + inp.name.split(":")[0],
                        attrs={"to": 6},
                    )  # 6: INT32

                    cast_node.inputs = [inp]
                    cast_node_out = gs.Variable(cast_node.name + ":0")
                    cast_node.outputs = [cast_node_out]
                    node.inputs[i] = cast_node_out
                    graph.nodes.append(cast_node)
                    graph.cleanup()
                    insert_cast_nodes = True
            for i, oup in enumerate(node.outputs):
                if onnx_original_tensor_type[oup.name] in onnx_type_mapping:
                    dtype = onnx_type_mapping[onnx_original_tensor_type[oup.name]]
                    cast_node = gs.Node(
                        op="Cast",
                        name="cast_back_for_" + oup.name.split(":")[0],
                        attrs={"to": dtype},
                    )

                    cast_node.outputs = [oup]
                    cast_node_out = gs.Variable(cast_node.name + ":0")
                    cast_node.inputs = [cast_node_out]
                    node.outputs[i] = cast_node_out
                    graph.nodes.append(cast_node)
                    graph.cleanup()
                    insert_cast_nodes = True
            count = count + 1
    assert count == len(node_name_to_plugin_name)
    if insert_cast_nodes:
        _remove_unnecessary_cast_nodes(graph)

    try:
        onnx.save(gs.export_onnx(graph), output_model_path)
    except:
        onnx.save(gs.export_onnx(graph), output_model_path, save_as_external_data=True)


def _remove_unnecessary_cast_nodes(graph):
    graph.toposort()
    cast_nodes = [
        node
        for node in graph.nodes
        if (node.op == "Cast" and node.outputs[0] not in graph.outputs and node.o().op == "Cast")
    ]
    for node in cast_nodes:
        if (
            node.attrs["to"] == 13  # uint64
            and len(node.inputs[0].inputs) <= 1
            and len(node.outputs[0].outputs) <= 1
        ):
            node.o().inputs = node.inputs
            node.inputs.clear()
            graph.cleanup()


def _compute_tensor_type(graph, tunning_nodes):
    onnx_original_tensor_type = {}

    for tunning_node in tunning_nodes:
        for inp in tunning_node.inputs:
            if inp.is_empty():
                continue
            onnx_original_tensor_type[inp.name] = inp.dtype.name

        for oup in tunning_node.outputs:
            if oup.is_empty():
                continue
            onnx_original_tensor_type[oup.name] = oup.dtype.name

    return onnx_original_tensor_type


def rewrite(
    graph,
    tunning_nodes,
    node_name_to_plugin_name,
    output_model_path,
):
    """
    Insert cast operator for operators which inputs or outputs has bool type.
    Modify operator type in onnx model for tensorRT can run plugin.
    """

    _onnx_original_tensor_type = _compute_tensor_type(graph, tunning_nodes)

    _handle_trt_not_support_type(
        graph,
        output_model_path,
        node_name_to_plugin_name,
        _onnx_original_tensor_type,
    )
