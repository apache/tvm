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
"""tvm.contrib.msc.core.tools.prune.pruner"""

from typing import List, Dict, Iterable, Tuple, Any

import tvm
from tvm.contrib.msc.core.ir import MSCGraph, WeightGraph, WeightJoint, MSCTensor
from tvm.contrib.msc.core.tools.tool import ToolType, BaseTool, Strategy
from tvm.contrib.msc.core import _ffi_api
from tvm.contrib.msc.core import utils as msc_utils
from .method import PruneMethod


class BasePruner(BaseTool):
    """Base pruner for all"""

    def setup(self) -> dict:
        """Setup the tool

        Returns
        -------
        info: dict
            The setup info.
        """

        # Build weight graphs
        if "prunable_types" in self._options:
            self._prunable_types = self._options["prunable_types"]
        else:
            self._prunable_types = {
                "constant": ["const"],
                "nn.conv2d": ["weight"],
                "msc.conv2d_bias": ["weight"],
                "msc.linear": ["weight"],
                "msc.linear_bias": ["weight"],
            }

        if "relation_types" in self._options:
            self._relation_types = self._options["relation_types"]
        else:
            self._relation_types = {
                "concatenate": "multi_inputs",
                "reshape": "reshape",
                "add": "passby",
                "substract": "passby",
                "multiply": "passby",
                "divide": "passby",
            }

        return super().setup()

    def reset(
        self,
        graphs: List[MSCGraph],
        weights: List[Dict[str, tvm.nd.array]],
        cache_dir: msc_utils.MSCDirectory = None,
    ) -> Tuple[List[MSCGraph], List[Dict[str, tvm.nd.array]]]:
        """Reset the tool with graphs and weights

        Parameters
        ----------
        graphs: list<MSCgraph>
            The msc graphs.
        weights: list<dic<str, tvm.nd.array>>
            The weights
        cache_dir: MSCDirectory
            cache path for save/load info

        Returns
        -------
        graphs: list<MSCgraph>
            The msc graphs.
        weights: list<dic<str, tvm.nd.array>>
            The weights
        """

        self._unpruned_tensors = {}
        res = super().reset(graphs, weights, cache_dir)
        if self.on_debug(3):
            for idx, graph in enumerate(self._weight_graphs):
                self._logger.debug(
                    msc_utils.msg_block("PRUNER.WEIGHT_GRAPH[{}].INFO".format(idx), graph.inspect())
                )
        return res

    def load_graphs(
        self, graphs: List[MSCGraph], weights: List[Dict[str, tvm.nd.array]]
    ) -> Tuple[List[MSCGraph], List[Dict[str, tvm.nd.array]]]:
        """Load the graphs and weights

        Parameters
        ----------
        graphs: list<MSCgraph>
            The msc graphs.
        weights: list<dic<str, tvm.nd.array>>
            The weights
        as_cache: bool
            Whether the graphs and weights are loaded from cache


        Returns
        -------
        graphs: list<MSCgraph>
            The msc graphs.
        weights: list<dic<str, tvm.nd.array>>
            The weights
        """

        self._weight_graphs = [
            _ffi_api.WeightGraph(graph, self._prunable_types, self._relation_types)
            for graph in graphs
        ]
        if not self._plan:
            return graphs, weights
        return self.prune_graphs(graphs, weights)

    def load_cache(self, cache_dir: msc_utils.MSCDirectory, cache_info: dict):
        """Save runner to cache

        Parameters
        -------
        cache_dir: MSCDirectory
            cache path for save/load info
        cache_info: dict
            The cache_info
        """

        assert (
            "weight_graphs" in cache_info
        ), "weight_graphs should be given in cache_info, get " + str(cache_info)
        self._weight_graphs = [
            WeightGraph.from_json(cache_dir.relpath(f)) for f in cache_info["weight_graphs"]
        ]

    def save_cache(self, cache_dir: msc_utils.MSCDirectory) -> dict:
        """Save runner to cache

        Parameters
        -------
        cache_dir: MSCDirectory
            cache path for save/load info

        Returns
        -------
        cache_info: dict
            The cache_info.
        """

        cache_info = {"weight_graphs": [g.name + "_graph.json" for g in self._weight_graphs]}
        with cache_dir:
            for graph, f_path in zip(self._weight_graphs, cache_info["weight_graphs"]):
                with open(f_path, "w") as f_graph:
                    f_graph.write(graph.to_json())
        return cache_info

    def _parse_strategys(self, strategy_list: dict) -> Dict[str, Strategy]:
        """Parse the strategy to get valid strategy

        Parameters
        -------
        strategy_list: dict
            The given strategy

        Returns
        -------
        strategys: dict<str, Strategy>
            The parsed strategy.
        """

        def _update_stages(strategy):
            if "stages" not in strategy:
                strategy["stages"] = [msc_utils.MSCStage.PRUNE]
            return strategy

        return super()._parse_strategys([_update_stages(s) for s in strategy_list])

    def _check_tensor(self, name: str, consumer: str) -> bool:
        """Check if the tensor should be processed

        Parameters
        -------
        name: str
            The name of the tensor.
        consumer: str
            The name of the consumer.

        Returns
        -------
        vaild: bool
            Whether to process the tensor.
        """

        if not self.has_w_node(name):
            return False
        strategy = self._get_tensor_strategy(name, consumer)
        if not strategy:
            return False
        if strategy.get_config("density", 1.0) == 1.0:
            return False
        return True

    def _process_tensor(
        self, tensor: Any, name: str, consumer: str, strategys: List[Strategy]
    ) -> Any:
        """Process tensor

        Parameters
        -------
        tensor: Any
            Tensor in framework
        name: str
            The name of the tensor.
        consumer: str
            The name of the consumer.
        strategys: list<Strategy>
            The strategys for the tensor.

        Returns
        -------
        tensor: Any
            The processed tensor.
        """

        if name in self._plan:
            return tensor

        assert len(strategys) == 1, "pruner should only has 1 strategy, get " + str(strategys)
        strategy = strategys[0]

        def _get_in_indices(w_node: WeightJoint) -> List[int]:
            """Get input indices for weight node"""
            if not w_node.parents:
                return []
            if w_node.name in self._plan and "in_indices" in self._plan[w_node.name]:
                return self._plan[w_node.name]["in_indices"]
            assert all(
                p.name in self._plan for p in w_node.parents
            ), "Missing some parents in runtime config " + str(w_node)
            if len(w_node.parents) == 1:
                return self._plan[w_node.parents[0].name]["out_indices"]
            if w_node.parents[0].friends:
                return self._plan[w_node.parents[0].friends[0].name]["out_indices"]
            raise Exception("Unexpected w_node " + str(w_node))

        def _prunable(w_node: WeightJoint) -> bool:
            """Check if weight node is prunable"""
            if w_node.get_attr("prune_strategy") != "prune":
                return False
            if not w_node.children:
                return False
            childrens = list(w_node.children)
            while childrens:
                current = childrens.pop(0)
                prune_strategy = current.get_attr("prune_strategy")
                if prune_strategy == "prune":
                    return True
                childrens.extend(list(current.children))
            return False

        w_node = self.find_w_node(name)
        in_axis, out_axis = self._get_io_axes(w_node)
        if w_node.weight.dim_at(in_axis) == 1:
            in_indices = []
        else:
            in_indices = _get_in_indices(w_node)
        self._plan[w_node.name] = {"in_indices": in_indices}
        if w_node.friends and w_node != w_node.friends[0]:
            lead_name = w_node.friends[0].name
            if lead_name not in self._plan:
                self._unpruned_tensors[name] = {
                    "lead_name": lead_name,
                    "tensor": tensor,
                    "consumer": consumer,
                }
                self._plan.pop(w_node.name)
                return tensor
            self._plan[w_node.name]["out_indices"] = self._plan[lead_name]["out_indices"]
        elif _prunable(w_node):
            self._plan[w_node.name] = strategy(
                self,
                self.get_data(w_node.name),
                w_node.name,
                consumer,
                in_axis=in_axis,
                out_axis=out_axis,
                in_indices=in_indices,
            )
        elif w_node.get_attr("prune_strategy") == "follow":
            self._plan[w_node.name]["out_indices"] = []
        elif w_node.get_attr("prune_strategy") == "passby":
            self._plan[w_node.name]["out_indices"] = in_indices
        else:
            self._plan[w_node.name]["out_indices"] = []
        lazy_pruned = set()
        for lazy_name, info in self._unpruned_tensors.items():
            if info["lead_name"] in self._plan:
                strategys = self._get_tensor_strategys(lazy_name, info["consumer"])
                lazy_tensor = self._process_tensor(
                    info["tensor"], lazy_name, info["consumer"], strategys
                )
                strategy_mark = ".".join([s.get_executor().name for s in strategys])
                self.debug_tensor(
                    lazy_tensor, lazy_name, consumer, "lazy processed({})".format(strategy_mark)
                )
                lazy_pruned.add(lazy_name)
        if lazy_pruned:
            self._unpruned_tensors = {
                k: v for k, v in self._unpruned_tensors.items() if k not in lazy_pruned
            }
        return tensor

    def prune_graphs(
        self, graphs: List[MSCGraph], weights: List[Dict[str, tvm.nd.array]]
    ) -> Tuple[List[MSCGraph], List[Dict[str, tvm.nd.array]]]:
        """Reset the tool

        Parameters
        ----------
        graphs: list<MSCgraph>
            The msc graphs.
        weights: list<dic<str, tvm.nd.array>>
            The weights

        Returns
        -------
        graphs: list<MSCgraph>
            The msc graphs.
        weights: list<dic<str, tvm.nd.array>>
            The weights
        """

        def _prune_by_shape(tensor: MSCTensor, shape: List[int]):
            return MSCTensor(tensor.name, tensor.dtype, tensor.layout.name, shape, tensor.alias)

        def _prune_by_channel(tensor: MSCTensor, dim, channel_axis: int = None):
            shape = tensor.get_shape()
            if channel_axis is None:
                channel_axis = tensor.layout_of("C")
            shape[channel_axis] = dim
            return _prune_by_shape(tensor, shape)

        new_graphs, new_weights = [], []
        pruned_weights_cnt = 0
        for graph, sub_weights in zip(graphs, weights):
            pruned_tensors, pruned_weights = {}, {}
            for node in graph.get_nodes():
                for weight in node.get_weights().values():
                    w_name = weight.name
                    if w_name in self._plan:
                        data = msc_utils.cast_array(sub_weights[w_name])
                        in_axis, out_axis = self._get_io_axes(self.find_w_node(w_name))
                        w_config = self._plan[w_name]
                        if w_config["in_indices"]:
                            data = PruneMethod.prune_axis(data, in_axis, w_config["in_indices"])
                        if w_config["out_indices"]:
                            data = PruneMethod.prune_axis(data, out_axis, w_config["out_indices"])
                        pruned_tensors[w_name] = _prune_by_shape(weight, data.shape)
                        pruned_weights[w_name] = tvm.nd.array(data)
                        pruned_weights_cnt += 1
                    else:
                        pruned_weights[w_name] = sub_weights[w_name]
                if node.optype == "constant" and node.weight_at("const").name in pruned_tensors:
                    ref_tensor = pruned_tensors[node.weight_at("const").name]
                    pruned_tensors[node.output_at(0).name] = MSCTensor(
                        node.output_at(0).name,
                        ref_tensor.dtype,
                        ref_tensor.layout.name,
                        ref_tensor.get_shape(),
                        ref_tensor.alias,
                    )
                elif (
                    node.optype in ("nn.conv2d", "msc.conv2d_bias", "msc.linear", "msc.linear_bias")
                    and node.weight_at("weight").name in pruned_tensors
                ):
                    out = node.output_at(0)
                    if node.optype in ("msc.linear", "msc.linear_bias"):
                        channel_axis = out.ndim - 1
                    else:
                        channel_axis = out.layout_of("C")
                    pruned_tensors[out.name] = _prune_by_channel(
                        out,
                        pruned_tensors[node.weight_at("weight").name].dim_at("O"),
                        channel_axis,
                    )
                else:
                    for out in node.get_outputs():
                        if out.name in self._plan:
                            pruned_tensors[out.name] = _prune_by_channel(
                                out, len(self._plan[out.name]["out_indices"])
                            )
                        elif (
                            node.get_inputs()
                            and node.input_at(0).name in pruned_tensors
                            and node.input_at(0).layout_of("C") >= 0
                            and out.layout_of("C") >= 0
                        ):
                            pruned_tensors[out.name] = _prune_by_channel(
                                out, pruned_tensors[node.input_at(0).name].dim_at("C")
                            )
            if self.on_debug(3):
                self._logger.debug(msc_utils.msg_block("Pruned Tensors", pruned_tensors))
            pruned_graph = _ffi_api.PruneWeights(graph, pruned_tensors)
            new_graphs.append(pruned_graph)
            new_weights.append(pruned_weights)

        # log compress rate
        def _flatten_size(weights):
            weight_size = 0
            for sub_weights in weights:
                for w_data in sub_weights.values():
                    weight_size += w_data.asnumpy().size
            return weight_size

        raw_size = _flatten_size(weights)
        new_size = _flatten_size(new_weights)
        self._logger.info(
            "{} weights pruned, compress to {:g}%".format(
                pruned_weights_cnt, new_size * 100 / raw_size
            )
        )
        return new_graphs, new_weights

    def visualize(self, visual_dir: msc_utils.MSCDirectory):
        """Visualize MSCGraphs

        Parameters
        -------
        visual_dir: MSCDirectory
            Visualize path for saving graph
        """

        for w_graph in self._weight_graphs:
            w_graph.visualize(visual_dir.relpath(w_graph.name + ".prototxt"))

    def finalize(self) -> dict:
        """Get the plan"""

        assert not self._unpruned_tensors, "Some tensors are not pruned " + str(
            self._unpruned_tensors
        )
        self._plan = {n: c for n, c in self._plan.items() if c["in_indices"] or c["out_indices"]}
        return super().finalize()

    def get_w_nodes(self) -> Iterable[WeightJoint]:
        """Get all the weight nodes in the weight_graphs.

        Returns
        -------
        nodes: generator<WeightJoint>
            The generator of weight nodes.
        """

        for g in self._weight_graphs:
            for n in g.get_nodes():
                yield n

    def has_w_node(self, name: str) -> bool:
        """Check if name in weight_graphs.

        Parameters
        ----------
        name: string
            The name of the node.

        Returns
        -------
        has_node: bool
            Whether node in weight_graphs.
        """

        for g in self._weight_graphs:
            if g.has_node(name):
                return True
        return False

    def find_w_node(self, name: str) -> WeightJoint:
        """Find weight node by name.

        Parameters
        ----------
        name: string
            The name of the node.

        Returns
        -------
        node: WeightJoint
            The found node.
        """

        for g in self._weight_graphs:
            if g.has_node(name):
                return g.find_node(name)
        raise Exception("Can not find node {} from graphs".format(name))

    def _get_io_axes(self, w_node: WeightJoint) -> Tuple[int, int]:
        """Get the input output axes

        Parameters
        ----------
        w_node: WeightJoint
            The weight node.

        Returns
        -------
        axes: (int, int)
            The input output axis.
        """

        if w_node.weight.ndim == 1:
            return 0, 0
        if w_node.has_attr("in_axis") and w_node.has_attr("out_axis"):
            return int(w_node.get_attr("in_axis")), int(w_node.get_attr("out_axis"))
        in_axis, out_axis = w_node.weight.layout_of("I"), w_node.weight.layout_of("O")
        if in_axis >= 0 and out_axis >= 0:
            return in_axis, out_axis
        if w_node.weight.layout_of("C") >= 0:
            return w_node.weight.layout_of("C"), w_node.weight.layout_of("C")
        raise Exception("Can not infer in_axis/out_axis from " + str(w_node))

    @classmethod
    def tool_type(cls):
        return ToolType.PRUNER


class DefaultPruner(BasePruner):
    @classmethod
    def tool_style(cls):
        return "default"


msc_utils.register_tool_cls(DefaultPruner)
