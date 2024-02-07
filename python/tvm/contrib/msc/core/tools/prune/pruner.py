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

from typing import List, Dict, Tuple, Any
import numpy as np

import tvm
from tvm.contrib.msc.core.ir import MSCGraph, WeightJoint, MSCTensor
from tvm.contrib.msc.core.tools.tool import ToolType, WeightTool, ToolStrategy
from tvm.contrib.msc.core import _ffi_api
from tvm.contrib.msc.core import utils as msc_utils
from .method import PruneMethod


class BasePruner(WeightTool):
    """Base pruner for all"""

    def _get_wtypes(self) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
        """Get the weight types from options

        Returns
        -------
        main_wtypes: dict<str,list<str>>
            The main weight types.
        relation_wtypes: dict<str, str>
            The relation weight types
        """

        if "main_wtypes" in self._options:
            main_wtypes = self._options["main_wtypes"]
        else:
            main_wtypes = {
                "constant": ["const"],
                "nn.conv2d": ["weight"],
                "msc.conv2d_bias": ["weight"],
                "msc.linear": ["weight"],
                "msc.linear_bias": ["weight"],
            }

        if "relation_wtypes" in self._options:
            relation_wtypes = self._options["relation_wtypes"]
        else:
            relation_wtypes = {
                "concatenate": "multi_inputs",
                "reshape": "reshape",
                "add": "passby",
                "substract": "passby",
                "multiply": "passby",
                "divide": "passby",
            }
        return main_wtypes, relation_wtypes

    def _parse_strategys(self, strategy_list: dict) -> Dict[str, ToolStrategy]:
        """Parse the strategy to get valid strategy

        Parameters
        -------
        strategy_list: dict
            The given strategy

        Returns
        -------
        strategys: dict<str, ToolStrategy>
            The parsed strategy.
        """

        def _update_stages(strategy):
            if "stages" not in strategy:
                strategy["stages"] = [msc_utils.MSCStage.PRUNE]
            strategy["tensor_types"] = ["weight", "output"]
            return strategy

        return super()._parse_strategys([_update_stages(s) for s in strategy_list])

    def _reset(
        self, graphs: List[MSCGraph], weights: Dict[str, tvm.nd.array]
    ) -> Tuple[List[MSCGraph], Dict[str, tvm.nd.array]]:
        """Reset the tool

        Parameters
        ----------
        graphs: list<MSCgraph>
            The msc graphs.
        weights: dict<str, tvm.nd.array>
            The weights.

        Returns
        -------
        graphs: list<MSCgraph>
            The msc graphs.
        weights: dict<str, tvm.nd.array>
            The weights.
        """

        self._meta_weights = weights
        graphs, weights = super()._reset(graphs, weights)
        if self._plan and self._enabled:
            return self.prune_graphs(graphs, weights)
        return graphs, weights

    def _execute_before_build(self, *args, **kwargs):
        """Execute before model build

        Parameters
        ----------
        args: list<Any>
            The arguments for model build.
        kwargs: dict<Any>
            The key word arguments for model build.
        """

        self._unpruned_tensors = {}
        super()._execute_before_build(*args, **kwargs)

    def _execute_after_build(self, output: Any) -> Any:
        """Execute after model build

        Parameters
        ----------
        output: Any
            The output reference of the model.

        Returns
        -------
        output: Any
           The modified output reference.
        """

        assert not self._unpruned_tensors, "Some tensors are not pruned " + str(
            self._unpruned_tensors
        )
        return super()._execute_after_build(output)

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
        return True

    def _process_tensor(
        self, tensor: Any, name: str, consumer: str, scope: str, strategys: List[ToolStrategy]
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
        scope: str
            The scope mark teacher| student| null.
        strategys: list<ToolStrategy>
            The strategys for the tensor.

        Returns
        -------
        tensor: Any
            The processed tensor.
        """

        if name in self._plan:
            return tensor

        self._prune_tensor(name, consumer, strategys)
        lazy_pruned = set()
        for lazy_name, info in self._unpruned_tensors.items():
            if info["lead_name"] in self._plan:
                strategys = self._get_tensor_strategys(lazy_name, info["consumer"])
                self._prune_tensor(lazy_name, info["consumer"], strategys)
                t_mark = ".".join([s.get_executor().name for s in strategys])
                self.debug_tensor(
                    self.find_tensor(lazy_name),
                    lazy_name,
                    consumer,
                    "lazy processed({})".format(t_mark),
                )
                lazy_pruned.add(lazy_name)
        if lazy_pruned:
            self._unpruned_tensors = {
                k: v for k, v in self._unpruned_tensors.items() if k not in lazy_pruned
            }
        return tensor

    def _prune_tensor(self, name: str, consumer: str, strategys: List[ToolStrategy]) -> Any:
        """Prune tensor

        Parameters
        -------
        name: str
            The name of the tensor.
        consumer: str
            The name of the consumer.
        scope: str
            The scope mark teacher| student| null.
        strategys: list<ToolStrategy>
            The strategys for the tensor.
        """

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
            if strategy.get_config().get("density", 1) == 1:
                return False
            if w_node.get_attr("weight_strategy") != "main":
                return False
            if not w_node.children:
                return False
            childrens = list(w_node.children)
            while childrens:
                current = childrens.pop(0)
                weight_strategy = current.get_attr("weight_strategy")
                if weight_strategy == "main":
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
                    "consumer": consumer,
                }
                self._plan.pop(w_node.name)
                return None
            self._plan[w_node.name]["out_indices"] = self._plan[lead_name]["out_indices"]
        elif _prunable(w_node):
            self._plan[w_node.name] = strategy(
                self,
                self.get_meta_data(w_node.name),
                w_node.name,
                consumer,
                in_axis=in_axis,
                out_axis=out_axis,
                in_indices=in_indices,
            )
        elif w_node.get_attr("weight_strategy") == "follow":
            self._plan[w_node.name]["out_indices"] = []
        elif w_node.get_attr("weight_strategy") == "passby":
            self._plan[w_node.name]["out_indices"] = in_indices
        else:
            self._plan[w_node.name]["out_indices"] = []

    def prune_graphs(
        self, graphs: List[MSCGraph], weights: Dict[str, tvm.nd.array]
    ) -> Tuple[List[MSCGraph], Dict[str, tvm.nd.array]]:
        """Reset the tool

        Parameters
        ----------
        graphs: list<MSCgraph>
            The msc graphs.
        weights: dict<str, tvm.nd.array>
            The weights.

        Returns
        -------
        graphs: list<MSCgraph>
            The msc graphs.
        weights: dict<str, tvm.nd.array>
            The weights.
        """

        def _prune_by_shape(tensor: MSCTensor, shape: List[int]):
            return MSCTensor(tensor.name, tensor.dtype, tensor.layout.name, shape, tensor.alias)

        def _prune_by_channel(tensor: MSCTensor, dim, channel_axis: int = None):
            shape = tensor.get_shape()
            if channel_axis is None:
                channel_axis = tensor.layout_of("C")
            shape[channel_axis] = dim
            return _prune_by_shape(tensor, shape)

        pruned_graphs, pruned_weights = [], {}
        pruned_cnt = 0
        for graph in graphs:
            pruned_tensors = {}
            for node in graph.get_nodes():
                for weight in node.get_weights().values():
                    w_node, w_name = self.find_w_node(weight.name), weight.name
                    if w_name not in self._plan:
                        pruned_weights[w_name] = weights[w_name]
                    elif w_node.get_attr("pruned_shape", "") != "":
                        pruned_weights[w_name] = weights[w_name]
                        pruned_shape = [int(i) for i in w_node.get_attr("pruned_shape").split(",")]
                        assert pruned_shape == list(
                            pruned_weights[w_name].shape
                        ), "pruned_shape {} mismatch with data shape {}".format(
                            pruned_shape, pruned_weights[w_name].shape
                        )
                    else:
                        data = msc_utils.cast_array(weights[w_name])
                        in_axis, out_axis = self._get_io_axes(self.find_w_node(w_name))
                        w_config = self._plan[w_name]
                        if w_config["in_indices"]:
                            data = PruneMethod.prune_axis(data, in_axis, w_config["in_indices"])
                        if w_config["out_indices"]:
                            data = PruneMethod.prune_axis(data, out_axis, w_config["out_indices"])
                        pruned_tensors[w_name] = _prune_by_shape(weight, data.shape)
                        pruned_weights[w_name] = tvm.nd.array(data)
                        w_node.set_attr(
                            "pruned_shape",
                            ",".join([str(i) for i in pruned_tensors[w_name].get_shape()]),
                        )
                        pruned_cnt += 1
                if node.optype == "constant":
                    if node.weight_at("const").name not in pruned_tensors:
                        continue
                    ref_tensor = pruned_tensors[node.weight_at("const").name]
                    pruned_tensors[node.output_at(0).name] = ref_tensor.clone(
                        name=node.output_at(0).name
                    )
                elif node.optype in self._main_wtypes:
                    if node.weight_at("weight").name not in pruned_tensors:
                        continue
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
                elif node.optype in self._relation_wtypes:
                    for out in node.get_outputs():
                        w_node = self.find_w_node(out.name)
                        if out.name not in self._plan or w_node.get_attr("pruned_shape", "") != "":
                            continue
                        pruned_tensors[out.name] = _prune_by_channel(
                            out, len(self._plan[out.name]["out_indices"])
                        )
                        w_node.set_attr(
                            "pruned_shape",
                            ",".join([str(i) for i in pruned_tensors[out.name].get_shape()]),
                        )
                elif node.get_inputs():
                    ref_input = node.input_at(0)
                    if ref_input.name not in pruned_tensors or ref_input.layout_of("C") < 0:
                        continue
                    for out in node.get_outputs():
                        if out.layout_of("C") < 0:
                            continue
                        pruned_tensors[out.name] = _prune_by_channel(
                            out, pruned_tensors[ref_input.name].dim_at("C")
                        )

            def _is_pruned(tensor: MSCTensor, graph: MSCGraph) -> bool:
                return tensor.get_shape() != graph.find_tensor(tensor.name).get_shape()

            pruned_tensors = {k: v for k, v in pruned_tensors.items() if _is_pruned(v, graph)}
            if self.on_debug(3, in_forward=False):
                self._logger.debug(msc_utils.msg_block("Pruned Tensors", pruned_tensors))

            if pruned_tensors:
                pruned_graph = _ffi_api.PruneWeights(graph, pruned_tensors)
                pruned_graphs.append(pruned_graph)
            else:
                pruned_graphs.append(graph)

        def _flatten_size(weights):
            weight_size = sum([w.asnumpy().size for w in weights.values()])
            return weight_size / 2**20

        raw_size = _flatten_size(weights)
        # log compress rate
        if pruned_cnt > 0:
            new_size = _flatten_size(pruned_weights)
            self._logger.info(
                "Prune %d weights, compress to %.2f%% (%.4f M->%.4f M)",
                pruned_cnt,
                new_size * 100 / raw_size,
                raw_size,
                new_size,
            )
        else:
            self._logger.info("No weights pruned, size %.4f M", raw_size)
        return pruned_graphs, pruned_weights

    def get_meta_data(self, name: str) -> np.ndarray:
        """Get meta weight as np.ndarray

        Parameters
        ----------
        name: str
            The name of data.

        Returns
        -------
        data: np.ndarray
            The data in np.ndarray format.
        """

        if name in self._meta_weights:
            return msc_utils.cast_array(self._meta_weights[name])
        raise Exception(
            "Can not find data {} from {} weights".format(name, len(self._meta_weights))
        )

    def create_tasks(self, **kwargs) -> List[dict]:
        """Create tasks for gym

        Parameters
        ----------
        kwargs: dict
           The kwargs for create tasks.

        Returns
        -------
        tasks: list<dict>
            The tasks.
        """

        tasks = []
        for w_node in self.get_w_nodes():
            if w_node.get_attr("weight_strategy") != "main":
                continue
            consumer = self.find_producer(w_node.name).name
            strategy = self._get_tensor_strategy(w_node.name, consumer)
            tasks.append(
                {
                    "tensor_names": [self.to_tensor_id(w_node.name, consumer)],
                    **strategy.meta,
                }
            )
        return tasks

    def plan_by_strategys(self, strategys: List[dict]) -> dict:
        """Plan the pruning with startegys and get plan

        Parameters
        -------
        strategys: list<dict>
            The given strategys

        Returns
        -------
        plan: dict
            The plan after new strategy applied.
        """

        self._tensor_cache, self._processed_tensor = {}, {}
        self._plan = {}
        self._strategys = self._parse_strategys(msc_utils.copy_dict(strategys))
        info = {k: v.inspect() for k, v in self._strategys.items()}
        title = "{}.PRUNE_STRATEGYS".format(self.tool_type().upper())
        self._logger.debug(msc_utils.msg_block(title, info, width=0))
        for w_node in self.get_w_nodes():
            consumer = self.find_consumers(w_node.name)[0]
            self.process_tensor(w_node.weight, w_node.name, consumer.name, "")
        self._plan = {n: c for n, c in self._plan.items() if c["in_indices"] or c["out_indices"]}
        return self._plan

    def finalize(self) -> dict:
        """Get the plan"""

        self._plan = {n: c for n, c in self._plan.items() if c["in_indices"] or c["out_indices"]}
        return super().finalize()

    @classmethod
    def tool_type(cls):
        return ToolType.PRUNER


class DefaultPruner(BasePruner):
    @classmethod
    def tool_style(cls):
        return "default"


msc_utils.register_tool_cls(DefaultPruner)
