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
# pylint: disable=unused-argument
"""tvm.contrib.msc.core.runtime.runner"""

import os
import json
import logging
from typing import Dict, Optional, Any, List, Tuple, Union
import numpy as np

import tvm
from tvm.contrib.msc.core.ir import MSCGraph
from tvm.contrib.msc.core.frontend import from_relax
from tvm.contrib.msc.core.tools import BaseTool, ToolType, create_tool
from tvm.contrib.msc.core.utils.namespace import MSCFramework
from tvm.contrib.msc.core.utils.message import MSCStage
from tvm.contrib.msc.core import utils as msc_utils
from tvm.contrib.msc.core import _ffi_api


class BaseRunner(object):
    """Basic runner of MSC

    Parameters
    ----------
    mod: IRModule
        The IRModule of relax.
    params: dict of <string:tvm.ndarray>
        The parameters of the IRModule.
    tools_config: dict
        The config of MSC Tools.
    translate_config: dict
        The config for translate IRModule to MSCGraph.
    codegen_config: dict
        The config for build MSCGraph to runnable model.
    stage: str
        The stage of runner.
    name: str
        The name of the runner
    device: str
        The device of the model, cpu| cuda| cuda:0|...
    is_training: bool
        Whether use model in training
    debug_level: int
        The debug level.
    logger: logging.Logger
        The logger
    """

    def __init__(
        self,
        mod: tvm.IRModule,
        tools_config: Optional[Dict[str, Any]] = None,
        translate_config: Optional[Dict[str, str]] = None,
        generate_config: Optional[Dict[str, str]] = None,
        stage: str = "default",
        name: str = "main",
        device: str = "cpu",
        is_training: bool = False,
        debug_level: int = 0,
        logger: logging.Logger = None,
    ):
        self._mod = mod
        self._tools_config = tools_config or {}
        self._translate_config = translate_config or {}
        self._generate_config = generate_config or {}
        self._stage = stage
        self._name = name
        self._device = device if self._device_enabled(device) else "cpu"
        self._is_training = is_training
        self._debug_level = debug_level
        self._logger = logger or msc_utils.get_global_logger()
        self._logger.info(
            msc_utils.msg_block(
                "RUNNER.SETUP({} @ {})".format(self._stage, self.framework), self.setup()
            )
        )

    def setup(self) -> dict:
        """Setup the runner

        Returns
        -------
        info: dict
            The setup info.
        """

        if "build_folder" not in self._generate_config:
            self._generate_config["build_folder"] = msc_utils.get_build_dir()
        if self._tools_config:
            if "codegen" not in self._generate_config:
                self._generate_config["codegen"] = {}
            self._generate_config["codegen"].update({"use_tools": True, "tools_tag": self._name})
        self._graphs, self._weights = [], []
        self._model, self._model_info = None, {}
        self._runnable = None
        self._tools = {}
        return {
            "tools_config": self._tools_config,
            "translate_config": self._translate_config,
            "generate_config": self._generate_config,
            "name": self._name,
            "device": self._device,
            "is_training": self._is_training,
            "debug_level": self._debug_level,
        }

    def change_stage(self, stage: str):
        """Change the stage of tools and strategy"""

        self._stage = stage
        for tool in self._tools.values():
            tool.change_stage(stage)

    def build(self, cache_dir: msc_utils.MSCDirectory = None, build_graph: bool = False) -> Any:
        """Build the runnable object

        Parameters
        -------
        cache_dir: MSCDirectory
            cache path for save/load info
        build_graph: bool
            Whether to build the MSCGraphs.

        Returns
        -------
        runnable: Any
           The runnable object.
        """

        if cache_dir and os.path.isfile(cache_dir.relpath("cache_info.json")):
            cache_info = msc_utils.load_dict(cache_dir.relpath("cache_info.json"))
        else:
            cache_info = {}

        # Create tools
        if self._tools_config:
            for t_type, config in self._tools_config.items():
                self._tools[t_type] = create_tool(
                    self.framework, t_type, self._name, stage=self._stage, **config
                )

        # Load graphs from cache
        if cache_info.get("graphs"):
            self._graphs, self._weights = self._load_graphs(cache_dir, cache_info["graphs"])
            self._logger.debug(
                "Load {} graphs from cache @ {}".format(len(self._graphs), cache_dir)
            )

        # Get or rebuild graphs
        if build_graph or not self._graphs:
            self._graphs, self._weights = self._translate()
            self._logger.debug("Translate {} graphs from module".format(len(self._graphs)))

        # reset graph for tools
        for tool in self._tools.values():
            self._graphs, self._weights = tool.reset(
                self._graphs, self._weights, cache_dir=cache_dir
            )

        if cache_info.get("model") and not build_graph:
            # Load model from cache
            self._model = self._load_model(cache_dir, cache_info["model"])
        else:
            # Generate model
            self._model = self._generate_model()
            if "loader" in self._generate_config:
                loader, generate_config = self._generate_config["loader"]
                self._model = loader(self._model, **generate_config)
                self._logger.info(
                    "Model({}) processed by customize loader {}({})".format(
                        self.framework, loader, generate_config
                    )
                )
        self._model_info = self._inspect_model()
        if self._debug_level >= 3:
            self._logger.debug(msc_utils.msg_block("RUNNER.MODEL_INFO", self._model_info))

        if cache_info.get("runnable") and not build_graph:
            # Load runnable from cache
            self._runnable = self._load_runnable(cache_dir, cache_info["runnable"])
        else:
            # Build runnable on device
            self._runnable = self._to_runnable(self._model, self._device, self._is_training)
        self._logger.info(
            "Runnable({}, {}) loaded on device {}".format(
                self.framework, "train" if self._is_training else "eval", self._device
            )
        )
        return self._runnable

    def save_cache(self, cache_dir: msc_utils.MSCDirectory):
        """Save runner to cache

        Parameters
        -------
        cache_dir: MSCDirectory
            cache path for save/load info
        """

        cache_info = {
            "graphs": self._save_graphs(cache_dir),
            "model": self._save_model(cache_dir),
            "runnable": self._save_runnable(cache_dir),
        }
        for tool in self._tools.values():
            cache_info.update(tool.save_cache(cache_dir))
        with open(cache_dir.relpath("cache_info.json"), "w") as f:
            f.write(json.dumps(cache_info, indent=2))
        if self._debug_level >= 3:
            self._logger.debug(
                msc_utils.msg_block("RUNNER.CACHE_INFO", {"folder": cache_dir, "info": cache_info})
            )

    def run(
        self, inputs: Union[List[np.ndarray], Dict[str, np.ndarray]], ret_type="dict"
    ) -> Union[List[np.ndarray], Dict[str, np.ndarray]]:
        """Run the model to get outputs

        Parameters
        -------
        inputs: list<data> or dict<str, data>
            The inputs in list or dict.
        ret_type: str
            The return type list| dict

        Returns
        -------
        outputs: dict<str, data>
            The outputs in dict.
        """

        model_inputs = self.get_inputs()
        model_outputs = self.get_outputs()
        if isinstance(inputs, (list, tuple)):
            assert len(inputs) == len(
                model_inputs
            ), "inputs({}) mismatch with model inputs {}".format(len(inputs), model_inputs)
            inputs = {info["name"]: data for info, data in zip(model_inputs, inputs)}
        assert isinstance(inputs, dict), "Expect inputs as list or dict, get {}({})".format(
            inputs, type(inputs)
        )
        assert all(
            isinstance(data, np.ndarray) for data in inputs.values()
        ), "Expected all inputs as np.ndarray"
        inputs = {i["name"]: inputs[i["name"]] for i in model_inputs}
        outputs = self._call_runnable(self._runnable, inputs, self._device)
        if ret_type == "dict":
            if isinstance(outputs, (list, tuple)):
                assert len(outputs) == len(
                    model_outputs
                ), "outputs({}) mismatch with model outputs {}".format(len(outputs), model_outputs)
                outputs = {info["name"]: data for info, data in zip(model_outputs, outputs)}
            if not isinstance(outputs, dict):
                assert len(model_outputs) == 1, "Expect model_outputs with len 1, get " + str(
                    model_outputs
                )
                outputs = {model_outputs[0]["name"]: outputs}
            outputs = {name: msc_utils.cast_array(data) for name, data in outputs.items()}
        elif ret_type == "list":
            if isinstance(outputs, dict):
                assert len(outputs) == len(
                    model_outputs
                ), "outputs({}) mismatch with model outputs {}".format(len(outputs), model_outputs)
                outputs = [outputs[o["name"]] for o in model_outputs]
            if not isinstance(outputs, (list, tuple)):
                outputs = [outputs]
            outputs = [msc_utils.cast_array(data) for data in outputs]
        return outputs

    def get_tool(self, tool_type: str) -> BaseTool:
        """Get tool by type

        Parameters
        -------
        tool_type: str
            The type of the tool prune| quantize| distill...

        Returns
        -------
        tool: BaseTool
            The saved tool.
        """

        return self._tools.get(tool_type)

    def apply_tool(self, tool_type: str, data_loader: Any = None) -> dict:
        """Execute tool and get plan

        Parameters
        -------
        tool_type: str
            The tool type, should be in ToolType
        data_loader:
            The data loader
        """

        assert tool_type in self._tools, "Can not find tool " + str(tool_type)
        if tool_type == ToolType.PRUNER:
            pruner = self.get_tool(ToolType.PRUNER)
            if not pruner.finalize():
                assert data_loader, "data_loader should be given to plan prune"
                for inputs in data_loader():
                    self.run(inputs)
                    break
            plan = pruner.finalize()
        else:
            plan = self.get_tool(tool_type).finalize()
        assert plan, "Failed to create plan for {}".format(tool_type)
        plan_file = self._tools_config[tool_type]["plan_file"]
        with open(plan_file, "w") as f:
            f.write(json.dumps(plan, indent=2))
        self._logger.info("Save %s plan -> %s", tool_type, plan_file)
        return plan

    def visualize(self, visual_dir: msc_utils.MSCDirectory):
        """Visualize MSCGraphs

        Parameters
        -------
        visual_dir: MSCDirectory
            Visualize path for saving graph
        """

        for graph in self._graphs:
            graph.visualize(visual_dir.relpath(graph.name + ".prototxt"))
        for tool in self._tools.values():
            tool.visualize(visual_dir)

    def get_inputs(self) -> List[Dict[str, str]]:
        """Get the inputs of the model

        Returns
        -------
        inputs: list<tensor_des>
            The inputs info.
        """

        return self._model_info["inputs"]

    def get_outputs(self) -> List[Dict[str, str]]:
        """Get the outputs of the model

        Returns
        -------
        outputs: list<tensor_des>
            The outputs info.
        """

        return self._model_info["outputs"]

    def destory(self):
        """Destory runner"""

        if self._model:
            self._model = None
        if self._runnable:
            self._runnable = None
        for tool in self._tools.values():
            tool.destory()

    def _translate(self) -> Tuple[List[MSCGraph], Dict[str, tvm.nd.array]]:
        """Translate IRModule to MSCgraphs

        Returns
        -------
        graph_list: list<MSCGraph>
            The translated graphs
        weights_list: list<dict<str, tvm.nd.array>>
            The translated weights
        """

        raise NotImplementedError("_translate is not implemented for " + str(self.__class__))

    def _load_graphs(
        self, cache_dir: msc_utils.MSCDirectory, cache_info: dict
    ) -> Tuple[List[MSCGraph], Dict[str, tvm.nd.array]]:
        """Load MSCgraphs from cache

        Parameters
        -------
        cache_dir: MSCDirectory
            cache path for save/load info
        cache_info: dict
            The cache info.

        Returns
        -------
        graph_list: list<MSCGraph>
            The translated graphs
        weights_list: list<dict<str, tvm.nd.array>>
            The translated weights
        """

        raise NotImplementedError("_load_graphs is not implemented for " + str(self.__class__))

    def _save_graphs(self, cache_dir: msc_utils.MSCDirectory) -> dict:
        """Save MSCgraphs to cache

        Parameters
        -------
        cache_dir: MSCDirectory
            cache path for save/load info

        Returns
        -------
        cache_info: dict
            The cache info.
        """

        raise NotImplementedError("_save_graphs is not implemented for " + str(self.__class__))

    def _generate_model(
        self, graphs: List[MSCGraph] = None, weights: List[Dict[str, tvm.nd.array]] = None
    ) -> Any:
        """Codegen the model according to framework

        Parameters
        -------
        graphs: list<MSCgraph>
            The msc graphs.
        weights: list<dic<str, tvm.nd.array>>
            The weights

        Returns
        -------
        model: Any
            The meta model
        """

        raise NotImplementedError("_load is not implemented for " + str(self.__class__))

    def _load_model(self, cache_dir: msc_utils.MSCDirectory, cache_info: dict) -> Any:
        """Load the model from cache

        Parameters
        -------
        cache_dir: MSCDirectory
            cache path for save/load info
        cache_info: dict
            The cache info.

        Returns
        -------
        model: Any
            The meta model
        """

        raise NotImplementedError("_load_model is not implemented for " + str(self.__class__))

    def _save_model(self, cache_dir: msc_utils.MSCDirectory) -> dict:
        """Save model to cache

        Parameters
        -------
        cache_dir: MSCDirectory
            cache path for save/load info

        Returns
        -------
        cache_info: dict
            The cache info.
        """

        # disable save model by default
        return {}

    def _to_runnable(self, model: Any, device: str, is_training: bool) -> Any:
        """Build runnable object

        Parameters
        -------
        model: Any
            The meta model.
        device: str
            The device for place model
        is_training: bool
            Whether to load model for training

        Returns
        -------
        runnable: Any
            The runnable
        """

        raise NotImplementedError("_to_runnable is not implemented for " + str(self.__class__))

    def _load_runnable(self, cache_dir: msc_utils.MSCDirectory, cache_info: dict) -> Any:
        """Load the runnable from cache

        Parameters
        -------
        cache_dir: MSCDirectory
            cache path for save/load info
        cache_info: dict
            The cache info.

        Returns
        -------
        runnable: Any
            The runnable
        """

        raise NotImplementedError("_load_runnable is not implemented for " + str(self.__class__))

    def _save_runnable(self, cache_dir: msc_utils.MSCDirectory) -> dict:
        """Save runnable to cache

        Parameters
        -------
        cache_dir: MSCDirectory
            cache path for save/load info

        Returns
        -------
        cache_info: dict
            The cache info.
        """

        # disable save runnable by default
        return {}

    def _inspect_model(self) -> dict:
        """Inspect the model

        Returns
        -------
        model_info: dict
            The inspected model info
        """

        raise NotImplementedError("_inspect_model is not implemented for " + str(self.__class__))

    def _call_runnable(
        self, runnable: Any, inputs: Dict[str, np.ndarray], device: str
    ) -> Union[List[np.ndarray], Dict[str, np.ndarray]]:
        """Call the runnable to get outputs

        Parameters
        -------
        model:
            The runnable model.
        inputs: dict<str, data>
            The inputs in dict.
        device: str
            The device.

        Returns
        -------
        outputs: list<data> or dict<str, data>
            The outputs in list or dict.
        """

        raise NotImplementedError("_call_runnable is not implemented for " + str(self.__class__))

    def _device_enabled(self, device: str) -> bool:
        """Check if the device is enabled

        Returns
        -------
        enabled: bool
            Whether the device is enabled.
        """

        return True

    @property
    def stage(self):
        return self._stage

    @property
    def model(self):
        return self._model

    @property
    def runnable(self):
        return self._runnable

    @property
    def model_info(self):
        return self._model_info

    @property
    def device(self):
        return self._device

    @property
    def codegen_func(self):
        raise NotImplementedError("codegen_func is not implemented for " + str(self.__class__))

    @property
    def framework(self):
        return MSCFramework.MSC


class ModelRunner(BaseRunner):
    """Model runner of MSC"""

    def _translate(self) -> Tuple[List[MSCGraph], Dict[str, tvm.nd.array]]:
        """Translate IRModule to MSCgraphs

        Returns
        -------
        graph_list: list<MSCGraph>
            The translated graphs
        weights_list: list<dict<str, tvm.nd.array>>
            The translated weights
        """

        graph, weights = from_relax(
            self._mod,
            trans_config=self._translate_config.get("transform"),
            build_config=self._translate_config.get("build"),
            opt_config=self._translate_config.get("optimize"),
        )
        return [graph], [weights]

    def _load_graphs(
        self, cache_dir: msc_utils.MSCDirectory, cache_info: dict
    ) -> Tuple[List[MSCGraph], Dict[str, tvm.nd.array]]:
        """Load MSCgraphs from cache

        Parameters
        -------
        cache_dir: MSCDirectory
            cache path for save/load info
        cache_info: dict
            The cache info.

        Returns
        -------
        graph_list: list<MSCGraph>
            The translated graphs
        weights_list: list<dict<str, tvm.nd.array>>
            The translated weights
        """

        assert "main" in cache_info, "main should be given in cache_info, get " + str(cache_info)
        graph = MSCGraph.from_json(cache_dir.relpath(cache_info["main"]["graph"]))
        with open(cache_dir.relpath(cache_info["main"]["weights"]), "rb") as f:
            weights = tvm.runtime.load_param_dict(f.read())
        return [graph], [weights]

    def _save_graphs(self, cache_dir: msc_utils.MSCDirectory) -> dict:
        """Save MSCgraphs to cache

        Parameters
        -------
        cache_dir: MSCDirectory
            cache path for save/load info

        Returns
        -------
        cache_info: dict
            The cache info.
        """

        main_info = {
            "graph": self._graphs[0].name + "_graph.json",
            "weights": self._graphs[0].name + "_params.bin",
        }
        with cache_dir:
            with open(main_info["graph"], "w") as f_graph:
                f_graph.write(self._graphs[0].to_json())
            with open(main_info["weights"], "wb") as f_params:
                f_params.write(tvm.runtime.save_param_dict(self._weights[0]))
        return {"main": main_info}

    def _generate_model(
        self, graphs: List[MSCGraph] = None, weights: List[Dict[str, tvm.nd.array]] = None
    ) -> Any:
        """Codegen the model according to framework

        Parameters
        -------
        graphs: list<MSCgraph>
            The msc graphs.
        weights: list<dic<str, tvm.nd.array>>
            The weights

        Returns
        -------
        model: Any
            The runnable model
        """

        return self.codegen_func(
            graphs or self._graphs[0],
            weights or self._weights[0],
            codegen_config=self._generate_config.get("codegen"),
            print_config=self._generate_config.get("print"),
            build_folder=self._generate_config["build_folder"],
        )

    def _inspect_model(self) -> dict:
        """Inspect the model

        Returns
        -------
        model_info: dict
            The inspected model info
        """

        return self._graphs[0].inspect()


class BYOCRunner(BaseRunner):
    """BYOC runner of MSC"""

    def setup(self) -> dict:
        """Setup the runner

        Returns
        -------
        info: dict
            The setup info.
        """

        self._byoc_mod, self._byoc_graph = None, None
        return super().setup()

    def visualize(self, visual_dir: msc_utils.MSCDirectory):
        """Visualize MSCGraphs

        Parameters
        -------
        visual_dir: MSCDirectory
            Visualize path for saving graph
        """

        super().visualize(visual_dir)
        self._byoc_graph.visualize(visual_dir.relpath(self._byoc_graph.name + ".prototxt"))

    def _translate(self) -> Tuple[List[MSCGraph], Dict[str, tvm.nd.array]]:
        """Translate IRModule to MSCgraphs

        Returns
        -------
        graph_list: list<MSCGraph>
            The translated graphs
        weights_list: list<dict<str, tvm.nd.array>>
            The translated weights
        """

        self._byoc_mod, graph_infos = self.partition_func(
            self._mod,
            trans_config=self._translate_config.get("transform"),
            build_config=self._translate_config.get("build"),
        )
        graphs, weights = [], []
        for graph, sub_weights in graph_infos:
            graphs.append(graph)
            weights.append(sub_weights)
        self._byoc_graph = _ffi_api.BuildFromRelax(
            self._byoc_mod, "main", msc_utils.dump_dict(self._translate_config.get("build"))
        )
        return graphs, weights

    def _load_graphs(
        self, cache_dir: msc_utils.MSCDirectory, cache_info: dict
    ) -> Tuple[List[MSCGraph], Dict[str, tvm.nd.array]]:
        """Load MSCgraphs from cache

        Parameters
        -------
        cache_dir: MSCDirectory
            cache path for save/load info
        cache_info: dict
            The cache info.

        Returns
        -------
        graph_list: list<MSCGraph>
            The translated graphs
        weights_list: list<dict<str, tvm.nd.array>>
            The translated weights
        """

        assert "byoc_mod" in cache_info, "byoc_mod should be given in cache_info, get " + str(
            cache_info
        )
        assert "byoc_graph" in cache_info, "byoc_graph should be given in cache_info, get " + str(
            cache_info
        )
        assert "sub_graphs" in cache_info, "sub_graphs should be given in cache_info, get " + str(
            cache_info
        )

        with open(cache_dir.relpath(cache_info["byoc_mod"]), "r") as f:
            self._byoc_mod = tvm.ir.load_json(f.read())
        graphs, weights = [], []
        for f_graph, f_weights in cache_info["sub_graphs"]:
            graphs.append(MSCGraph.from_json(cache_dir.relpath(f_graph)))
            with open(cache_dir.relpath(f_weights), "rb") as f:
                weights = tvm.runtime.load_param_dict(f.read())
        self._byoc_graph = MSCGraph.from_json(cache_dir.relpath(cache_info["byoc_graph"]))
        return graphs, weights

    def _save_graphs(self, cache_dir: msc_utils.MSCDirectory) -> dict:
        """Save MSCgraphs to cache

        Parameters
        -------
        cache_dir: MSCDirectory
            cache path for save/load info

        Returns
        -------
        cache_info: dict
            The cache info.
        """

        sub_graphs = [
            (graph.name + "_graph.info", graph.name + "_params.bin") for graph in self._graphs
        ]
        with cache_dir:
            for graph, weights, info in zip(self._graphs, self._weights, sub_graphs):
                with open(info[0], "w") as f_graph:
                    f_graph.write(graph.to_json())
                with open(info[1], "wb") as f_params:
                    f_params.write(tvm.runtime.save_param_dict(weights))
            with open("byoc_graph.json", "w") as f:
                f.write(self._byoc_graph.to_json())
            with open("byoc_module.json", "w") as f:
                f.write(tvm.ir.save_json(self._byoc_mod))
        return {
            "sub_graphs": sub_graphs,
            "byoc_graph": "byoc_graph.json",
            "byoc_mod": "byoc_module.json",
        }

    def _generate_model(
        self, graphs: List[MSCGraph] = None, weights: List[Dict[str, tvm.nd.array]] = None
    ) -> Any:
        """Codegen the model according to framework

        Parameters
        -------
        graphs: list<MSCgraph>
            The msc graphs.
        weights: list<dic<str, tvm.nd.array>>
            The weights

        Returns
        -------
        model: tvm.IRModule
            The relax module
        """

        graph_infos = list(zip(graphs or self._graphs, weights or self._weights))
        extra_option = self._generate_config.get("extra_option", {})
        if self._stage == MSCStage.COMPILE and not self.get_tool(ToolType.TRACKER):
            extra_option["tool_tag"] = ""
        else:
            extra_option["tool_tag"] = self._name
        return self.codegen_func(
            self._byoc_mod,
            graph_infos,
            codegen_configs=self._generate_config.get("codegen"),
            print_configs=self._generate_config.get("print"),
            extra_options=extra_option,
            build_folder=self._generate_config["build_folder"],
            output_folder=self._generate_config.get("output_folder", msc_utils.get_output_dir()),
        )

    def _to_runnable(self, model: Any, device: str, is_training: bool) -> Any:
        """Build runnable object

        Parameters
        -------
        model: Any
            The runnable model on cpu.
        device: str
            The device for place model
        is_training: bool
            Whether to load model for training

        Returns
        -------
        runnable: Any
            The runnable
        """

        model = tvm.relax.transform.LegalizeOps()(model)
        if device == "cpu":
            target = tvm.target.Target("llvm")
            with tvm.transform.PassContext(opt_level=3):
                relax_exec = tvm.relax.build(model, target)
                runnable = tvm.relax.VirtualMachine(relax_exec, tvm.cpu())
        elif device.startswith("cuda"):
            target = tvm.target.Target("cuda")
            with target:
                model = tvm.tir.transform.DefaultGPUSchedule()(model)
            with tvm.transform.PassContext(opt_level=3):
                relax_exec = tvm.relax.build(model, target)
                runnable = tvm.relax.VirtualMachine(relax_exec, tvm.cuda())
        else:
            raise NotImplementedError("Unsupported device " + str(device))
        return runnable

    def _call_runnable(
        self, runnable: tvm.relax.VirtualMachine, inputs: Dict[str, np.ndarray], device: str
    ) -> Union[List[np.ndarray], Dict[str, np.ndarray]]:
        """Call the runnable to get outputs

        Parameters
        -------
        runnable: tvm.relax.VirtualMachine
            The virtual machine.
        inputs: dict<str, data>
            The inputs in dict.
        device: str
            The device.

        Returns
        -------
        outputs: list<data>
            The outputs in list.
        """

        model_inputs = self.get_inputs()
        if device == "cpu":
            tvm_inputs = [tvm.nd.array(inputs[i["name"]]) for i in model_inputs]
        elif device.startswith("cuda"):
            dev_id = int(device.split(":")[1]) if ":" in device else 0
            tvm_inputs = [
                tvm.nd.array(inputs[i["name"]], device=tvm.cuda(dev_id)) for i in model_inputs
            ]
        else:
            raise NotImplementedError("Unsupported device " + str(device))
        return runnable["main"](*tvm_inputs)

    def _inspect_model(self) -> dict:
        """Inspect the model

        Returns
        -------
        model_info: dict
            The inspected model info
        """

        if self._debug_level >= 3:
            for idx, graph in enumerate(self._graphs):
                self._logger.debug(
                    msc_utils.msg_block("RUNNER.GRAPH[{}].INFO".format(idx), graph.inspect())
                )
        return self._byoc_graph.inspect()

    def _device_enabled(self, device: str) -> bool:
        """Check if the device is enabled

        Returns
        -------
        enabled: bool
            Whether the device is enabled.
        """

        if device == "cpu":
            return True
        if device.startswith("cuda"):
            dev_id = int(device.split(":")[1]) if ":" in device else 0
            return tvm.cuda(dev_id).exist
        return False

    @property
    def partition_func(self):
        raise NotImplementedError("partition_func is not implemented for " + str(self.__class__))
