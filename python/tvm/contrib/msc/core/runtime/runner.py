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
from typing import Dict, Optional, Any, List, Tuple, Union, Iterable
import numpy as np

import tvm
from tvm.contrib.msc.core.ir import MSCGraph
from tvm.contrib.msc.core.frontend import from_relax
from tvm.contrib.msc.core.tools import BaseTool, ToolType, ToolScope, create_tool, remove_tools
from tvm.contrib.msc.core.utils.namespace import MSCFramework
from tvm.contrib.msc.core.utils.message import MSCStage
from tvm.contrib.msc.core import utils as msc_utils
from tvm.contrib.msc.core import _ffi_api
from .hook import load_runner_hook


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
    build_config: dict
        The config for build runnable.
    device: str
        The device to build runnable.
    training: bool
        Whether compile model to trainable
    stage: str
        The stage of runner.
    plugin: PluginManager
        The plugin manager.
    name: str
        The name of the runner
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
        build_config: Optional[Dict[str, str]] = None,
        device: str = "cpu",
        training: bool = False,
        stage: str = "default",
        plugin: Any = None,
        name: str = "main",
        debug_level: int = 0,
        logger: logging.Logger = None,
    ):
        self._mod = mod
        self._tools_config = msc_utils.copy_dict(tools_config)
        self._translate_config = msc_utils.copy_dict(translate_config)
        self._generate_config = msc_utils.copy_dict(generate_config)
        self._build_config = msc_utils.copy_dict(build_config)
        self._device = device if self._device_enabled(device) else "cpu"
        self._stage = stage
        self._plugin = plugin
        self._name = name
        self._debug_level = debug_level
        self._training, self._trained = training, training
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
        self._graphs, self._weights = [], {}
        self._model, self._model_info = None, {}
        self._runnable = None
        # Setup tools
        self._tools = {}
        if self._tools_config:
            self._update_codegen({"use_tools": True, "tools_tag": self._name})
            for t_type, config in self._tools_config.items():
                self._tools[t_type] = create_tool(
                    self.framework,
                    t_type,
                    self._name,
                    training=self._training,
                    stage=self._stage,
                    **config,
                )
        if self._plugin:
            self._update_codegen({"use_plugin": True})
        return {
            "tools": {k: v.tool_style() for k, v in self._tools.items()},
            "plugin": self._plugin,
            "translate_config": self._translate_config,
            "generate_config": self._generate_config,
            "build_config": self._build_config,
            "device": self._device,
            "name": self._name,
            "debug_level": self._debug_level,
        }

    def change_stage(self, stage: str):
        """Change the stage of runner and tools"""

        self._stage = stage
        for tool in self._tools.values():
            tool.change_stage(stage)

    def change_logger(self, logger: logging.Logger):
        """Change the logger of runner and tools"""

        self._logger = logger
        for tool in self._tools.values():
            tool.change_logger(logger)

    def build(self, cache_dir: msc_utils.MSCDirectory = None, force_build: bool = False) -> Any:
        """Build the runnable object

        Parameters
        -------
        cache_dir: MSCDirectory
            cache path for save/load info
        force_build: bool
            Whether to force build the runner.

        Returns
        -------
        runnable: Any
           The runnable object.
        """

        if force_build:
            self._graphs, self._weights = [], {}
            self._model, self._model_info = None, {}
            self._runnable = None
        if cache_dir and os.path.isfile(cache_dir.relpath("cache_info.json")):
            cache_info = msc_utils.load_dict(cache_dir.relpath("cache_info.json"))
        else:
            cache_info = {}

        # Load graphs from cache
        if not self._graphs and cache_info.get("graphs"):
            self._graphs = self._load_graphs(cache_dir, cache_info["graphs"])
            assert "weights" in cache_info, "Missing weights in cache_info"
            with open(cache_dir.relpath(cache_info["weights"]), "rb") as f:
                self._weights = tvm.runtime.load_param_dict(f.read())
            self._logger.info(
                "Load %d graphs %d weights from %s",
                len(self._graphs),
                len(self._weights),
                cache_dir,
            )

        # Translate graphs from module
        if not self._graphs:
            self._graphs, self._weights = self.translate()
            self._logger.info(
                "Translate %d graphs %d weights from module", len(self._graphs), len(self._weights)
            )

        # Load model from cache
        if not self._model and cache_info.get("model"):
            self._graphs, self._weights = self.reset_tools(cache_dir=cache_dir)
            self._model = self._load_model(cache_dir, cache_info["model"])
            self._logger.info("Load model(%s) from %s", self.framework, cache_dir)

        # Generate model
        if not self._model:
            distiller = self.get_tool(ToolType.DISTILLER)
            if distiller and not distiller.distilled:
                build_root = self._generate_config["build_folder"]

                def _build_scope_model(scope: str, apply_hooks: bool):
                    self._update_codegen({"tools_scope": scope})
                    self._generate_config["build_folder"] = build_root.create_dir(scope)
                    return self.generate_model(apply_hooks=apply_hooks)

                # Generate distill model
                teacher_model = _build_scope_model(ToolScope.TEACHER, False)
                self._graphs, self._weights = self.reset_tools(cache_dir=cache_dir)
                student_model = _build_scope_model(ToolScope.STUDENT, True)
                self._model = distiller.build_model(teacher_model, student_model)
            else:
                # Generate normal model
                self._graphs, self._weights = self.reset_tools(cache_dir=cache_dir)
                self._model = self.generate_model()

            generate_msg = "Generate model({})".format(self.framework)
            if self._tools:
                self._logger.info("%s with tools: %s", generate_msg, ",".join(self._tools.keys()))
            else:
                self._logger.info("%s without tools", generate_msg)

        # Inspect model
        self._model_info = self._inspect_model()
        if self._debug_level >= 2:
            self._logger.debug(msc_utils.msg_block("RUNNER.MODEL_INFO", self._model_info))

        runnable_msg = "runnable({}, {}) @ {}".format(
            self.framework, "train" if self._training else "eval", self._device
        )
        # Load runnable from cache
        if not self._runnable and cache_info.get("runnable"):
            self._runnable = self._load_runnable(cache_dir, cache_info["runnable"])
            self._logger.info("Load %s from %s", runnable_msg, cache_dir)

        # Build runnable
        if not self._runnable:
            self._runnable = self.build_runnable()
            self._logger.info("Build %s", runnable_msg)
        return self._runnable

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
        if ret_type == "native":
            return outputs
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

    def save_cache(
        self,
        cache_dir: msc_utils.MSCDirectory,
        save_model: bool = True,
        save_runnable: bool = True,
        save_tools: bool = True,
    ):
        """Save runner to cache

        Parameters
        -------
        cache_dir: MSCDirectory
            cache path for save/load info
        save_model: bool
            Whether to save model.
        save_runnable: bool
            Whether to save runnable.
        save_tools: bool
            Whether to save tools.
        """

        cache_info = {"graphs": self._save_graphs(cache_dir), "weights": "graph_weights.bin"}
        with cache_dir:
            with open(cache_info["weights"], "wb") as f_params:
                f_params.write(tvm.runtime.save_param_dict(self._weights))
        if save_model and cache_info.get("graphs"):
            cache_info["model"] = self._save_model(cache_dir)
        if save_runnable and cache_info.get("model"):
            cache_info["runnable"] = self._save_runnable(cache_dir)
        if save_tools:
            for t_type, tool in self._tools.items():
                cache_info[t_type] = tool.save_cache(cache_dir)
        with open(cache_dir.relpath("cache_info.json"), "w") as f:
            f.write(json.dumps(cache_info, indent=2))
        self._logger.debug(
            msc_utils.msg_block("RUNNER.SAVE_CACHE", {"folder": cache_dir, "info": cache_info})
        )

    def translate(self, apply_hooks: bool = True) -> Tuple[List[MSCGraph], Dict[str, tvm.nd.array]]:
        """Translate IRModule to MSCgraphs

        Parameters
        -------
        apply_hooks: bool
            Whether to apply hooks.

        Returns
        -------
        graphs: list<MSCGraph>
            The translated graphs
        weights: dict<str, tvm.nd.array>
            The translated weights.
        """

        mod = self._mod
        if apply_hooks:
            for hook in self._translate_config.get("pre_hooks", []):
                mod = self._apply_hook("before translate", hook, mod)
        graphs, weights = self._translate(mod)
        if apply_hooks:
            for hook in self._translate_config.get("post_hooks", []):
                graphs, weights = self._apply_hook("after translate", hook, graphs, weights)
        return graphs, weights

    def _translate(self, mod: tvm.IRModule) -> Tuple[List[MSCGraph], Dict[str, tvm.nd.array]]:
        """Translate IRModule to MSCgraphs

        Parameters
        -------
        mod: tvm.IRModule
            The module to be translated.

        Returns
        -------
        graphs: list<MSCGraph>
            The translated graphs
        weights: dict<str, tvm.nd.array>
            The translated weights.
        """

        raise NotImplementedError("_translate is not implemented for " + str(self.__class__))

    def reset_tools(
        self,
        graphs: List[MSCGraph] = None,
        weights: List[Dict[str, tvm.nd.array]] = None,
        tools: List[BaseTool] = None,
        cache_dir: msc_utils.MSCDirectory = None,
    ):
        """Reset the tools

        Parameters
        -------
        graphs: list<MSCgraph>
            The msc graphs.
        weights: list<dict<str, tvm.nd.array>>
            The weights.
        tools: list<BaseTool>
            The tools.
        cache_dir: MSCDirectory
            cache path for save/load info.

        Returns
        -------
        graphs: list<MSCgraph>
            The msc graphs.
        weights: list<dict<str, tvm.nd.array>>
            The weights.
        """

        graphs = graphs or self._graphs
        weights = weights or self._weights
        tools = tools or self._tools.values()
        for tool in tools:
            graphs, weights = tool.reset(graphs, weights, cache_dir)
        return graphs, weights

    def generate_model(self, apply_hooks: bool = True) -> Any:
        """Codegen the model according to framework

        Parameters
        -------
        apply_hooks: bool
            Whether to apply hooks.

        Returns
        -------
        model: Any
            The meta model
        """

        graphs, weights = self._graphs, self._weights
        if apply_hooks:
            for hook in self._generate_config.get("pre_hooks", []):
                graphs, weights = self._apply_hook("before generate", hook, graphs, weights)
        model = self._generate_model(graphs, weights)
        if apply_hooks:
            for hook in self._generate_config.get("post_hooks", []):
                model = self._apply_hook("after generate", hook, model)
        return model

    def _generate_model(self, graphs: List[MSCGraph], weights: Dict[str, tvm.nd.array]) -> Any:
        """Codegen the model according to framework

        Parameters
        -------
        graphs: list<MSCgraph>
            The msc graphs.
        weights: dict<str, tvm.nd.array>
            The weights.

        Returns
        -------
        model: Any
            The meta model
        """

        raise NotImplementedError("_load is not implemented for " + str(self.__class__))

    def build_runnable(self, apply_hooks: bool = True) -> Any:
        """Build runnable object

        Parameters
        -------
        apply_hooks: bool
            Whether to apply hooks.

        Returns
        -------
        runnable: Any
            The runnable
        """

        model = self._model
        if apply_hooks:
            for hook in self._build_config.get("pre_hooks", []):
                model = self._apply_hook("before build", hook, model)
        runnable = self._build_runnable(model)
        if apply_hooks:
            for hook in self._build_config.get("post_hooks", []):
                runnable = self._apply_hook("after build", hook, runnable)
        return runnable

    def _build_runnable(self, model: Any) -> Any:
        """Build runnable object

        Parameters
        -------
        model: Any
            The meta model.

        Returns
        -------
        runnable: Any
            The runnable
        """

        raise NotImplementedError("_build_runnable is not implemented for " + str(self.__class__))

    def train(self):
        """Change status to train"""

        if not self._training:
            self._training = True
            for tool in self.get_tools():
                tool.train()
            self._train()

    def _train(self):
        """Change status to train"""

        self._runnable = self.build_runnable()

    def eval(self):
        """Change status to eval"""

        if self._training:
            self._trained = True
            self._training = False
            for tool in self.get_tools():
                tool.eval()
            self._eval()

    def _eval(self):
        """Change status to eval"""

        self._runnable = self.build_runnable()

    def get_tool_config(self, tool_type: str) -> dict:
        """Get tool by type

        Parameters
        -------
        tool_type: str
            The type of the tool prune| quantize| distill...

        Returns
        -------
        config: dict
            The tool config.
        """

        return self._tools_config.get(tool_type)

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

    def get_tools(self) -> Iterable[BaseTool]:
        """Get all saved tools by tag

        Returns
        -------
        tools: iterable<BaseTool>
            The saved tools.
        """

        for t_type in ToolType.all_types():
            tool = self.get_tool(t_type)
            if tool:
                yield tool

    def apply_tool(self, tool_type: str, data_loader: Any = None) -> str:
        """Execute tool and get plan

        Parameters
        -------
        tool_type: str
            The tool type, should be in ToolType
        data_loader:
            The data loader

        Returns
        -------
        plan_file: str
            The saved plan file.
        """

        assert tool_type in self._tools, "Can not find tool " + str(tool_type)
        if tool_type == ToolType.PRUNER:
            pruner = self.get_tool(ToolType.PRUNER)
            if not pruner.finalize():
                assert data_loader, "data_loader should be given to plan prune"
                for inputs in data_loader():
                    self.run(inputs, ret_type="native")
                    break
            plan = pruner.finalize()
        elif tool_type == ToolType.QUANTIZER:
            quantizer = self.get_tool(ToolType.QUANTIZER)
            while not quantizer.calibrated:
                assert data_loader, "data_loader should be given to plan prune"
                for inputs in data_loader():
                    self.run(inputs, ret_type="native")
                quantizer.calibrate()
            plan = quantizer.finalize()
        elif tool_type == ToolType.DISTILLER:
            distiller = self.get_tool(ToolType.DISTILLER)
            while not distiller.distilled:
                assert data_loader, "data_loader should be given to plan prune"
                for inputs in data_loader():
                    loss = self.run(inputs, ret_type="native")
                    distiller.learn(loss)
                distiller.distill()
            plan = distiller.finalize()
        else:
            plan = self.get_tool(tool_type).finalize()
        assert plan, "Failed to create plan for {}".format(tool_type)
        plan_file = self._tools_config[tool_type]["plan_file"]
        with open(plan_file, "w") as f:
            f.write(json.dumps(plan, indent=2))
        self._logger.info("Save %d plan(%s) -> %s", len(plan), tool_type, plan_file)
        return plan_file

    def _apply_hook(self, desc: str, hook_def: dict, *args, **kwargs) -> Any:
        """Load a registered hook

        Parameters
        ----------
        desc: str
            The description of the hook
        hook_def: dict
            The function and config of the hook.
        args: list<Any>
            The arguments for run method.
        kwargs: dict<Any>
            The key word arguments for run method.

        Returns
        -------
        result:
            The result
        """

        hook = load_runner_hook(hook_def)
        self._logger.info("Apply %s hook:\n  %s", desc, hook)
        return hook.apply(self, *args, **kwargs)

    def _update_codegen(self, config: Dict[str, Any]):
        """Update the codegen in generate_config

        Parameters
        -------
        config: dict
            The extra config for codegen.
        """

        if "codegen" not in self._generate_config:
            self._generate_config["codegen"] = {}
        codegen = self._generate_config["codegen"]
        if isinstance(codegen, dict):
            codegen.update(config)
        elif isinstance(codegen, (list, tuple)):
            for c in codegen:
                c.update(config)
        else:
            raise TypeError("Unexpecet codegen config " + str(codegen))

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

    def get_weights(self, framework: str = None, device: str = None) -> Iterable[tvm.nd.array]:
        """Get the weights from graphs

        Parameters
        -------
        framework: str
            The framework for weight.
        device: str
            The device for weight.

        Returns
        -------
        weights: generator<tvm.nd.array>
            The generator of weight datas.
        """

        device = device or self._device
        for graph in self._graphs:
            for weight in graph.get_weights():
                data = self._weights[weight.name]
                if framework:
                    data = msc_utils.cast_array(data, framework, device)
                yield data

    def destory(self):
        """Destory runner"""

        if self._model:
            self._model = None
        if self._runnable:
            self._runnable = None
        for tool in self.get_tools():
            tool.destory()
        remove_tools(self._name)

    def _load_graphs(self, cache_dir: msc_utils.MSCDirectory, cache_info: dict) -> List[MSCGraph]:
        """Load MSCGraphs from cache

        Parameters
        -------
        cache_dir: MSCDirectory
            cache path for save/load info
        cache_info: dict
            The cache info.

        Returns
        -------
        graphs: list<MSCGraph>
            The translated graphs
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
    def debug_level(self):
        return self._debug_level

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

    @classmethod
    def load_native(cls, model: Any) -> Any:
        """Load the native model

        Parameters
        -------
        model:
            The native model.

        Returns
        -------
        model:
            The loaded native model.
        """

        return model, "cpu"

    @classmethod
    def update_config(cls, stage: str, config: dict, model: Any = None) -> dict:
        """Update the config for parse

        Parameters
        -------
        stage: str
            The stage to be updated
        config: dict
            The config for pipeline.
        model:
            The native model.

        Returns
        -------
        config: dict
            The updated config.
        """

        if stage not in config:
            return config
        if stage in (MSCStage.BASELINE, MSCStage.OPTIMIZE, MSCStage.COMPILE):
            run_config = config[stage].get("run_config", {})
            if "translate_config" not in run_config:
                run_config["translate_config"] = {}
            if "build" not in run_config["translate_config"]:
                run_config["translate_config"]["build"] = {}
            if "generate_config" not in run_config:
                run_config["generate_config"] = {}
            run_config["translate_config"]["build"]["input_aliases"] = [
                i[0] for i in config["inputs"]
            ]
            run_config["translate_config"]["build"]["output_aliases"] = config["outputs"]
            config[stage]["run_config"] = run_config
        return config

    @classmethod
    def support_tool(cls, tool_type: str) -> bool:
        return True


class ModelRunner(BaseRunner):
    """Model runner of MSC"""

    def _translate(self, mod: tvm.IRModule) -> Tuple[List[MSCGraph], Dict[str, tvm.nd.array]]:
        """Translate IRModule to MSCgraphs

        Parameters
        -------
        mod: tvm.IRModule
            The module to be translated.

        Returns
        -------
        graphs: list<MSCGraph>
            The translated graphs
        weights: dict<str, tvm.nd.array>
            The translated weights.
        """

        graph, weights = from_relax(
            mod,
            trans_config=self._translate_config.get("transform"),
            build_config=self._translate_config.get("build"),
            opt_config=self._translate_config.get("optimize"),
        )
        return [graph], weights

    def _load_graphs(self, cache_dir: msc_utils.MSCDirectory, cache_info: dict) -> List[MSCGraph]:
        """Load MSCGraphs from cache

        Parameters
        -------
        cache_dir: MSCDirectory
            cache path for save/load info
        cache_info: dict
            The cache info.

        Returns
        -------
        graphs: list<MSCGraph>
            The translated graphs
        """

        assert "main" in cache_info, "main should be given in cache_info, get " + str(cache_info)
        graph = MSCGraph.from_json(cache_dir.relpath(cache_info["main"]["graph"]))
        return [graph]

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

        main_info = {"graph": self._graphs[0].name + "_graph.json"}
        with cache_dir:
            with open(main_info["graph"], "w") as f_graph:
                f_graph.write(self._graphs[0].to_json())
        return {"main": main_info}

    def _generate_model(self, graphs: List[MSCGraph], weights: Dict[str, tvm.nd.array]) -> Any:
        """Codegen the model according to framework

        Parameters
        -------
        graphs: list<MSCgraph>
            The msc graphs.
        weights: dict<str, tvm.nd.array>
            The weights.

        Returns
        -------
        model: Any
            The runnable model
        """

        return self.codegen_func(
            graphs[0],
            weights,
            codegen_config=self._generate_config.get("codegen"),
            print_config=self._generate_config.get("print"),
            build_folder=self._generate_config["build_folder"],
            plugin=self._plugin,
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

    def _translate(self, mod: tvm.IRModule) -> Tuple[List[MSCGraph], Dict[str, tvm.nd.array]]:
        """Translate IRModule to MSCgraphs

        Parameters
        -------
        mod: tvm.IRModule
            The module to be translated.

        Returns
        -------
        graphs: list<MSCGraph>
            The translated graphs
        weights: dict<str, tvm.nd.array>
            The translated weights.
        """

        self._byoc_mod, graphs, weights = self.partition_func(
            mod,
            trans_config=self._translate_config.get("transform"),
            build_config=self._translate_config.get("build"),
        )
        self._byoc_graph = _ffi_api.BuildFromRelax(
            self._byoc_mod, "main", msc_utils.dump_dict(self._translate_config.get("build"))
        )
        return graphs, weights

    def _load_graphs(self, cache_dir: msc_utils.MSCDirectory, cache_info: dict) -> List[MSCGraph]:
        """Load MSCgraphs from cache

        Parameters
        -------
        cache_dir: MSCDirectory
            cache path for save/load info
        cache_info: dict
            The cache info.

        Returns
        -------
        graphs: list<MSCGraph>
            The translated graphs
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
        graphs = [MSCGraph.from_json(cache_dir.relpath(g)) for g in cache_info["sub_graphs"]]
        self._byoc_graph = MSCGraph.from_json(cache_dir.relpath(cache_info["byoc_graph"]))
        return graphs

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

        sub_graphs = [g.name + "_graph.info" for g in self._graphs]
        with cache_dir:
            for graph, g_file in zip(self._graphs, sub_graphs):
                with open(g_file, "w") as f_graph:
                    f_graph.write(graph.to_json())
            with open("byoc_graph.json", "w") as f:
                f.write(self._byoc_graph.to_json())
            with open("byoc_module.json", "w") as f:
                f.write(tvm.ir.save_json(self._byoc_mod))
        return {
            "sub_graphs": sub_graphs,
            "byoc_graph": "byoc_graph.json",
            "byoc_mod": "byoc_module.json",
        }

    def _generate_model(self, graphs: List[MSCGraph], weights: Dict[str, tvm.nd.array]) -> Any:
        """Codegen the model according to framework

        Parameters
        -------
        graphs: list<MSCgraph>
            The msc graphs.
        weights: dict<str, tvm.nd.array>
            The weights.

        Returns
        -------
        model: tvm.IRModule
            The relax module
        """

        extra_option = self._generate_config.get("extra_option", {})
        if self._stage == MSCStage.COMPILE and not self.get_tool(ToolType.TRACKER):
            extra_option["tool_tag"] = ""
        else:
            extra_option["tool_tag"] = self._name
        return self.codegen_func(
            self._byoc_mod,
            graphs,
            weights,
            codegen_configs=self._generate_config.get("codegen"),
            print_configs=self._generate_config.get("print"),
            extra_options=extra_option,
            build_folder=self._generate_config["build_folder"],
            output_folder=self._generate_config.get("output_folder", msc_utils.get_output_dir()),
            plugin=self._plugin,
        )

    def _build_runnable(self, model: Any) -> Any:
        """Build runnable object

        Parameters
        -------
        model: Any
            The meta model.

        Returns
        -------
        runnable: Any
            The runnable
        """

        model = tvm.relax.transform.LegalizeOps()(model)
        if self._device == "cpu":
            target = tvm.target.Target("llvm")
            with tvm.transform.PassContext(opt_level=3):
                relax_exec = tvm.relax.build(model, target)
                runnable = tvm.relax.VirtualMachine(relax_exec, tvm.cpu())
        elif self._device.startswith("cuda"):
            target = tvm.target.Target("cuda")
            with target:
                model = tvm.tir.transform.DefaultGPUSchedule()(model)
            with tvm.transform.PassContext(opt_level=3):
                relax_exec = tvm.relax.build(model, target)
                runnable = tvm.relax.VirtualMachine(relax_exec, tvm.cuda())
        else:
            raise NotImplementedError("Unsupported device " + str(self._device))
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

        if self._debug_level >= 2:
            for idx, graph in enumerate(self._graphs):
                self._logger.debug(
                    msc_utils.msg_block("GRAPH[{}].INFO".format(idx), graph.inspect())
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
