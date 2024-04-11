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
from tvm.contrib.msc.core.codegen import to_relax
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
    tools_config: list<dict>
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
        Whether compile model to trainable.
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
        tools_config: Optional[List[dict]] = None,
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
        if tools_config:
            self._tools_type = [t["tool_type"] for t in tools_config]
            self._tools_config = {
                t["tool_type"]: msc_utils.copy_dict(t["tool_config"]) for t in tools_config
            }
        else:
            self._tools_type, self._tools_config = [], {}
        self._translate_config = msc_utils.copy_dict(translate_config)
        self._generate_config = msc_utils.copy_dict(generate_config)
        self._build_config = msc_utils.copy_dict(build_config)
        self._device = device if self.support_device(device) else "cpu"
        self._stage = stage
        self._plugin = plugin
        self._name = name
        self._debug_level = debug_level
        self._training, self._trained = training, training
        self._logger = logger or msc_utils.get_global_logger()
        self._logger.info(msc_utils.msg_block(self.runner_mark("SETUP"), self.setup()))
        self._tools = self.setup_tools()

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
        if self._plugin:
            self._update_codegen({"use_plugin": True})
        return {
            "tools": {k: v.get("tool_style", "default") for k, v in self._tools_config.items()},
            "plugin": self._plugin,
            "translate_config": self._translate_config,
            "generate_config": self._generate_config,
            "build_config": self._build_config,
            "device": self._device,
            "name": self._name,
            "debug_level": self._debug_level,
        }

    def setup_tools(self) -> Dict[str, BaseTool]:
        """Setup tools

        Returns
        -------
        tools: dict
            The tools.
        """

        tools = {}
        if self._tools_type:
            self._update_codegen({"use_tools": True, "tools_tag": self._name})
            for t_type in self._tools_type:
                tools[t_type] = create_tool(
                    self.framework,
                    t_type,
                    self._name,
                    training=self._training,
                    stage=self._stage,
                    **self._tools_config[t_type],
                )
        return tools

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

    def build(
        self,
        cache_dir: msc_utils.MSCDirectory = None,
        force_build: bool = False,
        disable_tools: List[str] = None,
    ) -> Any:
        """Build the runnable object

        Parameters
        -------
        cache_dir: MSCDirectory
            cache path for save/load info
        force_build: bool
            Whether to force build the runner.
        disable_tools: list<str>
            The tool types to be disabled.

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

        # set tools to reset
        if disable_tools:
            tools = [t for t in self.get_tools() if t.tool_type not in disable_tools]
        else:
            tools = None

        build_msg = ""
        # Load graphs from cache
        if not self._graphs and cache_info.get("graphs"):
            self._graphs = self._load_graphs(cache_dir, cache_info["graphs"])
            assert "weights" in cache_info, "Missing weights in cache_info"
            with open(cache_dir.relpath(cache_info["weights"]), "rb") as f:
                self._weights = tvm.runtime.load_param_dict(f.read())
            build_msg += "Load "

        # Translate graphs from module
        if not self._graphs:
            self._graphs, self._weights = self.translate()
            build_msg += "Translate "
        build_msg += "{} graphs {} weights -> ".format(len(self._graphs), len(self._weights))

        # Load model from cache
        if not self._model and cache_info.get("model"):
            self._graphs, self._weights = self.reset_tools(tools=tools, cache_dir=cache_dir)
            self._model = self._load_model(cache_dir, cache_info["model"])
            build_msg += "Load "

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
                self._graphs, self._weights = self.reset_tools(tools=tools, cache_dir=cache_dir)
                student_model = _build_scope_model(ToolScope.STUDENT, True)
                self._model = distiller.build_model(teacher_model, student_model)
            else:
                # Generate normal model
                self._graphs, self._weights = self.reset_tools(tools=tools, cache_dir=cache_dir)
                self._model = self.generate_model()
            build_msg += "Generate "

        # Add tool message
        if self._tools:
            build_msg += "model with tools " + str(",".join(self._tools.keys())) + " -> "
        else:
            build_msg += "model without tools -> "

        # Inspect model
        self._model_info = self._inspect_model()
        if self._debug_level >= 2:
            self._logger.debug(
                msc_utils.msg_block(self.runner_mark("MODEL_INFO"), self._model_info)
            )

        # Load runnable from cache
        if not self._runnable and cache_info.get("runnable"):
            self._runnable = self._load_runnable(cache_dir, cache_info["runnable"])
            build_msg += "Load "

        # Build runnable
        if not self._runnable:
            self._runnable = self.build_runnable()
            build_msg += "Build "
        build_msg += "runnable({}, {}) on {}".format(
            self.framework, "train" if self._training else "eval", self._device
        )
        self._logger.info(self.runner_mark(build_msg))
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

        in_names = [i["name"] for i in self.get_inputs()]
        inputs = msc_utils.format_datas(inputs, in_names, style="dict")
        outputs = self._call_runnable(self._runnable, inputs, self._device)
        if ret_type == "native":
            return outputs
        out_names = [o["name"] for o in self.get_outputs()]
        return msc_utils.format_datas(outputs, out_names, style=ret_type)

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
        title = self.runner_mark("SAVE_CACHE")
        self._logger.debug(msc_utils.msg_block(title, {"folder": cache_dir, "info": cache_info}))

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
        if tools is None:
            tools = list(self.get_tools())
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

    def export_module(self, folder: msc_utils.MSCDirectory) -> tvm.IRModule:
        """Export the module from graphs

        Parameters
        ----------
        folder: MSCDirectory
            The export folder.

        Returns
        -------
        module: IRModule
            The exported module
        """

        raise NotImplementedError("export_module is not implemented for " + str(self.__class__))

    def export_runnable(self, folder: msc_utils.MSCDirectory) -> dict:
        """Export the runnable

        Parameters
        -------
        folder: MSCDirectory
            The export folder.

        Returns
        -------
        info: dict
            The runnable info.
        """

        raise NotImplementedError("export_runnable is not implemented for " + str(self.__class__))

    def export_graphs(self, folder: msc_utils.MSCDirectory) -> dict:
        """Export the graphs

        Parameters
        -------
        folder: MSCDirectory
            The export folder.

        Returns
        -------
        info: dict
            The graphs info.
        """

        raise NotImplementedError("export_graphs is not implemented for " + str(self.__class__))

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
            self._training, self._trained = False, True
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

    def make_plan(self, tool_type: str, data_loader: Any = None) -> str:
        """Execute tool and get plan

        Parameters
        -------
        tool_type: str
            The tool type, should be in ToolType
        data_loader:
            The data loader.

        Returns
        -------
        plan_file: str
            The saved plan file.
        """

        def _finalize_tool(
            checker: callable, post_batch: callable = None, post_iter: callable = None
        ):
            tool = self.get_tool(tool_type)
            while not checker(tool):
                assert data_loader, "data_loader should be given to make plan for " + tool_type
                for inputs in data_loader():
                    outputs = self.run(inputs, ret_type="native")
                    if post_batch:
                        post_batch(tool, outputs)
                    if checker(tool):
                        break
                if post_iter:
                    post_iter(tool)
            return tool.finalize()

        assert tool_type in self._tools, "Can not find tool " + str(tool_type)
        if tool_type == ToolType.PRUNER:
            plan = _finalize_tool(lambda t: t.pruned)
        elif tool_type == ToolType.QUANTIZER:
            plan = _finalize_tool(lambda t: t.calibrated, post_iter=lambda t: t.calibrate())
        elif tool_type == ToolType.DISTILLER:
            plan = _finalize_tool(
                lambda t: t.distilled,
                post_batch=lambda t, outputs: t.learn(outputs),
                post_iter=lambda t: t.distill(),
            )
        elif tool_type == ToolType.TRACKER:
            plan = _finalize_tool(lambda t: t.tracked)
        else:
            plan = self.get_tool(tool_type).finalize()
        self._logger.debug("Made %d plan for %s", len(plan), tool_type)
        plan_file = self._tools_config[tool_type]["plan_file"]
        if plan:
            with open(plan_file, "w") as f:
                f.write(json.dumps(plan, indent=2))
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

    def visualize(self, visual_dir: msc_utils.MSCDirectory, export_graph: bool = False):
        """Visualize MSCGraphs

        Parameters
        -------
        visual_dir: MSCDirectory
            Visualize path for saving graph
        export_graph: bool
            Whether to export the graph
        """

        for graph in self._graphs:
            graph.visualize(visual_dir.relpath(graph.name + ".prototxt"))
            if export_graph:
                with open(visual_dir.relpath(graph.name + "_graph.json"), "w") as f_graph:
                    f_graph.write(graph.to_json())
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

    def get_runtime_params(self) -> Dict[str, tvm.nd.array]:
        """Get the runtime parameters

        Returns
        -------
        params: dict<str, tvm.nd.array>
            The parameters from runtime.
        """

        return self._get_runtime_params()

    def _get_runtime_params(self) -> Dict[str, tvm.nd.array]:
        """Get the runtime parameters

        Returns
        -------
        params: dict<str, tvm.nd.array>
            The parameters from runtime.
        """

        raise NotImplementedError(
            "_get_runtime_params is not implemented for " + str(self.__class__)
        )

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

    def runner_mark(self, msg: Any) -> str:
        """Mark the message with runner info

        Parameters
        -------
        msg: str
            The message

        Returns
        -------
        msg: str
            The message with mark.
        """

        return "RUNNER[{}]({} @ {}) {}".format(self._name, self.framework, self._stage, msg)

    @property
    def stage(self):
        return self._stage

    @property
    def debug_level(self):
        return self._debug_level

    @property
    def trained(self):
        return self._trained

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
    def load_native(cls, model: Any, config: dict) -> Tuple[Any, str, bool]:
        """Load the native model

        Parameters
        -------
        model:
            The native model.
        config: dict
            The config for pipeline.

        Returns
        -------
        model:
            The loaded native model.
        device: str
            The device of the model.
        training:
            Whether the model is for training.
        """

        return model, "cpu", False

    @classmethod
    def run_native(
        cls,
        model: Any,
        inputs: Dict[str, np.ndarray],
        input_names: List[str],
        output_names: List[str],
        warm_up: int = 10,
        repeat: int = 0,
    ) -> Tuple[Dict[str, np.ndarray], float]:
        """Run the datas and get outputs

        Parameters
        -------
        model:
            The nativate model.
        inputs: dict<str, data>
            The inputs in dict.
        input_names: list<str>
            The input names.
        output_names: list<str>
            The outut names.
        warm_up: int
            The warm_up num for profile.
        repeat: int
            The repeat num for profile.

        Returns
        -------
        outputs: dict<str, np.array>
            The outputs in dict.
        avg_time: float
            The average time.
        """

        raise NotImplementedError("run_native is not implemented for " + str(cls))

    @classmethod
    def dump_nativate(
        cls, model: Any, folder: msc_utils.MSCDirectory, dump_config: dict = None
    ) -> str:
        """Dump the nativate model

        Parameters
        -------
        model:
            The native model.
        folder: MSCDirectory
            The export folder.
        dump_config: dict
            The dump config.

        Returns
        -------
        export_path: str
            The exported path
        """

        raise NotImplementedError("dump_nativate is not implemented for " + str(cls))

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
    def support_device(cls, device: str) -> bool:
        """Check if the device is enabled

        Returns
        -------
        enabled: bool
            Whether the device is enabled.
        """

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

    def export_module(self, folder: msc_utils.MSCDirectory) -> tvm.IRModule:
        """Export the module from graphs

        Parameters
        ----------
        folder: MSCDirectory
            The export folder.

        Returns
        -------
        module: IRModule
            The exported module
        """

        build_folder = folder.create_dir("export_build", keep_history=False, cleanup=True)
        module = to_relax(
            self._graphs[0], self.get_runtime_params(), build_folder=build_folder, use_alias=False
        )
        return module

    def export_graphs(self, folder: msc_utils.MSCDirectory) -> dict:
        """Export the graphs

        Parameters
        -------
        folder: MSCDirectory
            The export folder.

        Returns
        -------
        info: dict
            The graphs info.
        """

        graphs = {"main": folder.relpath(self._graphs[0].name + "_graph.json")}
        with open(graphs["main"], "w") as f_graph:
            f_graph.write(self._graphs[0].to_json())
        return graphs


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
        self._executable = None
        return super().setup()

    def visualize(self, visual_dir: msc_utils.MSCDirectory, export_graph: bool = False):
        """Visualize MSCGraphs

        Parameters
        -------
        visual_dir: MSCDirectory
            Visualize path for saving graph
        export_graph: bool
            Whether to export the graph
        """

        super().visualize(visual_dir)
        self._byoc_graph.visualize(visual_dir.relpath(self._byoc_graph.name + ".prototxt"))
        if export_graph:
            with open(visual_dir.relpath(self._byoc_graph.name + "_graph.json"), "w") as f_graph:
                f_graph.write(self._byoc_graph.to_json())

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

        sub_graphs = [g.name + "_graph.json" for g in self._graphs]
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
        extra_option["tool_tag"] = "" if self._stage == MSCStage.COMPILE else self._name
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
                self._executable = tvm.relax.build(model, target)
                runnable = tvm.relax.VirtualMachine(self._executable, tvm.cpu())
        elif self._device.startswith("cuda"):
            target = tvm.target.Target("cuda")
            with target:
                model = tvm.tir.transform.DefaultGPUSchedule()(model)
            with tvm.transform.PassContext(opt_level=3):
                self._executable = tvm.relax.build(model, target)
                runnable = tvm.relax.VirtualMachine(self._executable, tvm.cuda())
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

        input_names = [i["name"] for i in self.get_inputs()]
        tvm_inputs = [
            msc_utils.cast_array(inputs[i], MSCFramework.TVM, device) for i in input_names
        ]
        return runnable["main"](*tvm_inputs)

    def _inspect_model(self) -> dict:
        """Inspect the model

        Returns
        -------
        model_info: dict
            The inspected model info
        """

        if self._debug_level >= 2:
            sub_graphs = {g.name: g.inspect for g in self._graphs}
            title = self.runner_mark("SUBGRAPHS({})".format(len(sub_graphs)))
            self._logger.debug(msc_utils.msg_block(title, sub_graphs))
        return self._byoc_graph.inspect()

    def export_runnable(self, folder: msc_utils.MSCDirectory) -> dict:
        """Export the runnable

        Parameters
        -------
        folder: MSCDirectory
            The export folder.

        Returns
        -------
        info: dict
            The runnable info.
        """

        export_lib = folder.relpath("lib.so")
        self._executable.export_library(export_lib)
        return {
            "lib": export_lib,
            "device": self.device,
            "model_type": self.framework,
            "abstract": self.model_info,
        }

    def export_graphs(self, folder: msc_utils.MSCDirectory) -> dict:
        """Export the graphs

        Parameters
        -------
        folder: MSCDirectory
            The export folder.

        Returns
        -------
        info: dict
            The graphs info.
        """

        graphs = {
            "byoc_graph": folder.relpath(self._byoc_graph.name + "_graph.json"),
            "sub_graphs": {g.name: folder.relpath(g.name + "_graph.json") for g in self._graphs},
        }
        with open(graphs["byoc_graph"], "w") as f:
            f.write(self._byoc_graph.to_json())
        for graph in self._graphs:
            with open(graphs["sub_graphs"][graph.name], "w") as f:
                f.write(graph.to_json())
        return graphs

    @property
    def partition_func(self):
        raise NotImplementedError("partition_func is not implemented for " + str(self.__class__))

    @classmethod
    def support_device(cls, device: str) -> bool:
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
