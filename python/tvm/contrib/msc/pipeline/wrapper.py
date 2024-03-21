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
"""tvm.contrib.msc.pipeline.wrapper"""

import shutil
from typing import Any, Union, List

from tvm.contrib.msc.core.tools.tool import BaseTool, ToolType
from tvm.contrib.msc.core.utils.message import MSCStage
from tvm.contrib.msc.core.utils.namespace import MSCFramework
from tvm.contrib.msc.core import utils as msc_utils
from .manager import MSCManager
from .config import create_config


class BaseWrapper(object):
    """Base Wrapper of models

    Parameters
    ----------
    model: Any
        The raw model in framwork.
    config: dict
        The config for pipeline
    plugins: dict
        The plugins for pipeline.
    """

    def __init__(
        self,
        model: Any,
        config: dict,
        workspace: str = "msc_workspace",
        plugins: dict = None,
    ):
        self._meta_model = model
        self._optimized_model, self._compiled_model = None, None
        self._config = config
        self._plugins = plugins
        verbose = config.get("verbose", "info")
        self._debug = verbose.startswith("debug")
        self._workspace = msc_utils.msc_dir(workspace, keep_history=self._debug)
        log_path = self._workspace.relpath("MSC_LOG", keep_history=False)
        self._config["logger"] = msc_utils.create_file_logger(verbose, log_path)
        self._manager = None
        self.setup()

    def __str__(self):
        if self.compiled:
            phase = "compiled"
        elif self.optimized:
            phase = "optimized"
        else:
            phase = "meta"
        return "({}) {}".format(phase, self._get_model().__str__())

    def __getattr__(self, name):
        if hasattr(self._get_model(), name):
            return getattr(self._get_model(), name)
        return self._get_model().__getattr__(name)

    def setup(self):
        """Setup the wrapper"""

        return

    def optimize(self, workspace: str = "Optimize"):
        """Optimize the model

        Parameters
        ----------
        workspace: str
            The workspace.
        """

        self.logger.info("[Wrapper] Start optimize model")
        config = msc_utils.copy_dict(self._config)
        config["workspace"] = self._workspace.create_dir(workspace)
        if MSCStage.OPTIMIZE not in config:
            config[MSCStage.OPTIMIZE] = {
                "run_type": self.model_type(),
                "profile": {"check": {"atol": 1e-3, "rtol": 1e-3}, "benchmark": {"repeat": -1}},
            }
        self._manager = MSCManager(self._meta_model, config, self._plugins, run_compile=False)
        report = self._manager.run_pipe()
        if report["success"]:
            self._optimized_model = self._manager.get_runnable("runnable")
        return self

    def compile(
        self, workspace: str = "Compile", ckpt_path: str = "Checkpoint", dump: bool = False
    ):
        """Compile the model

        Parameters
        ----------
        workspace: str
            The workspace.
        ckpt_path: str
            The path to export checkpoint.
        dump: bool
            Whether to dump the info.
        """

        if self._optimized_model:
            self.logger.info("[Wrapper] Start compile checkpoint")
            ckpt_path = self._workspace.create_dir(ckpt_path).path
            pipeline = self.export(ckpt_path, dump=dump)
            pipeline["config"]["workspace"] = self._workspace.create_dir(workspace)
            self._manager = MSCManager(**pipeline)
            report = self._manager.run_pipe()
            if report["success"]:
                self._compiled_model = self._manager.get_runnable("runnable")
            if not self._debug:
                shutil.rmtree(ckpt_path)
        else:
            self.logger.info("[Wrapper] Start compile model")
            config = msc_utils.copy_dict(self._config)
            config["workspace"] = self._workspace.create_dir(workspace)
            self._manager = MSCManager(self._meta_model, config, self._plugins)
            report = self._manager.run_pipe()
            if report["success"]:
                self._compiled_model = self._manager.get_runnable("runnable")
        return self

    def export(self, path: str = "msc_export", dump: bool = True) -> Union[str, dict]:
        """Export compile pipeline

        Parameters
        ----------
        path: str
            The export path.
        dump: bool
            Whether to dump the info.

        Returns
        -------
        export_path/pipeline: str/dict
            The exported path/pipeline info.
        """

        if not self._manager:
            self._manager = MSCManager(self._meta_model, self._config, self._plugins)
        exported = self._manager.export(path, dump=dump)
        if not self._debug:
            self._manager.destory()
        return exported

    def get_tools(self, tool_types: List[str]) -> List[BaseTool]:
        """Get the tools from manager

        Parameters
        ----------
        tool_types: list<str>
            The tool types.

        Returns
        -------
        tools: list<BaseTool>
            The tools.
        """

        if not self._manager:
            return []
        tool_types = tool_types or ToolType.all_types()
        tools = []
        for t in tool_types:
            tool = self._manager.runner.get_tool(t)
            if tool:
                tools.append(tool)
        return tools

    def disable_tools(self, tool_types: List[str]):
        """Disable the tools

        Parameters
        ----------
        tool_types: list<str>
            The tool types.
        """

        for tool in self.get_tools(tool_types):
            tool.disable()

    def enable_tools(self, tool_types: List[str]):
        """Enable the tools

        Parameters
        ----------
        tool_types: list<str>
            The tool types.
        """

        for tool in self.get_tools(tool_types):
            tool.enable()

    def _get_model(self) -> Any:
        return self._compiled_model or self._optimized_model or self._meta_model

    def _get_framework(self) -> str:
        return self._manager.runner.framework if self._manager else self.model_type()

    @property
    def optimized(self):
        return self._optimized_model is not None

    @property
    def compiled(self):
        return self._compiled_model is not None

    @property
    def device(self):
        if self._manager:
            return self._manager.runner.device
        return "cpu"

    @property
    def logger(self):
        return self._config["logger"]

    @classmethod
    def create_config(
        cls,
        inputs: List[dict],
        outputs: List[str],
        baseline_type: str = None,
        optimize_type: str = None,
        compile_type: str = None,
        **kwargs,
    ) -> dict:
        """Create config for msc pipeline

        Parameters
        ----------
        inputs: list<dict>
            The inputs info,
        outputs: list<str>
            The output names.
        baseline_type: str
            The baseline type.
        compile_type: str
            The compile type.
        optimize_type: str
            The optimize type.
        kwargs: dict
            The config kwargs.
        """

        return create_config(
            inputs, outputs, cls.model_type(), baseline_type, optimize_type, compile_type, **kwargs
        )

    @classmethod
    def model_type(cls):
        return MSCFramework.MSC


class TorchWrapper(BaseWrapper):
    """Wrapper of torch models"""

    def __call__(self, *inputs):
        framework = self._get_framework()
        if framework != MSCFramework.TORCH:
            inputs = [msc_utils.cast_array(i, framework, self.device) for i in inputs]
        outputs = self._get_model()(*inputs)
        if framework == MSCFramework.TORCH:
            return outputs
        if isinstance(outputs, (tuple, list)):
            return [msc_utils.cast_array(o, MSCFramework.TORCH, self.device) for o in outputs]
        return msc_utils.cast_array(outputs, MSCFramework.TORCH)

    def parameters(self):
        framework = self._get_framework()
        if framework == MSCFramework.TORCH:
            return self._get_model().parameters()
        return self._manager.runner.get_weights(MSCFramework.TORCH)

    def train(self):
        if self._manager:
            self._manager.runner.train()
        if self._get_framework() == MSCFramework.TORCH:
            return self._get_model().train()
        return self._get_model()

    def eval(self):
        if self._manager:
            self._manager.runner.eval()
        if self._get_framework() == MSCFramework.TORCH:
            return self._get_model().eval()
        return self._get_model()

    @classmethod
    def model_type(cls):
        return MSCFramework.TORCH
