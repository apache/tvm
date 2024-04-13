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
"""tvm.contrib.msc.pipeline.manager"""

from typing import Any, List, Tuple

from tvm.contrib.msc.core.gym.control import create_controller
from tvm.contrib.msc.core.utils.message import MSCStage
from tvm.contrib.msc.core import utils as msc_utils
from .pipeline import BasePipeline
from .worker import MSCPipeWorker


class MSCManager(BasePipeline):
    """Manager of Pipeline, process static model"""

    def setup(self) -> dict:
        """Setup the pipeline

        Returns
        -------
        info: dict
            The setup info.
        """

        self._worker = self.create_worker(self._model, "main")
        self._config = self._worker._config
        return super().setup()

    def _prepare(self, data_loader: Any) -> Tuple[dict, dict]:
        """Prepare datas for the pipeline.

        Parameters
        ----------
        data_loader:
            The data loader.

        Returns
        -------
        info: dict
            The info of prepare.
        report: dict
            The report of prepare.
        """

        return self._worker.prepare(data_loader)

    def _parse(self) -> Tuple[dict, dict]:
        """Parse relax module for the pipeline.

        Returns
        -------
        info: dict
            The info of parse.
        report: dict
            The report of parse.
        """

        return self._worker.parse()

    def _tool_applied(self, tool_type: str) -> bool:
        """Check if the tool is applied

        Parameters
        ----------
        tool_type: str
            The tool type.

        Returns
        -------
        applied: bool
            Whether the tool is applied.
        """

        return self._worker.tool_applied(tool_type)

    def _apply_tool(
        self, tool_type: str, knowledge: dict = None, data_loader: Any = None
    ) -> Tuple[dict, dict]:
        """Apply tool with runner

        Parameters
        ----------
        tool_type: str
            The tool type to apply.
        knowledge: dict
            The pre knowledge.
        data_loader:
            The data loader.

        Returns
        -------
        info: dict
            The info of apply tool.
        report: dict
            The report of apply tool.
        """

        return self._worker.apply_tool(tool_type, knowledge, data_loader)

    def _create_runtime(
        self,
        stage: str,
        tools: List[str] = None,
        run_type: str = None,
        run_config: dict = None,
        visualize: bool = True,
        profile: bool = True,
        use_cache: bool = True,
    ) -> Tuple[dict, dict]:
        """Create runtime.

        Parameters
        ----------
        stage: str
            The pipeline stage.
        tools: list<str>
            The tools to apply.
        run_type: str
            The type of runner.
        run_config: dict
            The config of runner.
        visualize: bool
            Whether to visualize the runner
        profile: bool
            Whether to profile the runner.
        use_cache: bool
            Whether to use cache.

        Returns
        -------
        info: dict
            The info of stage.
        report: dict
            The report of stage.
        """

        return self._worker.create_runner(
            stage, tools, run_type, run_config, visualize, profile, use_cache
        )

    def _run_gym(self, stage: str, config: dict, knowledge: dict, data_loader: Any) -> dict:
        """Run gym.

        Parameters
        ----------
        stage: str
            The pipeline stage.
        config: dict
            The gym config.
        knowledge: dict
            The pre knowledge.
        data_loader:
            The data loader.

        Returns
        -------
        knowledge: dict
            The learned knowledge.
        """

        extra_config = {
            "env": {
                "runner": self._worker.runner,
                "data_loader": data_loader,
                "knowledge": knowledge,
            },
            "verbose": self._verbose,
        }
        controller = create_controller(stage, config, extra_config)
        return controller.run()

    def _export_model(self, stage: str, folder: msc_utils.MSCDirectory, dump: bool = True) -> Any:
        """Export the model

        Parameters
        ----------
        stage: str
            The pipeline stage.
        folder: MSCDirectory
            The export folder.
        dump: bool
            Whether to dump info.

        Returns
        -------
        exported:
            The exported model.
        """

        return self._worker.export_model(stage, folder, dump)

    def _export_tool(self, tool_type: str, folder: msc_utils.MSCDirectory) -> dict:
        """Export the tool

        Parameters
        ----------
        tool_type: str
            The tool type.
        folder: MSCDirectory
            The export folder.

        Returns
        -------
        config: dict
            The exported tool config.
        """

        assert tool_type in self._tools_config, "Can not find tool_type " + str(tool_type)
        exp_config = {"tool_config": self._worker.export_tool(tool_type, folder)}
        return msc_utils.update_dict(self._tools_config[tool_type], exp_config)

    def _export_info(self, stage: str, folder: msc_utils.MSCDirectory) -> dict:
        """Export the info of pipeline

        Parameters
        ----------
        stage: str
            The pipeline stage.
        folder: MSCDirectory
            The export folder.

        Returns
        -------
        info: dict
            The info.
        """

        info = super()._export_info(stage, folder)
        if stage in (MSCStage.OPTIMIZE, MSCStage.COMPILE):
            info.update(self._worker.export_info(stage, folder))
        return info

    def _destory(self):
        """Destory the pipeline"""

        self._worker.destory()

    def get_runtime(self, ret_type: str = "runner") -> Any:
        """Get the runtime of pipeline

        Parameters
        ----------
        ret_type: str
            The return type runner| runnable| model.

        Returns
        -------
        runnable:
            The runnable object.
        """

        return self._worker.get_runnable(ret_type)

    def pipe_mark(self, msg: Any) -> str:
        """Mark the message with pipeline info

        Parameters
        -------
        msg: str
            The message

        Returns
        -------
        msg: str
            The message with mark.
        """

        return "MANAGER " + str(msg)

    @property
    def worker_cls(self):
        return MSCPipeWorker
