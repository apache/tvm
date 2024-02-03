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
"""tvm.contrib.msc.core.tools.track.tracker"""

from typing import Any, List
from tvm.contrib.msc.core.tools.tool import ToolType, BaseTool, ToolStrategy
from tvm.contrib.msc.core import utils as msc_utils


class BaseTracker(BaseTool):
    """Base tracker for all"""

    def setup(self) -> dict:
        """Setup the tool

        Returns
        -------
        info: dict
            The setup info.
        """

        # filter plan
        def _filter_info(info: dict) -> dict:
            return {k: v for k, v in info.items() if k != self._stage}

        self._plan = {k: _filter_info(v) for k, v in self._plan.items()}
        data_folder = msc_utils.get_dataset_dir().create_dir("Track")
        self._loaders = {}
        for folder in data_folder.listdir():
            if folder == self._stage:
                continue
            if msc_utils.is_simple_dataset(data_folder.relpath(folder)):
                self._loaders[folder] = msc_utils.SimpleDataLoader(data_folder.relpath(folder))
        self._saver = msc_utils.SimpleDataSaver(data_folder.relpath(self._stage))
        self._max_iter = self._options.get("max_iter", 1)
        info = super().setup()
        info.update({"saver": self._saver, "loaders": self._loaders})
        return info

    def finalize(self) -> dict:
        """Get the plan"""

        self._saver.finalize()
        return super().finalize()

    def _execute_after_forward(self, output: Any) -> Any:
        """Execute after model forward

        Parameters
        ----------
        output: Any
            The output reference of the model.

        Returns
        -------
        output: Any
           The modified output reference.
        """

        if self._forward_cnt < self._max_iter:
            passed = {}
            for info in self._plan.values():
                if "diffs" not in info[self._stage]:
                    continue
                for stage, p_info in info[self._stage]["diffs"].items():
                    if stage not in passed:
                        passed[stage] = {"total": 0, "passed": 0}
                    passed[stage]["total"] += 1
                    if p_info["pass"]:
                        passed[stage]["passed"] += 1
            msg = "Track({})[{}] {} datas".format(self._stage, self._forward_cnt, len(self._plan))
            if passed:
                msg += ", passed -> "
                msg += "; ".join(
                    ["{}: {}/{}".format(s, i["passed"], i["total"]) for s, i in passed.items()]
                )
            self._logger.info(msg)
        return output

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

        if self._forward_cnt >= self._max_iter:
            return False
        strategy = self._get_tensor_strategy(name, consumer)
        if not strategy:
            return False
        compare_to = strategy.get_config().get("compare_to", {})
        if self._stage in compare_to:
            return True
        for stages in compare_to.values():
            if self._stage in stages:
                return True
        return False

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

        return self._track_tensor(tensor, name, consumer, strategys)

    def _track_tensor(
        self, tensor: Any, name: str, consumer: str, strategys: List[ToolStrategy]
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
        strategys: list<ToolStrategy>
            The strategys for the tensor.

        Returns
        -------
        tensor: Any
            The processed tensor.
        """

        if self._stage in self._plan.get(name, {}):
            return tensor
        plan = self._plan.setdefault(name, {}).setdefault(self._stage, {})
        for strategy in strategys:
            plan.update(strategy(self, tensor, name, consumer))
        return tensor

    @classmethod
    def tool_type(cls):
        return ToolType.TRACKER


class DefaultTracker(BaseTracker):
    @classmethod
    def tool_style(cls):
        return "default"


msc_utils.register_tool_cls(DefaultTracker)
