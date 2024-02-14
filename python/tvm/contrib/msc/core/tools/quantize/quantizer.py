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
"""tvm.contrib.msc.core.tools.quantize.quantizer"""

from typing import List, Dict, Any

from tvm.contrib.msc.core.tools.tool import ToolType, BaseTool, ToolStrategy
from tvm.contrib.msc.core import utils as msc_utils


class QuantizeStage:
    GATHER = "gather"
    CALIBRATE = "calibrate"


class BaseQuantizer(BaseTool):
    """Base quantizer for all"""

    def setup(self) -> dict:
        """Setup the tool

        Returns
        -------
        info: dict
            The setup info.
        """

        if self._plan:
            self._calibrated = True
            self.change_stage(msc_utils.MSCStage.QUANTIZE)
        else:
            self._calibrated = False
            self._calibrate_plan = {}
            self.change_stage(QuantizeStage.GATHER)
        return super().setup()

    def calibrate(self) -> dict:
        """Calibrate the datas

        Returns
        -------
        plan: dict
            The calibrated plan.
        """

        new_plan = {}
        self.change_stage(QuantizeStage.CALIBRATE)
        for tensor_id, plan in self._calibrate_plan.items():
            if plan.get("calibrated", False):
                new_plan[tensor_id] = plan
                continue
            name, consumer = self.from_tensor_id(tensor_id)
            strategy = self._get_tensor_strategy(name, consumer)
            new_plan[tensor_id] = strategy(self, name, consumer, plan)
        if any(not plan.get("calibrated", False) for plan in new_plan.values()):
            self._calibrate_plan = new_plan
            self.change_stage(QuantizeStage.GATHER)
        else:
            self._calibrated = True
            for name, plan in new_plan.items():
                self._plan[name] = {k: v for k, v in plan.items() if k not in ("calibrated")}
            self.change_stage(msc_utils.MSCStage.QUANTIZE)
        self._forward_cnt = 0
        return new_plan

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
                strategy["stages"] = [msc_utils.MSCStage.QUANTIZE]
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

        if self._calibrated:
            tensor_id = self.to_tensor_id(name, consumer)
            if tensor_id not in self._plan:
                return False
            return self._plan.get(tensor_id, {}).get("nbits", 8) != -1
        strategys = self._get_tensor_strategys(name, consumer)
        if not strategys:
            return False
        if any(s.get_config().get("nbits", 8) == -1 for s in strategys):
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

        if not self._calibrated:
            return self._gather_tensor(tensor, name, consumer, strategys)
        return self._quantize_tensor(tensor, name, consumer, strategys)

    def _gather_tensor(
        self, tensor: Any, name: str, consumer: str, strategys: List[ToolStrategy]
    ) -> Any:
        """Gather tensor datas

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

        assert len(strategys) == 1, "gather should only has 1 strategy, get " + str(strategys)
        tensor_id = self.to_tensor_id(name, consumer)
        plan = self._calibrate_plan.get(tensor_id, {})
        if plan.get("calibrated", False):
            return tensor
        self._calibrate_plan[tensor_id] = strategys[0](self, tensor, name, consumer, plan)
        return tensor

    def _quantize_tensor(
        self, tensor: Any, name: str, consumer: str, strategys: List[ToolStrategy]
    ) -> Any:
        """Quantize tensor

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

        tensor_id = self.to_tensor_id(name, consumer)
        for strategy in strategys:
            tensor = strategy(self, tensor, name, consumer, **self._plan[tensor_id])
        return tensor

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

        tasks, recorded = [], set()
        for tensor_id, plan in self._plan.items():
            name, _ = self.from_tensor_id(tensor_id)
            if self.is_weight(name) and not kwargs.get("quantize_weights", False):
                continue
            if name not in recorded:
                tasks.append({"name": tensor_id, **plan})
                if self._cache_processed:
                    recorded.add(name)
        return tasks

    @property
    def calibrated(self):
        return self._calibrated

    @classmethod
    def tool_type(cls):
        return ToolType.QUANTIZER


class DefaultQuantizer(BaseQuantizer):
    @classmethod
    def tool_style(cls):
        return "default"


msc_utils.register_tool_cls(DefaultQuantizer)
