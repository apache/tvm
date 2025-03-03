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
"""tvm.contrib.msc.core.gym.quantize_env"""

from typing import List, Union
from tvm.contrib.msc.core.tools import BaseTool, ToolType
from tvm.contrib.msc.core import utils as msc_utils
from .base_env import BaseEnv


@msc_utils.register_gym_object
class QuantizeEnv(BaseEnv):
    """Environment for quantize"""

    def _init_tool(self) -> BaseTool:
        """Get the main tool"""

        self._runner.make_plan(ToolType.QUANTIZER, self._data_loader)
        return self._runner.get_tool(ToolType.QUANTIZER)

    def _update_tool(self, action: dict, task_id: int):
        """Update the tool

        Parameters
        ----------
        action: dict
            The current action.
        task_id: int
            The current task id.
        """

        self._tool.change_strategys([self._get_strategy(action, task_id)])

    def _summary(self, actions: List[dict], rewards: List[dict]) -> Union[dict, str]:
        """Summary the final plan

        Parameters
        ----------
        actions: list<dict>
            The final actions.
        rewards: list<dict>
            The final rewards.

        Returns
        -------
        knowledge: dict| str
            The learned knowledge or file.
        """

        strategys = self.tool._parse_strategys(
            [self._get_strategy(act, idx) for idx, act in enumerate(actions)]
        )
        plan = self.tool.plan
        for name, info in plan.items():
            if name not in strategys:
                continue
            info.update(strategys[name].get_executor(msc_utils.MSCStage.QUANTIZE).config)
        summary_file = msc_utils.get_cache_dir().relpath("gym_summary.json")
        return msc_utils.save_dict(plan, summary_file)

    @classmethod
    def role_type(cls):
        return msc_utils.MSCStage.QUANTIZE + ".default"
