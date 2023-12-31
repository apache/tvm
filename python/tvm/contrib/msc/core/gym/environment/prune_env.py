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
"""tvm.contrib.msc.core.gym.prune_env"""

from typing import List
from tvm.contrib.msc.core.tools import BaseTool, ToolType
from tvm.contrib.msc.core import utils as msc_utils
from .base_env import BaseEnv


class PruneEnv(BaseEnv):
    """Environment for prune"""

    def _init_tool(self) -> BaseTool:
        """Get the main tool"""

        config = self._runner.get_tool_config(ToolType.PRUNER)
        self._meta_strategys = config["strategys"]
        for s in self._meta_strategys:
            s.update({"density": 1})
        return self._runner.get_tool(ToolType.PRUNER)

    def _update_tool(self, action: dict, task_id: int):
        """Update the tool

        Parameters
        ----------
        action: dict
            The current action.
        task_id: int
            The current task id.
        """

        task_strategy = self._get_strategy(action, task_id)
        self._tool.plan_by_strategys(self._meta_strategys + [task_strategy])

    def _summary(self, actions: List[dict], rewards: List[dict]) -> dict:
        """Summary the final plan

        Parameters
        ----------
        actions: list<dict>
            The final actions.
        rewards: list<dict>
            The final rewards.

        Returns
        -------
        plan: dict
            The final plan.
        """

        strategys = [self._get_strategy(act, idx) for idx, act in enumerate(actions)]
        return self._tool.plan_by_strategys(self._meta_strategys + strategys)

    def _get_strategy(self, action: dict, task_id: int) -> dict:
        """Get strategy from task_id

        Parameters
        ----------
        action: float
            The current action.
        task_id: int
            The current task id.

        Returns
        -------
        strategy: dict
            The strategy.
        """

        strategy = msc_utils.copy_dict(self.get_task(task_id))
        strategy.update(**action)
        return strategy

    @classmethod
    def env_type(cls):
        return msc_utils.MSCStage.PRUNE + ".default"


msc_utils.register_gym_env(PruneEnv)
