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

import os
from typing import List
from tvm.contrib.msc.core.tools import BaseTool, ToolType
from tvm.contrib.msc.core import utils as msc_utils
from .base_env import BaseEnv


class QuantizeEnv(BaseEnv):
    """Environment for quantize"""

    def _init_tool(self) -> BaseTool:
        """Get the main tool"""

        plan_file = self._runner.apply_tool(ToolType.QUANTIZER, self._data_loader)
        self._meta_plan = msc_utils.load_dict(plan_file)
        os.remove(plan_file)
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

        plan = msc_utils.copy_dict(self._meta_plan)
        plan.update(self._get_plan(action, task_id))
        self._tool.set_plan(plan)

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

        plan = msc_utils.copy_dict(self._meta_plan)
        for idx, act in enumerate(actions):
            plan.update(self._get_plan(act, idx))
        return plan

    def _get_plan(self, action: dict, task_id: int) -> dict:
        """Get plan from task_id

        Parameters
        ----------
        action: float
            The current action.
        task_id: int
            The current task id.

        Returns
        -------
        plan: dict
            The plan.
        """

        plan = msc_utils.copy_dict(self.get_task(task_id))
        plan.update(**action)
        name = plan.pop("name")
        return {name: plan}

    @classmethod
    def env_type(cls):
        return msc_utils.MSCStage.QUANTIZE + ".default"


msc_utils.register_gym_env(QuantizeEnv)
