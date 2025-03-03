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

from typing import List, Union
from tvm.contrib.msc.core.tools import BaseTool, ToolType
from tvm.contrib.msc.core import utils as msc_utils
from .base_env import BaseEnv


@msc_utils.register_gym_object
class PruneEnv(BaseEnv):
    """Environment for prune"""

    def _init_tool(self) -> BaseTool:
        """Get the main tool"""

        config = self._runner.get_tool_config(ToolType.PRUNER)
        self._meta_strategys = msc_utils.copy_dict(config["strategys"])
        self._meta_strategys = [self._update_strategy(s, density=1) for s in self._meta_strategys]
        tool = self._runner.get_tool(ToolType.PRUNER)
        tool.change_strategys(self._meta_strategys)
        return tool

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
        self._apply_strategys(self._meta_strategys + [task_strategy])

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

        strategys = self._meta_strategys + [
            self._get_strategy(act, idx) for idx, act in enumerate(actions)
        ]
        return self._apply_strategys(strategys)

    def _apply_strategys(self, strategys: List[dict]) -> str:
        """Apply the strategys

        Parameters
        ----------
        strategys: list<dict>
            The given strategys

        Returns
        -------
        plan_file: str
            The plan after strategys applied.
        """

        self._tool.change_strategys(strategys)
        self._runner.build(self._cache_dir, force_build=True)
        return self._runner.make_plan(self._tool.tool_type(), self._data_loader)

    @classmethod
    def role_type(cls):
        return msc_utils.MSCStage.PRUNE + ".default"
