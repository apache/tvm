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
"""tvm.contrib.msc.core.gym.search_agent"""

from typing import Any, List
from tvm.contrib.msc.core import utils as msc_utils
from .base_agent import BaseAgent


class BaseSearchAgent(BaseAgent):
    """Base Search Agent of MSC.Gym"""

    def setup(self) -> dict:
        """Setup the tool

        Returns
        -------
        info: dict
            The setup info.
        """

        self._max_search = self._options.get("max_search", -1)
        return super().setup()

    @classmethod
    def agent_type(cls):
        return "search.base"


class GridSearchAgent(BaseSearchAgent):
    """GridSearch agent"""

    def _choose_action(
        self, task_id: int, observation: Any, action_space: List[dict]
    ) -> List[dict]:
        """Choose action based on observation

        Parameters
        ----------
        task_id: int
            The current task id.
        observation:
            The current observation.
        action_space: list<dict>
            The possible action space

        Returns
        -------
        actions: list<dict>
            The actions for next task.
        """

        return action_space

    def _learn(self):
        """Learn from knowledge

        Returns
        -------
        actions: list<float>
            The learned actions.
        rewards: list<dict>
            The learned rewards.
        """

        best_actions = [None] * len(self._knowledge["actions"])
        best_rewards = [None] * len(self._knowledge["rewards"])
        idx = 0
        for actions, rewards in zip(self._knowledge["actions"], self._knowledge["rewards"]):
            best_score = None
            for action, reward in zip(actions, rewards):
                score = self._evaluate(reward)
                if best_score is None or score > best_score:
                    best_actions[idx] = action
                    best_rewards[idx] = reward
                    best_score = score
            idx += 1
        return best_actions, best_rewards

    @classmethod
    def agent_type(cls):
        return "search.grid"


class BinarySearchAgent(BaseSearchAgent):
    """BinarySearch agent"""

    def reset(self):
        """Reset the agent"""

        self._ranges = [{"start": 0, "end": -1} for _ in range(self._max_task)]
        super().reset()

    def _choose_action(
        self, task_id: int, observation: Any, action_space: List[dict]
    ) -> List[dict]:
        """Choose action based on observation

        Parameters
        ----------
        task_id: int
            The current task id.
        observation:
            The current observation.
        action_space: list<dict>
            The possible action space

        Returns
        -------
        actions: list<dict>
            The actions for next task.
        """

        if self._ranges[task_id]["end"] == -1:
            self._ranges[task_id]["end"] = len(action_space)
            return [action_space[self._ranges[task_id]["start"]]]
        pos = (self._ranges[task_id]["start"] + self._ranges[task_id]["end"]) / 2
        return [action_space[pos]]

    def _store(self, task_id: int):
        """Store rewards

        Parameters
        ----------
        task_id: int
            The current task id.

        Returns
        -------
        next_task: int
            The next task id.
        """

        rewards = self._knowledge["rewards"][task_id]
        start = self._ranges[task_id]["start"]
        end = self._ranges[task_id]["end"]
        if len(rewards) > 1:
            if self._evaluate(rewards[-1]) > self._evaluate(rewards[-2]):
                self._ranges[task_id]["end"] = (start + end) // 2
            else:
                self._ranges[task_id]["start"] = (start + end) // 2
        if start - end <= 1:
            return task_id + 1
        return task_id

    def _learn(self):
        """Learn from knowledge

        Returns
        -------
        actions: list<float>
            The learned actions.
        rewards: list<dict>
            The learned rewards.
        """

        actions = [a[-1] for a in self._knowledge["actions"]]
        rewards = [r[-1] for r in self._knowledge["rewards"]]
        return actions, rewards

    @classmethod
    def agent_type(cls):
        return "search.binary"


msc_utils.register_gym_agent(GridSearchAgent)
