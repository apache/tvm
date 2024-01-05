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
"""tvm.contrib.msc.core.gym.base_agent"""

import copy
import logging
from typing import Dict, Any, List, Tuple
from tvm.contrib.msc.core import utils as msc_utils


class BaseAgent(object):
    """Basic Agent of MSC.Gym

    Parameters
    ----------
    name: str
        The name of agent.
    workspace: MSCDirectory
        The worksapce.
    executors: dict
        The executors of the agent.
    options: dict
        The extra options for the agent.
    debug_level: int
        The debug level.
    verbose: str
        The verbose level.
    logger: logging.Logger
        The logger
    """

    def __init__(
        self,
        name: str,
        workspace: msc_utils.MSCDirectory,
        executors: dict,
        options: dict = None,
        debug_level: int = 0,
        verbose: str = None,
        logger: logging.Logger = None,
    ):
        self._name = name
        self._workspace = workspace
        self._executors = self._parse_executors(msc_utils.copy_dict(executors))
        self._options = options or {}
        self._debug_level = debug_level
        if logger:
            self._logger = logger
        else:
            if not verbose:
                verbose = "debug" if debug_level > 0 else "info"
            self._logger = msc_utils.create_file_logger(verbose, workspace.relpath("AGENT_LOG"))
        self._logger.info(
            msc_utils.msg_block("AGENT.SETUP({})".format(self.agent_type()), self.setup())
        )

    def _parse_executors(self, executors_dict: dict) -> Dict[str, Tuple[callable, dict]]:
        """Parse the executors

        Parameters
        ----------
        executors_dict: dict
            The given executors.

        Returns
        -------
        executors_dict: dict
            The parsed executors.
        """

        executors = {}
        for name, raw_config in executors_dict.items():
            method_type = (
                raw_config.pop("method_type") if "method_type" in raw_config else "agent.default"
            )
            method_cls = msc_utils.get_registered_gym_method(method_type)
            assert "method" in raw_config, "method should be given to find agent method"
            method_name, method = raw_config.pop("method"), None
            if hasattr(method_cls, method_name):
                method = getattr(method_cls, method_name)
            if not method:
                method = msc_utils.get_registered_func(method_name)
            assert method, "Can not find method " + str(method_name)
            executors[name] = (method_name, method, copy.deepcopy(raw_config))
        return executors

    def setup(self) -> dict:
        """Setup the agent

        Returns
        -------
        info: dict
            The setup info.
        """

        self._knowledge = {"observations": [], "actions": [], "rewards": []}
        return {
            "name": self._name,
            "workspace": self._workspace,
            "executors": {k: "{}({})".format(v[0], v[2]) for k, v in self._executors.items()},
            "options": self._options,
            "debug_level": self._debug_level,
        }

    def init(self, max_task: int, baseline: Dict[str, Any]):
        """Init the agent

        Parameters
        ----------
        max_task: int
            The max task for agent.
        baseline: dict
            The baseline of environment.
        """

        self._max_task = max_task
        self._baseline = baseline

    def reset(self):
        """Reset the agent"""

        self._knowledge = {"observations": [], "actions": [], "rewards": []}

    def choose_action(self, task_id: int, observation: Any, action_space: List[dict]) -> List[dict]:
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

        actions = self._choose_action(task_id, observation, action_space)
        if task_id == len(self._knowledge["observations"]):
            self._knowledge["observations"].append(observation)
            self._knowledge["actions"].append(actions)
        elif task_id == len(self._knowledge["observations"]) - 1:
            self._knowledge["actions"][-1].extend(actions)
        else:
            raise TypeError(
                "Step id should be either {0} or {0}-1, get {1}".format(
                    len(self._knowledge["observations"]), task_id
                )
            )
        return actions

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

        raise NotImplementedError("_choose_action is not implemented in BaseAgent")

    def store(self, task_id: int, rewards: List[dict]) -> int:
        """Store rewards

        Parameters
        ----------
        task_id: int
            The current task id.
        rewards: list<dict>
            The rewards for each action

        Returns
        -------
        next_task: int
            The next task id.
        """

        if task_id == len(self._knowledge["rewards"]):
            self._knowledge["rewards"].append(rewards)
        elif task_id == len(self._knowledge["rewards"]) - 1:
            self._knowledge["rewards"][-1].extend(rewards)
        else:
            raise TypeError(
                "Step id should be either {0} or {0}-1, get {1}".format(
                    len(self._knowledge["rewards"]), task_id
                )
            )
        return self._store(task_id)

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

        return task_id + 1

    def learn(self):
        """Learn from knowledge

        Returns
        -------
        actions: list<dict>
            The learned actions.
        rewards: list<dict>
            The learned rewards.
        """

        self._logger.debug(msc_utils.msg_block("AGENT.LEARN", self._knowledge))
        return self._learn()

    def _learn(self):
        """Learn from knowledge

        Returns
        -------
        actions: list<dict>
            The learned actions.
        rewards: list<dict>
            The learned rewards.
        """

        raise NotImplementedError("_learn is not implemented in BaseAgent")

    def destory(self):
        """Destory the agent"""

        return None

    def _execute(self, name: str, *args, **kwargs) -> Any:
        """Run executor

        Parameters
        ----------
        name: str
            The executor name.
        args: list<Any>
            The arguments for execute.
        kwargs: dict<Any>
            The key word arguments for execute.

        Returns
        -------
        res:
            The execute result.
        """

        assert name in self._executors, "Can not find {} in executors: {}".format(
            name, self._executors.keys()
        )
        _, method, config = self._executors[name]
        kwargs.update({k: v for k, v in config.items() if k not in kwargs})
        return method(self, *args, **kwargs)

    def _evaluate(self, reward: dict) -> float:
        """Evaluate a reward with baseline

        Parameters
        ----------
        reward: dict
            The reward for.

        Returns
        -------
        score: float
            The score of the reward.
        """

        return self._execute("evaluate", self._baseline, reward)

    @classmethod
    def agent_type(cls):
        return "base"


msc_utils.register_gym_agent(BaseAgent)
