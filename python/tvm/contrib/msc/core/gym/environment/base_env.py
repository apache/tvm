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
"""tvm.contrib.msc.core.gym.base_env"""

import copy
import logging
from typing import Dict, Any, List, Tuple
from tvm.contrib.msc.core.runtime import BaseRunner
from tvm.contrib.msc.core.tools import BaseTool
from tvm.contrib.msc.core import utils as msc_utils


class BaseEnv(object):
    """Basic Environment of MSC.Gym

    Parameters
    ----------
    runner: BaseRunner
        The runner.
    data_loader:
        The data_loader
    workspace: MSCDirectory
        The worksapce.
    executors: dict
        The executors of the environment.
    knowledge: dict
        The predefined knowledge.
    options: dict
        The extra options for the environment.
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
        runner: BaseRunner,
        data_loader: Any,
        workspace: msc_utils.MSCDirectory,
        executors: dict,
        knowledge: dict = None,
        options: dict = None,
        max_tasks: int = -1,
        debug_level: int = 0,
        verbose: str = None,
        logger: logging.Logger = None,
    ):
        self._name = name
        self._runner = runner
        self._data_loader = data_loader
        self._workspace = workspace
        self._knowledge = knowledge
        self._executors = self._parse_executors(msc_utils.copy_dict(executors))
        self._options = options or {}
        self._max_tasks = max_tasks
        self._debug_level = debug_level
        if logger:
            self._logger = logger
        else:
            if not verbose:
                verbose = "debug" if debug_level > 0 else "info"
            self._logger = msc_utils.create_file_logger(verbose, workspace.relpath("ENV_LOG"))
        self._logger.info(
            msc_utils.msg_block("ENV.SETUP({})".format(self.env_type()), self.setup())
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
                raw_config.pop("method_type") if "method_type" in raw_config else "env.default"
            )
            method_cls = msc_utils.get_registered_gym_method(method_type)
            assert "method" in raw_config, "method should be given to find enviironment method"
            method_name, method = raw_config.pop("method"), None
            if hasattr(method_cls, method_name):
                method = getattr(method_cls, method_name)
            if not method:
                method = msc_utils.get_registered_func(method_name)
            assert method, "Can not find method " + str(method_name)
            executors[name] = (method_name, method, copy.deepcopy(raw_config))
        return executors

    def setup(self) -> dict:
        """Setup the environment

        Returns
        -------
        info: dict
            The setup info.
        """

        self._cache_dir = self._workspace.create_dir("Cache")
        self._tasks = []
        return {
            "name": self._name,
            "runner": self._runner,
            "data_loader": self._data_loader,
            "workspace": self._workspace,
            "executors": {k: "{}({})".format(v[0], v[2]) for k, v in self._executors.items()},
            "options": self._options,
            "max_tasks": self._max_tasks,
            "debug_level": self._debug_level,
        }

    def init(self) -> Tuple[int, Dict[str, Any]]:
        """Init the agent

        Returns
        -------
        max_tasks: int
            The max task for agent.
        baseline: dict
            The baseline of environment.
        """

        self._runner.change_logger(self._logger)
        # save cache for tasks
        self._runner.save_cache(self._cache_dir)
        self._tool = self._init_tool()
        # create tasks
        self._tasks = self._execute("create_tasks", self._tool)
        if self._max_tasks > 0:
            self._tasks = self._tasks[: self._max_tasks]
        # get baseline
        self._tool.disable()
        self._runner.build(self._cache_dir, force_build=True)
        baseline = self._reward_runner(-1)
        self._tool.enable()
        tasks_info = {"tasks_num": len(self._tasks), "tasks": self._tasks}
        self._logger.info(msc_utils.msg_block("ENV.TASKS", tasks_info, width=0))
        return len(self._tasks), baseline

    def _init_tool(self) -> BaseTool:
        """Get the main tool"""

        raise NotImplementedError("_init_tool is not implemented in BaseEnv")

    def reset(self) -> Tuple[List[float], List[dict]]:
        """Reset the environment

        Returns
        -------
        observation: list<float>
            The next observation.
        action_space: list<dict>
            The next action space.
        """

        return None

    def get_state(self, task_id: int) -> Tuple[List[float], List[dict]]:
        """Get the state

        Parameters
        ----------
        task_id: int
            The current task id.

        Returns
        -------
        observation: list<float>
            The next observation.
        action_space: list<dict>
            The next action space.
        """

        if "observation" in self._executors:
            observation = self._execute("observation", task_id)
        else:
            observation = [task_id]
        if "action_space" in self._executors:
            action_space = self._execute("action_space", task_id)
        else:
            action_space = list(range(5))
        return observation, action_space

    def step(self, actions: List[dict], task_id: int) -> Tuple[List[float], List[dict], List[dict]]:
        """Step and get rewards

        Parameters
        ----------
        actions: list<dict>
            The current actions.
        task_id: int
            The current task id.

        Returns
        -------
        observation: list<float>
            The next observation.
        action_space: list<dict>
            The next action space.
        rewards: list<dict>
            The rewards
        """

        rewards = []
        for idx, action in enumerate(actions):
            self._update_tool(action, task_id)
            self._runner.build(self._cache_dir, force_build=True)
            rewards.append(self._reward_runner(task_id))
            self._logger.info(
                "Task[%d/%d] Action[%d/%d] %s -> reward %s",
                task_id,
                len(self._tasks),
                idx,
                len(actions),
                action,
                rewards[-1],
            )
        return rewards

    def _update_tool(self, action: dict, task_id: int):
        """Update the tool

        Parameters
        ----------
        action: dict
            The current action.
        task_id: int
            The current task id.
        """

        raise NotImplementedError("_update_tool is not implemented in BaseEnv")

    def summary(self, actions: List[dict], rewards: List[dict]) -> dict:
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

        self._logger.info("Env Summary with %d actions, %d rewards", len(actions), len(rewards))
        return self._summary(actions, rewards)

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

        raise NotImplementedError("_summary is not implemented in BaseEnv")

    def get_task(self, task_id: int) -> dict:
        """Get task according to task_id

        Parameters
        ----------
        task_id: int
            The task id.

        Returns
        -------
        task_config: dict
            The task config.
        """

        return self._tasks[task_id]

    def destory(self):
        """Destory the environment"""

        return None

    def _reward_runner(self, task_id: int) -> dict:
        """Reward runner for current task

        Parameters
        ----------
        task_id: int
            The current task id.

        Returns
        -------
        reward: dict
            The reward
        """

        if "reward_runner" in self._executors:
            return self._execute("reward_runner", self._runner, self._data_loader, task_id)
        elif "reward_outputs" in self._executors:
            reward = {}
            for inputs in self._data_loader():
                outputs = self._runner.run(inputs)
                reward = self._execute("reward_outputs", reward, outputs, task_id)
            return reward
        else:
            raise Exception("reward_runner or reward_outputs should be given in executors")

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

    @classmethod
    def env_type(cls):
        return "base"
