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
# pylint: disable=unused-argument
"""tvm.contrib.msc.core.gym.agent.method"""

from typing import Any, List
import numpy as np

from tvm.contrib.msc.core.runtime import BaseRunner
from tvm.contrib.msc.core.tools import BaseTool
from tvm.contrib.msc.core import utils as msc_utils


class EnvMethod(object):
    """Default prune method"""

    @classmethod
    def tasks_tool_extract(cls, env: Any, tool: BaseTool, **kwargs) -> List[dict]:
        """Extract tasks from tool

        Parameters
        ----------
        env: BaseEnv
            The evironment.
        tool: BaseTool
            The main tool
        kwargs: dict
           The kwargs for create tasks.

        Returns
        -------
        tasks: list<dict>
            The tasks for environment.
        """

        return tool.create_tasks(**kwargs)

    @classmethod
    def reward_compare_baseline(
        cls,
        env: Any,
        runner: BaseRunner,
        data_loader: callable,
        task_id: int,
        loss_type: str = "lp_norm",
        loss_config: dict = None,
    ) -> dict:
        """Reward runner with baseline

        Parameters
        ----------
        env: BaseEnv
            The evironment.
        runner: BaseRunner
            The runner.
        data_loader: callable
            The data loader.
        task_id: int
            The task id.
        loss_type: str
            The loss type
        loss_config: dict
            The loss config

        Returns
        -------
        reward: dict
            The reward.
        """

        datas_path = env._workspace.create_dir("Baseline").path
        if task_id == -1:
            with msc_utils.SimpleDataSaver(datas_path) as saver:
                for inputs in data_loader():
                    outputs = runner.run(inputs)
                    saver.save_datas(outputs)
            return {"loss": 1}

        loss_config = loss_config or {}
        loader, loss = msc_utils.SimpleDataLoader(datas_path), 0

        def _get_loss(golden, result):
            if loss_type == "lp_norm":
                power = loss_config.get("power", 2)
                return np.mean(np.power(np.abs(golden - result), power))
            raise NotImplementedError("loss type {} is not implemented".format(loss_type))

        for idx, inputs in enumerate(data_loader()):
            outputs = runner.run(inputs)
            baseline = loader[idx]
            for name, data in outputs.items():
                loss += _get_loss(baseline[name], data)
        return {"loss": loss / len(loader)}

    @classmethod
    def action_linear_space(
        cls, env: Any, task_id: int, start: float = 0.1, end: float = 0.9, step: float = 0.1
    ) -> List[float]:
        """Get linear action space

        Parameters
        ----------
        env: BaseEnv
            The evironment.
        task_id: int
            The task id.
        start: float
            The start value.
        end: float
            The end value.
        step: float
            The step value.

        Returns
        -------
        actions: list<float>
            The actions.
        """

        actions = [start]
        while actions[-1] < end:
            actions.append(actions[-1] + step)
        return actions

    @classmethod
    def action_prune_density(
        cls, env: Any, task_id: int, start: float = 0.1, end: float = 0.9, step: float = 0.1
    ) -> List[dict]:
        """Get linear density

        Parameters
        ----------
        env: BaseEnv
            The evironment.
        task_id: int
            The task id.
        start: float
            The start value.
        end: float
            The end value.
        step: float
            The step value.

        Returns
        -------
        actions: list<dict>
            The actions.
        """

        return [{"density": a} for a in cls.action_linear_space(env, task_id, start, end, step)]

    @classmethod
    def action_quantize_scale(
        cls, env: Any, task_id: int, start: float = 0.1, end: float = 0.9, step: float = 0.1
    ) -> List[dict]:
        """Get linear density

        Parameters
        ----------
        env: BaseEnv
            The evironment.
        task_id: int
            The task id.
        start: float
            The start value.
        end: float
            The end value.
        step: float
            The step value.

        Returns
        -------
        actions: list<dict>
            The actions.
        """

        task = env.get_task(task_id)
        return [
            {"scale": task["scale"] * a}
            for a in cls.action_linear_space(env, task_id, start, end, step)
        ]

    @classmethod
    def method_type(cls):
        return "env.default"


msc_utils.register_gym_method(EnvMethod)
