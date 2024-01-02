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
"""tvm.contrib.msc.core.gym.configer"""

from tvm.contrib.msc.core import utils as msc_utils


class BaseConfiger(object):
    """Configer for Gym

    Parameters
    ----------
    stage: str
        The stage for gym, should be in MSCStage.
    """

    def __init__(self, stage: str):
        self._stage = stage

    def update(self, raw_config: dict) -> dict:
        """Config the raw config

        Parameters
        ----------
        raw_config: dict
            The raw config.

        Returns
        -------
        config: dict
            The update config.
        """

        raise NotImplementedError("update is not implemented in BaseConfiger")


class DefaultConfiger(BaseConfiger):
    """Default configer for gym"""

    def update(self, raw_config: dict) -> dict:
        """Config the raw config

        Parameters
        ----------
        raw_config: dict
            The raw config.

        Returns
        -------
        config: dict
            The update config.
        """

        config = msc_utils.copy_dict(raw_config)
        assert "env" in config and "agent" in config, "env and agent should be given to run gym"
        if "env_type" not in config["env"]:
            config["env"]["env_type"] = self._stage + ".default"
        if "agent_type" not in config["agent"]:
            config["agent"]["agent_type"] = "search.grid"
        if "executors" not in config["env"]:
            config["env"]["executors"] = {}
        # update executors
        env_executors = {
            "reward_runner": {"method": "reward_compare_baseline"},
            "create_tasks": {"method": "tasks_tool_extract"},
        }
        config["env"]["executors"].update(
            {k: v for k, v in env_executors.items() if k not in config["env"]["executors"]}
        )
        if "executors" not in config["agent"]:
            config["agent"]["executors"] = {}
        agent_executors = {"evaluate": {"method": "evaluate_by_loss"}}
        config["agent"]["executors"].update(
            {k: v for k, v in agent_executors.items() if k not in config["agent"]["executors"]}
        )
        return config

    @classmethod
    def config_type(cls):
        return "default"


msc_utils.register_gym_configer(DefaultConfiger)
