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

from typing import Any
from tvm.contrib.msc.core import utils as msc_utils


class AgentMethod(object):
    """Default prune method"""

    @classmethod
    def evaluate_by_loss(cls, agent: Any, baseline: dict, reward: dict) -> float:
        """Evaluate the raw loss

        Parameters
        ----------
        agent: BaseAgent
            The base agent.
        baseline: dict
            The baseline.
        reward: dict
            The reward.

        Returns
        -------
        score: float
            The score.
        """

        assert "loss" in reward, "loss should be given to evaluate loss"
        return 1 / reward["loss"]

    @classmethod
    def evaluate_by_thresh(cls, agent: Any, baseline: dict, reward: dict, thresh: float) -> float:
        """Evaluate the raw loss

        Parameters
        ----------
        agent: BaseAgent
            The base agent.
        baseline: dict
            The baseline.
        reward: dict
            The reward.
        thresh: float
            The threshold

        Returns
        -------
        score: float
            The score.
        """

        assert "reward" in reward, "reward should be given to evaluate threshold"
        if reward["reward"] >= thresh:
            return thresh
        return reward["reward"]

    @classmethod
    def method_type(cls):
        return "agent.default"


msc_utils.register_gym_method(AgentMethod)
