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
"""tvm.contrib.msc.core.tools.prune.configer"""

from typing import Union
from tvm.contrib.msc.core.tools.tool import ToolType
from tvm.contrib.msc.core.tools.configer import ToolConfiger
from tvm.contrib.msc.core import utils as msc_utils


class PruneConfiger(ToolConfiger):
    """Configer for prune"""

    def config_gym(self, raw_config: Union[dict, str]) -> dict:
        """Config the gym

        Parameters
        ----------
        gym_config: dict
            The raw config.

        Returns
        -------
        gym_config: dict
            The update config.
        """

        if isinstance(raw_config, dict):
            return raw_config
        if raw_config == "default":
            return {
                "env": {
                    "executors": {
                        "action_space": {
                            "method": "action_prune_density",
                            "start": 0.2,
                            "end": 0.8,
                            "step": 0.1,
                        }
                    },
                },
                "agent": {"role_type": "search.grid", "executors": {}},
            }
        else:
            raise TypeError("Unexpected gym config " + str(raw_config))

    @classmethod
    def tool_type(cls):
        return ToolType.PRUNER


@msc_utils.register_tool_configer
class DefaultPruneConfiger(PruneConfiger):
    """Default configer for prune"""

    def config_tool(self) -> dict:
        """Get the default config of tool

        Returns
        -------
        config: dict
            The default config.
        """

        return {
            "plan_file": "msc_pruner.json",
            "strategys": [
                {
                    "methods": {
                        "weights": {"method_name": "per_channel", "density": 0.8},
                        "output": {"method_name": "per_channel", "density": 0.8},
                    }
                }
            ],
        }

    @classmethod
    def config_style(cls):
        return "default"
