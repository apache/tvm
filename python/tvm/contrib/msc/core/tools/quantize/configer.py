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
"""tvm.contrib.msc.core.tools.quantize.configer"""

from typing import Union

from tvm.contrib.msc.core.tools.tool import ToolType
from tvm.contrib.msc.core.tools.configer import ToolConfiger
from tvm.contrib.msc.core import utils as msc_utils
from .quantizer import QuantizeStage


class QuantizeConfiger(ToolConfiger):
    """Configer for quantize"""

    def config_gym(self, gym_config: Union[dict, str]) -> dict:
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

        if isinstance(gym_config, dict):
            return gym_config
        if gym_config == "default":
            return {
                "env": {
                    "executors": {
                        "action_space": {
                            "method": "action_quantize_scale",
                            "start": 0.8,
                            "end": 1.2,
                            "step": 0.1,
                        }
                    },
                },
                "agent": {"agent_type": "search.grid", "executors": {}},
            }
        else:
            raise TypeError("Unexpected gym config " + str(gym_config))

    @classmethod
    def tool_type(cls):
        return ToolType.QUANTIZER


@msc_utils.register_tool_configer
class DefaultQuantizeConfiger(QuantizeConfiger):
    """Default configer for quantize"""

    def config_tool(self) -> dict:
        """Get the default config of tool

        Returns
        -------
        config: dict
            The default config.
        """

        op_types = [
            "nn.conv1d",
            "msc.conv1d_bias",
            "nn.conv2d",
            "msc.conv2d_bias",
            "nn.conv3d",
            "msc.conv3d_bias",
            "msc.linear",
            "msc.linear_bias",
            "nn.avg_pool1d",
            "nn.avg_pool2d",
            "nn.avg_pool3d",
        ]

        return {
            "plan_file": "msc_quantizer.json",
            "strategys": [
                {
                    "methods": {
                        "input": "gather_maxmin",
                        "output": "gather_maxmin",
                        "weights": "gather_max_per_channel",
                    },
                    "op_types": op_types,
                    "stages": [QuantizeStage.GATHER],
                },
                {
                    "methods": {"input": "calibrate_maxmin", "output": "calibrate_maxmin"},
                    "op_types": op_types,
                    "stages": [QuantizeStage.CALIBRATE],
                },
                {
                    "methods": {
                        "input": "quantize_normal",
                        "weights": "quantize_normal",
                        "output": "dequantize_normal",
                    },
                    "op_types": op_types,
                },
            ],
        }

    @classmethod
    def config_style(cls):
        return "default"
