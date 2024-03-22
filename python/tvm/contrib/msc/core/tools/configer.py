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
"""tvm.contrib.msc.core.tools.configer"""

from typing import Union
from tvm.contrib.msc.core import utils as msc_utils
from .tool import ToolType


class ToolConfiger(object):
    """Base configer for tool"""

    def config(self, raw_config: dict = None) -> dict:
        """Get the config

        Parameters
        ----------
        raw_config: dict
            The raw config.

        Returns
        -------
        config: dict
            The update config.
        """

        config = {}
        if isinstance(raw_config, dict) and "gym_configs" in raw_config:
            config["gym_configs"] = [self.config_gym(g) for g in raw_config.pop("gym_configs")]
        if raw_config:
            config["tool_config"] = self.update_tool(raw_config)
        else:
            config["tool_config"] = self.config_tool()
        config.update(self.config_apply())
        return config

    def config_tool(self) -> dict:
        """Get the default config of tool

        Returns
        -------
        config: dict
            The default config.
        """

        raise NotImplementedError("config_tool is not implemented in ToolConfiger")

    def update_tool(self, raw_config: dict) -> dict:
        """Update tool config from raw_config

        Parameters
        ----------
        raw_config: dict
            The raw config.

        Returns
        -------
        config: dict
            The update config.
        """

        config = self.config_tool()
        return msc_utils.update_dict(config, raw_config)

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

        raise NotImplementedError("config_gym is not implemented in ToolConfiger")

    def config_apply(self) -> dict:
        """Get the config for apply

        Returns
        -------
        config: dict
            The apply config.
        """

        return {}

    @classmethod
    def tool_type(cls):
        return ToolType.BASE
