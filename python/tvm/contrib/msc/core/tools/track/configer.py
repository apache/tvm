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
"""tvm.contrib.msc.core.tools.track.configer"""

from tvm.contrib.msc.core.tools.tool import ToolType
from tvm.contrib.msc.core.tools.configer import ToolConfiger
from tvm.contrib.msc.core.utils import MSCStage
from tvm.contrib.msc.core import utils as msc_utils


class TrackConfiger(ToolConfiger):
    """Configer for track"""

    @classmethod
    def tool_type(cls):
        return ToolType.TRACKER


@msc_utils.register_tool_configer
class DefaultTrackConfiger(TrackConfiger):
    """Default configer for track"""

    def config_tool(self) -> dict:
        """Get the default config of tool

        Returns
        -------
        config: dict
            The default config.
        """

        return {
            "plan_file": "msc_tracker.json",
            "strategys": [
                {
                    "methods": {
                        "output": {
                            "method_name": "save_compared",
                            "compare_to": {
                                MSCStage.OPTIMIZE: [MSCStage.BASELINE],
                                MSCStage.COMPILE: [MSCStage.OPTIMIZE, MSCStage.BASELINE],
                            },
                        }
                    },
                    "op_types": ["nn.relu"],
                }
            ],
        }

    @classmethod
    def config_style(cls):
        return "default"
