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
"""tvm.contrib.msc.core.tools.distill.configer"""

from tvm.contrib.msc.core.tools.tool import ToolType
from tvm.contrib.msc.core.tools.configer import ToolConfiger
from tvm.contrib.msc.core import utils as msc_utils


class DistillConfiger(ToolConfiger):
    """Configer for distill"""

    @classmethod
    def tool_type(cls):
        return ToolType.DISTILLER


@msc_utils.register_tool_configer
class DefaultDistillConfiger(DistillConfiger):
    """Default configer for distill"""

    def config_tool(self) -> dict:
        """Get the default config of tool

        Returns
        -------
        config: dict
            The default config.
        """

        return {
            "plan_file": "msc_distiller.json",
            "strategys": [
                {
                    "methods": {"mark": "loss_lp_norm"},
                    "marks": ["loss"],
                },
            ],
        }

    @classmethod
    def config_style(cls):
        return "default"
