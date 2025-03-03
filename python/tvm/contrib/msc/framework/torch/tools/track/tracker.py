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
"""tvm.contrib.msc.framework.torch.tools.track.tracker"""

from tvm.contrib.msc.core.tools.tool import ToolType
from tvm.contrib.msc.core.tools.track import BaseTracker
from tvm.contrib.msc.core.utils.namespace import MSCFramework
from tvm.contrib.msc.core import utils as msc_utils


class TorchTrackerFactory(object):
    """Tracker factory for torch"""

    def create(self, base_cls: BaseTracker) -> BaseTracker:
        """Create adaptive tracker

        Parameters
        ----------
        base_cls: BaseTracker
            The base tracker class

        Returns
        -------
        tracker_cls: BaseTracker
            The tracker class.
        """

        @msc_utils.register_tool
        class Tracker(base_cls):
            """Adaptive tracker for torch"""

            @classmethod
            def framework(cls):
                return MSCFramework.TORCH

        return Tracker


factory = TorchTrackerFactory()
tools = msc_utils.get_registered_tool(MSCFramework.MSC, ToolType.TRACKER, tool_style="all")
for tool in tools.values():
    factory.create(tool)
