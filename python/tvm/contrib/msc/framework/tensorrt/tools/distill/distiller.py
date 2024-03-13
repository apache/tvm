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
"""tvm.contrib.msc.framework.tensorrt.tools.distill.distiller"""

from tvm.contrib.msc.core.tools.tool import ToolType
from tvm.contrib.msc.core.tools.distill import BaseDistiller
from tvm.contrib.msc.core.utils.namespace import MSCFramework
from tvm.contrib.msc.core import utils as msc_utils


class TensorRTDistillerFactory(object):
    """Distiller factory for tensorrt"""

    def create(self, base_cls: BaseDistiller) -> BaseDistiller:
        """Create adaptive distiller

        Parameters
        ----------
        base_cls: BaseDistiller
            The base distiller class

        Returns
        -------
        distiller_cls: BaseDistiller
            The distiller class.
        """

        class Distiller(base_cls):
            """Adaptive distiller for tensorrt"""

            @classmethod
            def framework(cls):
                return MSCFramework.TENSORRT

        return Distiller


factory = TensorRTDistillerFactory()
tools = msc_utils.get_registered_tool_cls(MSCFramework.MSC, ToolType.DISTILLER, tool_style="all")
for tool in tools.values():
    msc_utils.register_tool_cls(factory.create(tool))
