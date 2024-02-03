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
"""tvm.contrib.msc.framework.tensorflow.tools.quantize.quantizer"""

from tvm.contrib.msc.core.tools.tool import ToolType
from tvm.contrib.msc.core.tools.quantize import BaseQuantizer
from tvm.contrib.msc.core.utils.namespace import MSCFramework
from tvm.contrib.msc.core import utils as msc_utils


class TensorflowQuantizerFactory(object):
    """Quantizer factory for tensorflow"""

    def create(self, base_cls: BaseQuantizer) -> BaseQuantizer:
        """Create adaptive quantizer

        Parameters
        ----------
        base_cls: BaseQuantizer
            The base quantizer class

        Returns
        -------
        quantizer_cls: BaseQuantizer
            The quantizer class.
        """

        class Quantizer(base_cls):
            """Adaptive quantizer for tensorflow"""

            @classmethod
            def framework(cls):
                return MSCFramework.TENSORFLOW

        return Quantizer


factory = TensorflowQuantizerFactory()
tools = msc_utils.get_registered_tool_cls(MSCFramework.MSC, ToolType.QUANTIZER, tool_style="all")
for tool in tools.values():
    msc_utils.register_tool_cls(factory.create(tool))
