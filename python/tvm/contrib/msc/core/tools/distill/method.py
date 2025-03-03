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
"""tvm.contrib.msc.core.tools.distill.method"""

from typing import List
import numpy as np

from tvm.contrib.msc.core.tools.tool import ToolType, BaseTool
from tvm.contrib.msc.core.utils.namespace import MSCFramework
from tvm.contrib.msc.core import utils as msc_utils


@msc_utils.register_tool_method
class DistillMethod(object):
    """Default distill method"""

    @classmethod
    def loss_lp_norm(
        cls,
        distiller: BaseTool,
        t_outputs: List[np.ndarray],
        s_outputs: List[np.ndarray],
        power: int = 2,
    ):
        """Calculate loss with mse

        Parameters
        ----------
        distiller: BaseDistiller
            The distiller
        t_outputs: list<np.ndarray>
            The teacher outputs.
        s_outputs: list<np.ndarray>
            The student outputs.
        power: int
            The power factor.

        Returns
        -------
        loss: float
            The loss.
        """

        loss = 0
        for t_out, s_out in zip(t_outputs, s_outputs):
            loss += np.mean(np.power(np.abs(t_out - s_out), power))
        return loss

    @classmethod
    def framework(cls):
        return MSCFramework.MSC

    @classmethod
    def tool_type(cls):
        return ToolType.DISTILLER

    @classmethod
    def method_style(cls):
        return "default"
