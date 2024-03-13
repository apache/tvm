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
"""tvm.contrib.msc.framework.torch.tools.distill.method"""

from typing import List

import torch
from tvm.contrib.msc.core.tools.distill import DistillMethod, BaseDistiller
from tvm.contrib.msc.core.utils.namespace import MSCFramework
from tvm.contrib.msc.core import utils as msc_utils


@msc_utils.register_tool_method
class TorchDistillMethod(DistillMethod):
    """Default quantize method for torch"""

    @classmethod
    def loss_kl_divergence(
        cls,
        distiller: BaseDistiller,
        t_outputs: List[torch.Tensor],
        s_outputs: List[torch.Tensor],
        temperature: int = 5,
        softmax_dim: int = -1,
    ):
        """Calculate loss with mse

        Parameters
        ----------
        distiller: BaseDistiller
            The distiller
        t_outputs: list<torch.Tensor>
            The teacher outputs.
        s_outputs: list<torch.Tensor>
            The student outputs.
        temperature: int
            The temperature factor.
        softmax_dim: int
            If >=0, use softmax_dim for softmax loss

        Returns
        -------
        loss: float
            The loss.
        """

        kd_loss, loss = torch.nn.KLDivLoss(), 0
        if softmax_dim >= 0:
            log_softmax = torch.nn.LogSoftmax(dim=softmax_dim)
            softmax = torch.nn.Softmax(dim=softmax_dim)

        def _distill_loss(t_out, s_out):
            if softmax_dim >= 0:
                return (
                    temperature
                    * temperature
                    * kd_loss(log_softmax(s_out / temperature), softmax(t_out / temperature))
                )
            return kd_loss(s_out / temperature, t_out / temperature)

        for t_out, s_out in zip(t_outputs, s_outputs):
            loss += _distill_loss(t_out, s_out)
        return loss

    @classmethod
    def loss_lp_norm(
        cls,
        distiller: BaseDistiller,
        t_outputs: List[torch.Tensor],
        s_outputs: List[torch.Tensor],
        power: int = 2,
    ):
        """Calculate loss with mse

        Parameters
        ----------
        distiller: BaseDistiller
            The distiller
        t_outputs: list<torch.Tensor>
            The teacher outputs.
        s_outputs: list<torch.Tensor>
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
            loss += torch.pow((t_out - s_out).abs(), power).mean()
        return loss

    @classmethod
    def framework(cls):
        return MSCFramework.TORCH
