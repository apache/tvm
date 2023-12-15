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
"""tvm.contrib.msc.framework.torch.tools.quantize.method"""

import numpy as np
import torch
from tvm.contrib.msc.core.tools.quantize import QuantizeMethod, BaseQuantizer
from tvm.contrib.msc.core.utils.namespace import MSCFramework
from tvm.contrib.msc.core import utils as msc_utils


class TorchQuantizeMethod(QuantizeMethod):
    """Default quantize method for torch"""

    @classmethod
    def amplify_data(
        cls,
        data: torch.Tensor,
        scale: float,
        min_val: float,
        max_val: float,
        rounding: str = "round",
    ) -> torch.Tensor:
        """Amplify the data

        Parameters
        ----------
        data: torch.Tensor
            The source data.
        scale: float
            The scale factor
        min_val: float
            The min.
        max_val: float
            The max.
        rounding: str
            The round method

        Returns
        -------
        data: torch.Tensor
            The processed data.
        """

        if rounding == "null":
            return torch.clamp(data * scale, min_val, max_val)
        if rounding == "floor":
            return torch.clamp(torch.floor(data * scale), min_val, max_val)
        if rounding == "ceil":
            return torch.clamp(torch.ceil(data * scale), min_val, max_val)
        if rounding == "round":
            return torch.clamp(torch.round(data * scale), min_val, max_val)
        if rounding == "trunc":
            return torch.clamp(torch.trunc(data * scale), min_val, max_val)
        if rounding == "logic_round":
            data = torch.clamp(data * scale, min_val, max_val)
            negative_ceil = torch.where(
                torch.logical_and(data < 0, (data - torch.floor(data)) == 0.5), torch.ceil(data), 0
            )
            data = torch.where(
                torch.logical_and(data < 0, (data - torch.floor(data)) == 0.5), 0, data
            )
            data = torch.where((data - torch.floor(data)) >= 0.5, torch.ceil(data), data)
            data = torch.where((data - torch.floor(data)) < 0.5, torch.floor(data), data)
            return data + negative_ceil
        raise TypeError("Unexpected rounding " + str(rounding))

    @classmethod
    def gather_maxmin(
        cls,
        quantizer: BaseQuantizer,
        data: torch.Tensor,
        name: str,
        consumer: str,
        plan: dict,
        nbits: int = 8,
    ) -> dict:
        """Gather the data by max/min

        Parameters
        ----------
        quantizer: BaseQuantizer
            The quantizer
        data: np.ndarray
            The source data.
        name: str
            The name of the tensor.
        consumer: str
            The name of the consumer.
        plan: dict
            The pre-calibrated plan.
        nbits: int
            The number bits for quantize.

        Returns
        -------
        plan: dict
            The plan of the tensor.
        """

        abs_max_list = plan.get("abs_max_list", [])
        abs_max_list.append(float(torch.abs(data).max()))
        max_list = plan.get("max_list", [])
        max_list.append(float(data.max()))
        min_list = plan.get("min_list", [])
        min_list.append(float(data.min()))
        return {
            "abs_max_list": abs_max_list,
            "max_list": max_list,
            "min_list": min_list,
            "calibrated": False,
        }

    @classmethod
    def gather_max_per_channel(
        cls,
        quantizer: BaseQuantizer,
        data: torch.Tensor,
        name: str,
        consumer: str,
        plan: dict,
        nbits: int = 8,
        channel: str = "O",
        auto_unsign: bool = False,
    ) -> dict:
        """Gather the data by max_per_channel

        Parameters
        ----------
        quantizer: BaseQuantizer
            The quantizer
        data: np.ndarray
            The source data.
        name: str
            The name of the tensor.
        consumer: str
            The name of the consumer.
        plan: dict
            The pre-calibrated plan.
        nbits: int
            The number bits for quantize.
        channel: str
            The channel reference.
        auto_unsign: bool
            Whether to use auto unsign.

        Returns
        -------
        plan: dict
            The plan of the tensor.
        """

        weight = quantizer.find_tensor(name)
        axis = weight.layout_of(channel)
        channel_max = [torch.abs(d).max() for d in torch.chunk(data, data.shape[axis], dim=axis)]
        sign = data.min() < 0 if auto_unsign else True
        valid_range = 2 ** (nbits - int(sign)) - 1
        scale = [valid_range / float(m) for m in channel_max]
        return {"scale": scale, "sign": sign, "axis": axis, "calibrated": True}

    @classmethod
    def quantize_normal(
        cls,
        quantizer: BaseQuantizer,
        data: torch.Tensor,
        name: str,
        consumer: str,
        scale: float,
        nbits: int = 8,
        axis: int = -1,
        sign: bool = True,
        rounding: str = "round",
        epsilon: float = 1.0 / (1 << 24),
    ) -> torch.Tensor:
        """Calibrate the data by kl_divergence

        Parameters
        ----------
        quantizer: BaseQuantizer
            The quantizer
        data: torch.Tensor
            The source data.
        name: str
            The name of the tensor.
        consumer: str
            The name of the consumer.
        scale: float
            The scale factor
        nbits: int
            The number bits for quantize.
        axis: int
            The axis.
        sign: bool
            Whether to use sign.
        rounding str
            The rounding method.
        epsilon: float
            The epsilon for get scale.

        Returns
        -------
        data: torch.Tensor
            The processed tensor.
        """

        valid_range = 2 ** (nbits - int(sign)) - 1
        min_val = -valid_range if sign else 0
        scale_tensor = quantizer._get_tensor_cache(name, consumer, "scale_tensor")
        if scale_tensor is None:
            scale_tensor = cls.get_scale_tensor(data, scale, axis, epsilon)
            if isinstance(scale_tensor, np.ndarray):
                scale_tensor = torch.from_numpy(scale_tensor).to(data.device)
            quantizer._save_tensor_cache(name, consumer, "scale_tensor", scale_tensor)
        data = cls.amplify_data(data, scale_tensor, min_val, valid_range, rounding)
        return data / scale_tensor

    @classmethod
    def framework(cls):
        return MSCFramework.TORCH


msc_utils.register_tool_method(TorchQuantizeMethod)
