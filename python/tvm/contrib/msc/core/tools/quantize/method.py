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
"""tvm.contrib.msc.core.tools.quantize.method"""

from typing import Union, Any
import numpy as np

from tvm.contrib.msc.core.tools.tool import ToolType, BaseTool
from tvm.contrib.msc.core.utils.namespace import MSCFramework
from tvm.contrib.msc.core import utils as msc_utils


@msc_utils.register_tool_method
class QuantizeMethod(object):
    """Default quantize method"""

    @classmethod
    def amplify_data(
        cls, data: np.array, scale: float, min_val: float, max_val: float, rounding: str = "round"
    ) -> np.ndarray:
        """Amplify the data

        Parameters
        ----------
        data: np.ndarray
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
        data: np.ndarray
            The processed data.
        """

        if rounding == "null":
            return np.clip(data * scale, min_val, max_val)
        if rounding == "floor":
            return np.clip(np.floor(data * scale), min_val, max_val)
        if rounding == "ceil":
            return np.clip(np.ceil(data * scale), min_val, max_val)
        if rounding == "round":
            return np.clip(np.round(data * scale), min_val, max_val)
        if rounding == "trunc":
            return np.clip(np.trunc(data * scale), min_val, max_val)
        if rounding == "logic_round":
            data = np.clip(data * scale, min_val, max_val)
            negative_ceil = np.where(
                np.logical_and(data < 0, (data - np.floor(data)) == 0.5), np.ceil(data), 0
            )
            data = np.where(np.logical_and(data < 0, (data - np.floor(data)) == 0.5), 0, data)
            data = np.where((data - np.floor(data)) >= 0.5, np.ceil(data), data)
            data = np.where((data - np.floor(data)) < 0.5, np.floor(data), data)
            return data + negative_ceil
        raise TypeError("Unexpected rounding " + str(rounding))

    @classmethod
    def get_scale_tensor(
        cls,
        data: Any,
        scale: float,
        axis: int = -1,
        epsilon: float = 1.0 / (1 << 24),
        expand_dims: bool = True,
    ) -> Union[float, np.ndarray]:
        """Get the scale tensor

        Parameters
        ----------
        quantizer: BaseQuantizer
            The quantizer
        data: array_like
            The source data.
        name: str
            The name of the tensor.
        consumer: str
            The name of the consumer.
        scale: float
            The scale factor
        axis: int
            The axis.
        epsilon: float
            The epsilon for get scale.
        expand_dims: bool
            Whether to expand dims

        Returns
        -------
        scale_tensor: np.ndarray
            The processed tensor.
        """

        data = msc_utils.cast_array(data)
        if isinstance(scale, list):
            scale_tensor = np.array(scale).astype(data.dtype)
            if expand_dims:
                scale_shape = [s if idx == axis else 1 for idx, s in enumerate(data.shape)]
                scale_tensor = scale_tensor.reshape(scale_shape)
            if scale_tensor.min() <= epsilon:
                scale_mask = scale_tensor <= epsilon
                scale_tensor[scale_mask] = 0
        elif scale <= epsilon:
            scale_tensor = 0
        else:
            scale_tensor = scale
        return scale_tensor

    @classmethod
    def gather_maxmin(
        cls,
        quantizer: BaseTool,
        data: np.ndarray,
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
        abs_max_list.append(float(np.abs(data).max()))
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
    def gather_kl_divergence(
        cls,
        quantizer: BaseTool,
        data: np.ndarray,
        name: str,
        consumer: str,
        plan: dict,
        nbits: int = 8,
        bins: int = 4096,
    ) -> dict:
        """Gather the data by kl_divergence

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
        bins: int
            The number bins.

        Returns
        -------
        plan: dict
            The plan of the tensor.
        """

        if not plan or "abs_max" not in plan:
            return cls.gather_maxmin(quantizer, name, data, plan, nbits)
        hist, edge = np.histogram(data, bins=bins, range=[-plan["abs_max"], plan["abs_max"]])
        hist_list = plan.get("hist_list", [])
        return {"hist_list": hist_list + [hist], "edge": edge, **plan}

    @classmethod
    def gather_max_per_channel(
        cls,
        quantizer: BaseTool,
        data: np.ndarray,
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
        channel_datas = np.split(data, data.shape[axis], axis)
        channel_max = [float(np.abs(d).max()) for d in channel_datas]
        sign = data.min() < 0 if auto_unsign else True
        valid_range = 2 ** (nbits - int(sign)) - 1
        scale = [valid_range / m for m in channel_max]
        return {"scale": scale, "sign": sign, "axis": axis, "calibrated": True}

    @classmethod
    def calibrate_maxmin(
        cls,
        quantizer: BaseTool,
        name: str,
        consumer: str,
        plan: dict,
        nbits: int = 8,
        auto_unsign: bool = False,
    ) -> dict:
        """Calibrate the data by kl_divergence

        Parameters
        ----------
        quantizer: BaseQuantizer
            The quantizer
        name: str
            The name of the tensor.
        consumer: str
            The name of the consumer.
        plan: dict
            The pre-calibrated plan.
        nbits: int
            The number bits for quantize.
        auto_unsign: bool
            Whether to use auto unsign.

        Returns
        -------
        plan: dict
            The plan of the tensor.
        """

        sign = plan["min"] < 0 if auto_unsign else True
        valid_range = 2 ** (nbits - int(sign)) - 1
        abs_max = float(np.array(plan["abs_max_list"]).max())
        return {"scale": valid_range / abs_max, "sign": sign, "calibrated": True}

    @classmethod
    def calibrate_kl_divergence(
        cls,
        quantizer: BaseTool,
        name: str,
        consumer: str,
        plan: dict,
        nbits: int = 8,
        bins: int = 4096,
        auto_unsign: bool = False,
    ) -> dict:
        """Calibrate the data by kl_divergence

        Parameters
        ----------
        quantizer: BaseQuantizer
            The quantizer
        name: str
            The name of the tensor.
        consumer: str
            The name of the consumer.
        plan: dict
            The pre-calibrated plan.
        nbits: int
            The number bits for quantize.
        bins: int
            The number bins.
        auto_unsign: bool
            Whether to use auto unsign.

        Returns
        -------
        plan: dict
            The plan of the tensor.
        """

        # pylint: disable=import-outside-toplevel
        import ctypes
        from tvm.relay import quantize as _quantize

        if plan and "abs_max_list" in plan:
            return {
                "abs_max": float(np.array(plan["abs_max_list"]).max()),
                "max": float(np.array(plan["max_list"]).max()),
                "min": float(np.array(plan["min_list"]).min()),
                "calibrated": False,
            }

        def get_pointer(arr, ctypes_type):
            ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes_type))
            return ctypes.cast(ptr, ctypes.c_void_p)

        sign = plan["min"] < 0 if auto_unsign else True
        hist = np.array(plan["hist_list"]).sum(axis=0)
        hist_ptr = get_pointer(hist.astype(np.int64), ctypes.c_int64)
        edge_ptr = get_pointer(plan["edge"].astype(np.float32), ctypes.c_float)
        valid_range = 2 ** (nbits - int(sign)) - 1
        scale = _quantize._quantize.FindScaleByKLMinimization(hist_ptr, edge_ptr, bins, valid_range)
        return {"scale": valid_range / scale, "sign": sign, "calibrated": True}

    @classmethod
    def quantize_normal(
        cls,
        quantizer: BaseTool,
        data: np.ndarray,
        name: str,
        consumer: str,
        scale: float,
        nbits: int = 8,
        axis: int = -1,
        sign: bool = True,
        rounding: str = "round",
        epsilon: float = 1.0 / (1 << 24),
    ) -> np.ndarray:
        """Calibrate the data by kl_divergence

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
        data: array like
            The processed tensor.
        """

        valid_range = 2 ** (nbits - int(sign)) - 1
        min_val = -valid_range if sign else 0
        scale_tensor = quantizer._get_tensor_cache(name, consumer, "scale_tensor")
        if scale_tensor is None:
            scale_tensor = cls.get_scale_tensor(data, scale, axis, epsilon)
            quantizer._save_tensor_cache(name, consumer, "scale_tensor", scale_tensor)
        data = cls.amplify_data(data, scale_tensor, min_val, valid_range, rounding)
        return data / scale

    @classmethod
    def dequantize_normal(
        cls,
        quantizer: BaseTool,
        data: np.ndarray,
        name: str,
        consumer: str,
        scale: float = -1.0,
        nbits: int = 8,
        axis: int = -1,
        sign: bool = True,
        rounding: str = "round",
        epsilon: float = 1.0 / (1 << 24),
    ) -> np.ndarray:
        """Calibrate the data by kl_divergence

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
        data: array like
            The processed tensor.
        """

        return data

    @classmethod
    def framework(cls):
        return MSCFramework.MSC

    @classmethod
    def tool_type(cls):
        return ToolType.QUANTIZER

    @classmethod
    def method_style(cls):
        return "default"
