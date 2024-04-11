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
"""tvm.contrib.msc.framework.tvm.tools.quantize.method"""

from typing import Tuple
import numpy as np

import tvm
from tvm.relax import op as relax_op
from tvm.contrib.msc.core.tools.quantize import QuantizeMethod, BaseQuantizer
from tvm.contrib.msc.core.utils.namespace import MSCFramework
from tvm.contrib.msc.core import utils as msc_utils
from tvm.contrib.msc.core import _ffi_api


@msc_utils.register_tool_method
class TVMQuantizeMethod(QuantizeMethod):
    """Default quantize method for tvm"""

    @classmethod
    def get_quantize_cache(
        cls,
        quantizer: BaseQuantizer,
        data: tvm.relax.Var,
        name: str,
        consumer: str,
        scale: float,
        axis: int = -1,
        epsilon: float = 1.0 / (1 << 24),
    ) -> Tuple[tvm.relax.Constant, tvm.relax.Constant]:
        """Calibrate the data by kl_divergence

        Parameters
        ----------
        quantizer: BaseQuantizer
            The quantizer
        data: tvm.relax.Var
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

        Returns
        -------
        scale_tensor: tvm.relax.Constant
            The scale_tensor.
        zero_point: tvm.relax.Constant
            The zero_point.
        """

        name_prefix = name if quantizer._cache_processed else quantizer.to_tensor_id(name, consumer)
        scale_tensor = quantizer._get_tensor_cache(name, consumer, "scale_tensor")
        zero_point = quantizer._get_tensor_cache(name, consumer, "zero_point")
        if scale_tensor is None:
            scale_tensor = cls.get_scale_tensor(data, scale, axis, epsilon, expand_dims=False)
            scale_tensor = 1 / scale_tensor
            if isinstance(scale_tensor, float):
                scale_tensor = np.array(scale_tensor)
            scale_tensor = scale_tensor.astype(quantizer.find_tensor(name).dtype_name)
            zero_point = np.zeros_like(scale_tensor).astype("int8")
            scale_span = _ffi_api.SpanCreateWithAttr("name", name_prefix + "_scale")
            scale_tensor = tvm.relax.Constant(tvm.nd.array(scale_tensor), span=scale_span)
            zp_span = _ffi_api.SpanCreateWithAttr("name", name_prefix + "_zero_point")
            zero_point = tvm.relax.Constant(tvm.nd.array(zero_point), span=zp_span)
            quantizer._save_tensor_cache(name, consumer, "scale_tensor", scale_tensor)
            quantizer._save_tensor_cache(name, consumer, "zero_point", zero_point)
        return scale_tensor, zero_point

    @classmethod
    def quantize_normal(
        cls,
        quantizer: BaseQuantizer,
        data: tvm.relax.Var,
        name: str,
        consumer: str,
        scale: float,
        nbits: int = 8,
        axis: int = -1,
        sign: bool = True,
        rounding: str = "round",
        epsilon: float = 1.0 / (1 << 24),
    ) -> tvm.relax.Var:
        """Calibrate the data by kl_divergence

        Parameters
        ----------
        quantizer: BaseQuantizer
            The quantizer
        data: tvm.relax.Var
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
        data: tvm.relax.Var
            The processed tensor.
        """

        if nbits == 8:
            dtype = "int8"
        else:
            raise TypeError("Unexpected nbits " + str(nbits))
        name_prefix = name if quantizer._cache_processed else quantizer.to_tensor_id(name, consumer)
        scale_tensor, zero_point = cls.get_quantize_cache(
            quantizer, data, name, consumer, scale, axis, epsilon
        )
        expr = relax_op.quantize(data, scale_tensor, zero_point, axis, dtype)
        return quantizer._block_builder.emit(expr, name_hint=name_prefix + "_quantize")

    @classmethod
    def dequantize_normal(
        cls,
        quantizer: BaseQuantizer,
        data: tvm.relax.Var,
        name: str,
        consumer: str,
        scale: float = -1.0,
        nbits: int = 8,
        axis: int = -1,
        sign: bool = True,
        rounding: str = "round",
        epsilon: float = 1.0 / (1 << 24),
    ) -> tvm.relax.Var:
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

        name_prefix = name if quantizer._cache_processed else quantizer.to_tensor_id(name, consumer)
        scale_tensor, zero_point = cls.get_quantize_cache(
            quantizer, data, name, consumer, scale, axis, epsilon
        )
        expr = relax_op.dequantize(
            data, scale_tensor, zero_point, axis, quantizer.find_tensor(name).dtype
        )
        return quantizer._block_builder.emit(expr, name_hint=name_prefix + "_dequantize")

    @classmethod
    def framework(cls):
        return MSCFramework.TVM
