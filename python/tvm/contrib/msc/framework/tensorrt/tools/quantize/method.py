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
"""tvm.contrib.msc.framework.tensorrt.tools.quantize.method"""

from typing import Dict

from tvm.contrib.msc.core.tools.quantize import QuantizeMethod, BaseQuantizer
from tvm.contrib.msc.core.utils.namespace import MSCFramework
from tvm.contrib.msc.core import utils as msc_utils


@msc_utils.register_tool_method
class TensorRTQuantizeMethod(QuantizeMethod):
    """Default quantize method for tensorrt"""

    @classmethod
    def quantize_normal(
        cls,
        quantizer: BaseQuantizer,
        tensor_ctx: Dict[str, str],
        name: str,
        consumer: str,
        scale: float,
        nbits: int = 8,
        axis: int = -1,
        sign: bool = True,
        rounding: str = "round",
        epsilon: float = 1.0 / (1 << 24),
    ) -> Dict[str, str]:
        """Calibrate the data by kl_divergence

        Parameters
        ----------
        quantizer: BaseQuantizer
            The quantizer
        tensor_ctx: dict<str, str>
            Tensor describe items.
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
        tensor_ctx: dict<str, str>
            Tensor describe items.
        """

        if quantizer.is_weight(name):
            return tensor_ctx
        dtype = quantizer.find_tensor(name).dtype_name
        precision = "DataType::k"
        if nbits == 8:
            precision += "INT8"
        elif dtype == "float16":
            precision += "HALF"
        elif dtype == "float32":
            precision += "FLOAT"
        else:
            raise TypeError("nbits {} is not supported".format(nbits))
        tensor_ctx["processed"].extend(
            [
                "{}->setPrecision({})".format(tensor_ctx["producer"], precision),
                "{0}->setDynamicRange(-{1}, {1})".format(tensor_ctx["tensor"], scale),
            ]
        )
        return tensor_ctx

    @classmethod
    def dequantize_normal(
        cls,
        quantizer: BaseQuantizer,
        tensor_ctx: Dict[str, str],
        name: str,
        consumer: str,
        scale: float,
        nbits: int = 8,
        axis: int = -1,
        sign: bool = True,
        rounding: str = "round",
        epsilon: float = 1.0 / (1 << 24),
    ) -> Dict[str, str]:
        """Calibrate the data by kl_divergence

        Parameters
        ----------
        quantizer: BaseQuantizer
            The quantizer
        tensor_ctx: dict<str, str>
            Tensor describe items.
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
        tensor_ctx: dict<str, str>
            Tensor describe items.
        """

        return cls.quantize_normal(
            quantizer, tensor_ctx, name, consumer, scale, nbits, axis, sign, rounding, epsilon
        )

    @classmethod
    def framework(cls):
        return MSCFramework.TENSORRT
