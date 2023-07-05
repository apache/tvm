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
"""The TIR serialization specification for Arm(R) Ethos(TM)-U NPU."""
from typing import Union
from typing import get_type_hints
from inspect import isclass

import tvm
from tvm.relay.backend.contrib.ethosu import util


def create_serial_object(serialized_type, deserialized_elements):
    """
    This function will create serialized type that is one of the subclasses
    of tvm.relay.backend.contrib.ethosu.tir.spec.SerializableFormat

    Parameters
    ----------
    serialized_type : a subclass type of SerializableFormat

    deserialized_elements : list
        The list of arguments that needs to packed to create SerializableFormat objects

    Returns
    -------
    The constructed object of type serialized_type
    """

    def _create_serial_object(internal_serialized_type, read_element_idx=0):
        """The internal function that increments the read_element_idx
        when creating nested serial objects"""
        arg_len = util.get_arg_count(internal_serialized_type.__init__) - 1
        serial_init_types = get_type_hints(internal_serialized_type.__init__)
        serial_init_arg_names = list(serial_init_types.keys())
        serial_init_args = []
        assert arg_len == len(serial_init_arg_names)
        for si_arg_name in serial_init_arg_names:
            si_arg_type = serial_init_types[si_arg_name]
            if isclass(si_arg_type) and issubclass(si_arg_type, SerializableFormat):
                sia, read_element_idx = _create_serial_object(si_arg_type, read_element_idx)
                serial_init_args.append(sia)
            else:
                serial_init_args.append(deserialized_elements[read_element_idx])
                read_element_idx += 1
        return internal_serialized_type(*serial_init_args), read_element_idx

    # Just return the primary serial object
    return _create_serial_object(serialized_type)[0]


class SerializableFormat:
    """Base class to retrieve arguments on a predefined ordering"""

    def __iter__(self):
        # Note class attribute definition order is preserved - see PEP 520
        for name in self.__dict__:
            value = self.__getattribute__(name)
            if isinstance(value, SerializableFormat):
                yield from list(value)
            else:
                yield value

    def __getitem__(self, index):
        # Note class attribute definition order is preserved - see PEP 520
        name = list(self.__dict__.keys())[index]
        return self.__getattribute__(name)


class SerialFeatureMap(SerializableFormat):
    """Specialization class to retrieve arguments of a Feature Map
    (similiar to NpuFeatureMap of Vela) on a predefined ordering"""

    def __init__(
        self,
        data_type: str,
        height: int,
        width: int,
        channels: int,
        tile_height_0: int,
        tile_height_1: int,
        tile_width_0: int,
        tile_address_0: tvm.tir.expr.BufferLoad,
        tile_address_1: Union[tvm.tir.expr.BufferLoad, int],
        tile_address_2: Union[tvm.tir.expr.BufferLoad, int],
        tile_address_3: Union[tvm.tir.expr.BufferLoad, int],
        scale: float,
        zero_point: int,
        layout: str,
        stride_h: int,
        stride_w: int,
        stride_c: int,
    ):
        self.data_type = data_type
        self.height = height
        self.width = width
        self.channels = channels
        self.tile_height_0 = tile_height_0
        self.tile_height_1 = tile_height_1
        self.tile_width_0 = tile_width_0
        self.tile_address_0 = tile_address_0
        self.tile_address_1 = tile_address_1
        self.tile_address_2 = tile_address_2
        self.tile_address_3 = tile_address_3
        self.scale = scale
        self.zero_point = zero_point
        self.layout = layout
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.stride_c = stride_c


class SerialKernel(SerializableFormat):
    """Specialization class to retrieve arguments of a Kernel
    (similiar to NpuKernel of Vela) on a predefined ordering"""

    def __init__(
        self,
        width: int,
        height: int,
        stride_w: int,
        stride_h: int,
        dilation_w: int,
        dilation_h: int,
    ):
        self.width = width
        self.height = height
        self.stride_w = stride_w
        self.stride_h = stride_h
        self.dilation_w = dilation_w
        self.dilation_h = dilation_h


class SerialAddressRange(SerializableFormat):
    """Specialization class to retrieve arguments of a AddressRange
    (similiar to NpuAddressRange of Vela) on a predefined ordering"""

    def __init__(self, address: tvm.tir.expr.BufferLoad, length: int):
        self.address = address
        self.length = length


class SerialPadding(SerializableFormat):
    """Specialization class to retrieve arguments of a Padding
    (similiar to NpuPadding of Vela) on a predefined ordering"""

    def __init__(self, top: int, left: int, bottom: int, right: int):
        self.top = top
        self.left = left
        self.bottom = bottom
        self.right = right


class SerialActivation(SerializableFormat):
    """Specialization class to retrieve arguments of a Activation
    (similiar to NpuActivation of Vela) on a predefined ordering"""

    def __init__(self, op: str, clip_min: int, clip_max: int):
        self.op = op
        self.clip_min = clip_min
        self.clip_max = clip_max


class SerialBlockConfig(SerializableFormat):
    """Specialization class to retrieve arguments of a BlockConfig
    (similar to NpuBlockConfig of Vela) on a predefined ordering"""

    def __init__(self, height: int, width: int, depth: int):
        self.height = height
        self.width = width
        self.depth = depth


class SerialRescaleConfig(SerializableFormat):
    """Specialization class to retrieve arguments of a rescale parameters
    (to fill in rescale field in Vela NpuElementWiseOperation) on a predefined ordering"""

    def __init__(self, use_rescale: bool, scale: int, shift: int):
        self.use_rescale = use_rescale
        self.scale = scale
        self.shift = shift


class Serial2DConvolution(SerializableFormat):
    """Specialization class to retrieve arguments of
    a ethosu.conv2d tir extern call on a predefined ordering"""

    def __init__(
        self,
        ifm: SerialFeatureMap,
        ofm: SerialFeatureMap,
        kernel: SerialKernel,
        weight: SerialAddressRange,
        weight2: SerialAddressRange,
        weight_zero_point: int,
        scale_bias: SerialAddressRange,
        scale_bias2: SerialAddressRange,
        padding: SerialPadding,
        activation: SerialActivation,
        rounding_mode: str,
        upscale: str,
        block_config: SerialBlockConfig,
    ):
        self.ifm = ifm
        self.ofm = ofm
        self.kernel = kernel
        self.weight = weight
        self.weight2 = weight2
        self.weight_zero_point = weight_zero_point
        self.scale_bias = scale_bias
        self.scale_bias2 = scale_bias2
        self.padding = padding
        self.activation = activation
        self.rounding_mode = rounding_mode
        self.upscale = upscale
        self.block_config = block_config


class Serial2DDepthwise(SerializableFormat):
    """Specialization class to retrieve arguments of
    a ethosu.depthwise_conv2d TIR extern call on a predefined ordering"""

    def __init__(
        self,
        ifm: SerialFeatureMap,
        ofm: SerialFeatureMap,
        kernel: SerialKernel,
        weight: SerialAddressRange,
        weight_zero_point: int,
        scale_bias: SerialAddressRange,
        padding: SerialPadding,
        activation: SerialActivation,
        rounding_mode: str,
        upscale: str,
        block_config: SerialBlockConfig,
    ):
        self.ifm = ifm
        self.ofm = ofm
        self.kernel = kernel
        self.weight = weight
        self.weight_zero_point = weight_zero_point
        self.scale_bias = scale_bias
        self.padding = padding
        self.activation = activation
        self.rounding_mode = rounding_mode
        self.upscale = upscale
        self.block_config = block_config


class SerialCopy(SerializableFormat):
    """Specialization class to retrieve arguments of
    a ethosu.copy tir extern call on a predefined ordering"""

    def __init__(
        self,
        read_address: tvm.tir.expr.BufferLoad,
        length: int,
        write_address: tvm.tir.expr.BufferLoad,
    ):
        self.read_address = read_address
        self.length = length
        self.write_address = write_address


class SerialPooling(SerializableFormat):
    """Specialization class to retrieve arguments of
    a ethosu.pooling tir extern call on a predefined ordering"""

    def __init__(
        self,
        ifm: SerialFeatureMap,
        ofm: SerialFeatureMap,
        pooling_type: str,
        pool_shape: SerialKernel,
        padding: SerialPadding,
        activation: SerialActivation,
        rounding_mode: str,
        upscale: str,
        block_config: SerialBlockConfig,
    ):
        self.ifm = ifm
        self.ofm = ofm
        self.pooling_type = pooling_type
        self.pool_shape = pool_shape
        self.padding = padding
        self.activation = activation
        self.rounding_mode = rounding_mode
        self.upscale = upscale
        self.block_config = block_config


class SerialBinaryElementwise(SerializableFormat):
    """Specialization class to retrieve arguments of
    a ethosu.binary_elementwise tir extern call on a predefined ordering"""

    def __init__(
        self,
        ifm: SerialFeatureMap,
        ifm2: SerialFeatureMap,
        ofm: SerialFeatureMap,
        operator_type: str,
        reversed_operands: bool,
        activation: SerialActivation,
        rounding_mode: str,
        block_config: SerialBlockConfig,
        rescale_config: SerialRescaleConfig,
    ):
        self.ifm = ifm
        self.ifm2 = ifm2
        self.ofm = ofm
        self.operator_type = operator_type
        self.reversed_operands = reversed_operands
        self.activation = activation
        self.rounding_mode = rounding_mode
        self.block_config = block_config
        self.rescale_config = rescale_config


class SerialUnaryElementwise(SerializableFormat):
    """Specialization class to retrieve arguments of
    a ethosu.unary_elementwise tir extern call on a predefined ordering"""

    def __init__(
        self,
        ifm: SerialFeatureMap,
        ofm: SerialFeatureMap,
        operator_type: str,
        activation: SerialActivation,
        rounding_mode: str,
        block_config: SerialBlockConfig,
    ):
        self.ifm = ifm
        self.ofm = ofm
        self.operator_type = operator_type
        self.activation = activation
        self.rounding_mode = rounding_mode
        self.block_config = block_config
