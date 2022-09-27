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
# pylint: disable=invalid-name
# pylint: disable=too-many-nested-blocks
"""Device config class to hold information about the target hardware"""
from typing import Tuple, List, Dict, Optional
from functools import reduce

import math
import numpy as np

import tvm
from . import BlockConfig
from . import StripeConfig
from . import Propagator


def _round_up(a: int, b: int) -> int:
    """Round up to a multiple of b"""
    return ((a + b - 1) // b) * b


def _round_up_div(a: int, b: int) -> int:
    """Divide by b and round up to a multiple of b"""
    return (a + b - 1) // b


class _Shape:
    """Helper class for dealing with Tensor shapes of different layouts"""

    def __init__(self, shape: List[int], layout="NHWC"):
        if layout == "NHCWB16":
            self.height = int(shape[1])
            self.width = int(shape[3])
            self.depth = int(shape[2]) * int(shape[4])
        else:
            # identity layout is NHWC but the shape is not always 4
            length = len(shape)
            if length == 4:
                self.height = int(shape[1])
                self.width = int(shape[2])
                self.depth = int(shape[3])
            elif length == 3:
                self.height = int(shape[0])
                self.width = int(shape[1])
                self.depth = int(shape[2])
            elif length == 2:
                self.height = int(shape[0])
                self.width = int(shape[1])
                self.depth = 1
            elif length == 1:
                self.height = int(shape[0])
                self.width = 1
                self.depth = 1

    def round_up(self, other: "_Shape"):
        self.height = _round_up(self.height, other.height)
        self.width = _round_up(self.width, other.width)
        self.depth = _round_up(self.depth, other.depth)

    def area(self) -> int:
        return self.height * self.width

    def as_list(self):
        return [1, self.height, self.width, self.depth]


class EthosuDeviceConfig:
    """Arm(R) Ethos(TM)-U NPU config class"""

    def __init__(self, device: str, disable_block_bulling: bool = False):
        self._device = device
        self._subkernel_limits = (8, 8)
        self._output_cycles = (1, 2, 3, 4, 6)
        self._split_depth = 16
        self._max_block_shape = _Shape([1, 32, 64, 128])
        self._bank_size_bytes = 1024
        self._disable_block_culling = disable_block_bulling
        if self._device == "ethos-u55-256":
            self._micro_block = _Shape([1, 2, 2, 8])
            self._input_micro_block = _Shape([1, 2, 2, 8])
            self._delay_cycles = (2, 2)
            self._activation_cycles = (0.25, 1)
            self._output_units = 8

            self._total_banks = 48
            self._reserved_banks = 4
            self._input_granularity = {1: 8, 2: 8, 4: 16}
            self._accumulator_granularity = {4: 16, 5: 20}
            self._lut_reserved = True
        elif self._device == "ethos-u55-128":
            self._micro_block = _Shape([1, 1, 2, 8])
            self._input_micro_block = _Shape([1, 1, 2, 8])
            self._delay_cycles = (2, 3)
            self._activation_cycles = (0.5, 1)
            self._output_units = 4

            self._total_banks = 24
            self._reserved_banks = 4
            self._input_granularity = {1: 4, 2: 4, 4: 8}
            self._accumulator_granularity = {4: 8, 5: 12}
            self._lut_reserved = True
        elif self._device == "ethos-u55-64":
            self._micro_block = _Shape([1, 1, 1, 8])
            self._input_micro_block = _Shape([1, 1, 1, 8])
            self._delay_cycles = (2, 3)
            self._activation_cycles = (1, 1)
            self._output_units = 2

            self._total_banks = 16
            self._reserved_banks = 2
            self._input_granularity = {1: 2, 2: 2, 4: 4}
            self._accumulator_granularity = {4: 4, 5: 8}
            self._lut_reserved = False
        elif self._device == "ethos-u55-32":
            self._micro_block = _Shape([1, 1, 1, 4])
            self._input_micro_block = _Shape([1, 1, 1, 8])
            self._delay_cycles = (3, 7)
            self._activation_cycles = (1, 2)
            self._output_units = 1

            self._total_banks = 16
            self._reserved_banks = 2
            self._input_granularity = {1: 2, 2: 2, 4: 4}
            self._accumulator_granularity = {4: 4, 5: 4}
            self._lut_reserved = False

    def _get_output_cycles(
        self, op_type: str, op_str: str, ifm_dtype: str, ofm_dtype: str, activation: str
    ) -> float:
        """Estimate cycles per output element for an NPU operator

        Parameters
        ----------
        op_type : str
            The NPU primitive operator
                "ethosu_pooling"
        op_str : str
            The type of NPU operator.
                "MAX"
        ifm_dtype: str
            Datatype of the Input Feature Map tensor (IFM)
        ofm_dtype: str
            Datatype of the Output Feature Map tensor (OFM)
        activation : str
            The activation function to use.
                "NONE" - no activation function.
                "CLIP" - clip the output between clip_min and clip_max.
                "TANH" - tanh activation function.
                "SIGMOID" - sigmoid activation function.
                "LUT" - use a look-up table to perform the activation function.

        Returns
        -------
        float
            The cycles per output element
        """
        cycles = 0
        bw_limit = 0
        if op_type == "ethosu_pooling" and op_str == "MAX":
            cycles = self._output_cycles[0]
        elif op_type in ("ethosu_pooling", "ethosu_conv2d", "ethosu_depthwise_conv2d"):
            cycles = self._output_cycles[1] if ifm_dtype == "int8" else self._output_cycles[2]
        elif op_type == "ethosu_binary_elementwise":
            # Binary Bandwidth Limitations
            if ifm_dtype == "int8":
                bw_limit = 0.125 if ofm_dtype == "int8" else 0.75
            elif ifm_dtype == "int16":
                bw_limit = 0.75 if ofm_dtype == "int16" else 1
            else:
                bw_limit = 1.5

            if op_str in ("MIN", "MAX"):
                cycles = self._output_cycles[1]
            elif op_str == "MUL":
                cycles = self._output_cycles[2]
            if op_str in ("ADD", "SUB"):
                if ofm_dtype == "int32":
                    cycles = (
                        self._output_cycles[2] if ifm_dtype == "int32" else self._output_cycles[3]
                    )
                else:
                    cycles = self._output_cycles[4]

        elif op_type == "ethosu_unary_elementwise":
            # Unary Bandwidth Limitations
            if ifm_dtype == "int16":
                bw_limit = 0.25
            elif ifm_dtype == "int32":
                bw_limit = 1

            if op_str == "CLZ":
                cycles = self._output_cycles[1]
            elif op_str in ("SHL", "SHR"):
                cycles = self._output_cycles[2]
            elif op_str in ("LRELU", "ABS"):
                cycles = self._output_cycles[1]
                if ifm_dtype == "int16":
                    bw_limit = 0.5

        act_cycles = 0
        if activation == "CLIP":
            act_cycles = self._activation_cycles[0]
        elif activation in ("LUT", "TANH", "SIGMOID"):
            act_cycles = self._activation_cycles[1]

        return max((cycles / self._output_units), act_cycles, bw_limit)

    def _get_delay_cycles(self, op_type: str, ifm_dtype: str) -> int:
        """Get the number of delay cycles during a bubble

        Parameters
        ----------
        op_type : str
            The NPU primitive operator
                "ethosu_pooling"
        op_str : str
            The type of NPU operator.
                "MAX"
        ifm_dtype: str
            Datatype of the Input Feature Map tensor (IFM)

        Returns
        ----------
        int
            The amount of delay cycles
        """
        if op_type in ("ethosu_conv2d", "ethosu_depthwise2d", "ethosu_pooling"):
            if ifm_dtype == "int16":
                return self._delay_cycles[1]

            return self._delay_cycles[0]

        return 0

    def _get_weight_decoder_cycles(self, op_type: str) -> int:
        """Get cycle estimate for weight decoding

        Parameters
        ----------
        op_type: str
            The NPU primitive operator
                "ethosu_pooling"

        Returns
        ----------
        int
            Estimated cycles for weight decoding
        """
        if op_type in ("ethosu_conv2d", "ethosu_depthwise2d"):
            return 32 * self._micro_block.depth // 8

        return 0

    def get_output_quantum(self, ofm_layout: str) -> Tuple[int]:
        """Get the atomic output volume

        Parameters
        ----------
        ofm_layout : str
            The layout of the Output Feature Map tensor. Can be "NHWC" or "NHCWB16".

        Returns
        ----------
        Tuple[int]
            The atomic output volume formatted to the ofm_layout parameter
        """
        if ofm_layout == "NHCWB16":
            return [
                1,
                self._micro_block.height,
                1,
                self._micro_block.width,
                self._micro_block.depth,
            ]

        return self._micro_block.as_list()

    def _align(self, x: int, n: int) -> int:
        return int(math.ceil(x / n) * n)

    def _get_input_size(
        self, output_size: int, kernel_stride: int, border: int, upscaling_factor: int
    ) -> int:
        return int(math.ceil(((output_size - 1) * kernel_stride + border)) / upscaling_factor)

    def _get_dilated_kernel_size(self, kernel_size: int, dilation: int) -> int:
        return (kernel_size - 1) * dilation + 1

    def _get_input_block(
        self,
        output_block: _Shape,
        input_shape: _Shape,
        dtype: str,
        op_type: str,
        partkernel: bool,
        stride_h: int,
        stride_w: int,
        dilated_kernel_h: int,
        dilated_kernel_w: int,
        upscaling_factor: int,
    ) -> _Shape:
        height = self._get_input_size(
            output_block.height,
            stride_h,
            min(dilated_kernel_h, self._subkernel_limits[0]),
            upscaling_factor,
        )
        width = self._get_input_size(
            output_block.width,
            stride_w,
            min(dilated_kernel_w, self._subkernel_limits[1]),
            upscaling_factor,
        )

        if op_type == "ethosu_conv2d":
            if dtype == "int8":
                if partkernel:
                    depth = self._align(min(32, input_shape.depth), 8)
                else:
                    depth = self._align(min(16, input_shape.depth), 8)
            elif dtype == "int16":
                depth = self._align(min(16, input_shape.depth), 4)
            else:
                depth = self._align(min(8, input_shape.depth), 2)
        else:
            depth = output_block.depth

        return _Shape(
            [
                1,
                self._align(height, self._micro_block.height),
                self._align(width, self._micro_block.width),
                depth,
            ]
        )

    def get_kernel_steps(
        self,
        op_type: str,
        dilated_kernel_h: int,
        dilated_kernel_w: int,
        ifm_dtype: str,
        partkernel: bool = False,
    ) -> List[int]:
        """Calculate the total number of subkernels and their sizes

        Parameters
        ----------
        op_type : str
            The NPU primitive operator
                "ethosu_pooling"
        dilated_kernel_h: int
            Height of dilated kernel
        dilated_kernel_w: int
            Width of dilated kernel
        ifm_dtype: str
            Datatype of the Input Feature Map tensor (IFM)
        partkernel: bool
            Flag showing whether part-kernel first traversal is used

        Returns
        ----------
        List[int]
            List where each entry contains the amount of elements in one of the subkernels
        """
        if op_type == "ethosu_binary_elementwise":
            return [1]

        subkernels = self._get_subkernels(dilated_kernel_h, dilated_kernel_w)

        # Determine the number of kernel steps per subkernel
        kernel_steps = []
        for y, x in subkernels:
            subkernel_elements = x * y
            if op_type == "ethosu_conv2d" and partkernel:
                # Part-kernel-first traversal conv2d
                divisor = 4 if ifm_dtype == "int8" else 2
                kernel_steps.append(int(_round_up_div(subkernel_elements, divisor)))
            elif op_type == "ethosu_depthwise_conv2d":
                kernel_steps.append(int(_round_up_div(subkernel_elements, 4)))
            else:
                # Depth-first traversal conv2d or pooling
                kernel_steps.append(int(subkernel_elements))

        return kernel_steps

    def _get_subkernels(self, dilated_kernel_h: int, dilated_kernel_w: int):
        num_subkernels_y = _round_up_div(dilated_kernel_h, self._subkernel_limits[0])
        num_subkernels_x = _round_up_div(dilated_kernel_w, self._subkernel_limits[1])
        subkernels_y = [
            min((dilated_kernel_h - i * self._subkernel_limits[0]), self._subkernel_limits[0])
            for i in range(num_subkernels_y)
        ]
        subkernels_x = [
            min((dilated_kernel_w - i * self._subkernel_limits[1]), self._subkernel_limits[1])
            for i in range(num_subkernels_x)
        ]

        subkernels = []
        for y in subkernels_y:
            for x in subkernels_x:
                subkernels.append((y, x))

        return subkernels

    def _get_accumulator_width(self, op_type: str, ifm_dtype: str):
        if ifm_dtype == "int16" and op_type != "ethosu_pooling":
            return 5

        return 4

    def is_partkernel(
        self, op_type: str, ifm_channels: int, ifm_dtype: str, kernel_elements: int
    ) -> bool:
        """Determine which block traversal strategy has better DPU utilization

        Parameters
        ----------
        op_type: str
            The NPU primitive operator
                "ethosu_pooling"
        ifm_channels: int
            Number of input channels
        ifm_dtype: str
            Datatype of the Input Feature Map tensor (IFM)
        kernel_elements: int
            Total number of elements in the kernel

        Returns
        ----------
        bool
            True if partkernel first has best DPU utilization
        """
        if op_type != "ethosu_conv2d":
            return False

        depth_first_utilization = ifm_channels / _round_up(
            ifm_channels, 32 if ifm_dtype == "int8" else 16
        )
        part_kernel_first_utilization = (ifm_channels / _round_up(ifm_channels, 8)) * (
            kernel_elements / _round_up(kernel_elements, 4 if ifm_dtype == "int8" else 2)
        )

        return part_kernel_first_utilization > depth_first_utilization or ifm_channels <= 8

    def _get_input_banks(self, input_block_shape, input_bytewidth):
        input_bytes = input_block_shape.area() * self._align(
            input_block_shape.depth * input_bytewidth, 8
        )
        input_banks = _round_up_div(input_bytes, self._bank_size_bytes) * 2
        input_banks = _round_up(input_banks, self._input_granularity[input_bytewidth])

        return input_banks

    def _get_accumulator_banks(self, output_block_shape, acc_bytewidth):
        acc_depth = _round_up(output_block_shape.depth, 8)
        acc_bytes = output_block_shape.area() * self._align(acc_depth, 8) * acc_bytewidth
        acc_banks = _round_up_div(acc_bytes, self._bank_size_bytes) * 2
        acc_banks = _round_up(acc_banks, self._accumulator_granularity[acc_bytewidth])

        return acc_banks

    @staticmethod
    def _create_layout_block(nhwc_block_config, layout):
        """A helper function to convert to brick layout"""
        if layout == "NHCWB16":
            return [
                nhwc_block_config[0],
                nhwc_block_config[1],
                1 + ((nhwc_block_config[3] - 1) // 16),
                nhwc_block_config[2],
                16,
            ]
        # else it could only be NHWC
        return nhwc_block_config

    def get_elementwise_block_config(
        self,
        ifm_propagator: Propagator,
        ifm2_propagator: Optional[Propagator],
        op_attrs: Dict,
        ofm_shape: List[int],
        output_layout: str,
        input_layout: str,
        input2_layout: Optional[str],
        ifm_dtype: str,
        ofm_dtype: str,
    ) -> List[BlockConfig]:
        """Get a suitable block config for an elementwise operator

        Parameters
        ----------
        ifm_propagator: Propagator,
            The propagator containing the data dependencies between input and output
        ifm2_propagator: Propagator,
            The propagator containing the data dependencies between input2 and output
        op_attrs: Dict,
            Dictionary containing operator attributes
        ofm_shape: List[int],
            Shape of the output tensor
        output_layout: str,
            The layout of the Output Feature Map tensor. Can be "NHWC" or "NHCWB16".
        input_layout: str,
            The layout of the Input Feature Map tensor. Can be "NHWC" or "NHCWB16".
        input2_layout: str,
            The layout of the Input2 Feature Map tensor. Can be "NHWC" or "NHCWB16".
        ifm_dtype: str,
            Datatype of the Input Feature Map tensor (IFM)
        ofm_dtype: str,
            Datatype of the Output Feature Map tensor (OFM)

        Returns
        ----------
        List[BlockConfig]
            List containing a single suitable block config
        """
        block_config = []
        output_shape = [int(a) for a in ofm_shape]

        op_type = op_attrs.get("op")
        op_str = op_attrs.get("op_str")
        activation = op_attrs.get("activation", "NONE")

        input_bytewidth = 1 if ifm_dtype == "int8" else 2 if ifm_dtype == "int16" else 4
        banks_available = self._total_banks - self._reserved_banks
        if activation == "LUT" and not self._lut_reserved:
            banks_available -= 2

        # Handle user-forced block config
        options = tvm.transform.PassContext.current().config.get("relay.ext.ethos-u.options", None)
        if options and options.dev_force_block_config:
            block_config = [int(v) for v in options.dev_force_block_config.split("x")]
            assert len(block_config) == 3
            if output_layout == "NHWC":
                block_shape = [output_shape[0], block_config[0], block_config[1], block_config[2]]
            else:
                block_shape = [
                    output_shape[0],
                    block_config[0],
                    1 + ((block_config[2] - 1) // 16),
                    block_config[1],
                    16,
                ]
            output_cycles = self._get_output_cycles(
                op_type, op_str, ifm_dtype, ofm_dtype, activation
            )
            output_cycles *= reduce(lambda a, b: a * b, block_shape, 1)
            output_cycles = int(math.ceil(output_cycles))
            return [BlockConfig(block_shape, block_shape, 0, output_cycles)]

        # Split the block in half until it fits into SHRAM
        max_height, max_width, max_depth = self._max_block_shape.as_list()[1:]
        if output_layout == "NHCWB16":
            output_height = output_shape[1]
            output_width = output_shape[3]
            output_channels = output_shape[2] * 16
        else:
            output_height = output_shape[1]
            output_width = output_shape[2]
            output_channels = output_shape[3]

        output_nhwc_block = [
            1,
            _round_up(min(output_height, max_height), self._micro_block.height),
            _round_up(min(output_width, max_width), self._micro_block.width),
            _round_up(min(output_channels, max_depth), self._micro_block.depth),
        ]
        output_block = self._create_layout_block(output_nhwc_block, output_layout)
        split_order = (a for a in [1, 2, 3])
        split_axis = next(split_order)

        offset = [0] * len(output_block)
        stripes = [1] * len(output_block)
        order = [1, 2, 4, 3, 0] if output_layout == "NHCWB16" else [1, 2, 3, 4]
        while True:
            # Create stripe config for output block
            output_stripe_config = StripeConfig(
                output_block, output_block, output_block, order, stripes, offset
            )

            # Propagate the output to obtain the two input blocks
            input_block = _Shape(ifm_propagator.propagate(output_stripe_config).shape, input_layout)
            if ifm2_propagator:
                input2_block = _Shape(
                    ifm2_propagator.propagate(output_stripe_config).shape, input2_layout
                )
            else:
                # Unary elementwise
                input2_block = input_block

            input_block.round_up(self._input_micro_block)
            input2_block.round_up(self._input_micro_block)

            # Banks required for input block
            input_banks = self._get_input_banks(input_block, input_bytewidth)
            # Banks required for input2 block
            input2_banks = self._get_input_banks(input2_block, input_bytewidth)

            # Check whether or not both IFMs fit into SHRAM
            if (input_banks + input2_banks) <= banks_available:
                output_cycles = self._get_output_cycles(
                    op_type, op_str, ifm_dtype, ofm_dtype, activation
                )
                output_cycles *= reduce(lambda a, b: a * b, output_block, 1)
                output_cycles = int(math.ceil(output_cycles))
                block_config.append(
                    BlockConfig(input_block.as_list(), output_block, 0, output_cycles)
                )
                break

            if output_nhwc_block[split_axis] == self._micro_block.as_list()[split_axis]:
                split_axis = next(split_order)

            output_nhwc_block[split_axis] = _round_up(
                _round_up_div(output_nhwc_block[split_axis], 2),
                self._micro_block.as_list()[split_axis],
            )
            output_block = self._create_layout_block(output_nhwc_block, output_layout)

        return block_config

    def _get_subkernel_propagator(
        self, op_attrs, ifm_propagator, input_layout, output_layout, depth
    ):
        op_type = op_attrs.get("op")
        stride_h = int(op_attrs.get("stride_h", 1))
        stride_w = int(op_attrs.get("stride_w", 1))
        transform = ifm_propagator.transform

        if op_type != "ethosu_identity":
            if input_layout == "NHCWB16":
                transform[1][-1] = min(transform[1][-1], self._subkernel_limits[0] - stride_h)
                transform[3][-1] = min(transform[3][-1], self._subkernel_limits[1] - stride_w)
            else:
                transform[1][-1] = min(transform[1][-1], self._subkernel_limits[0] - stride_h)
                transform[2][-1] = min(transform[2][-1], self._subkernel_limits[1] - stride_w)

            if op_type in ("ethosu_pooling", "ethosu_depthwise_conv2d"):
                if output_layout == "NHCWB16" and input_layout == "NHWC":
                    transform[3][-1] = depth
                elif output_layout == "NHCWB16" and input_layout == "NHCWB16":
                    transform[2][-1] = 1 + ((depth - 1) // 16)

        return Propagator(transform, ifm_propagator.offset)

    def get_valid_block_configs(
        self,
        ifm_propagator: Propagator,
        op_attrs: Dict,
        ofm_shape: List[int],
        ofm_channels: int,
        ifm_channels: int,
        output_layout: str,
        input_layout: str,
        ifm_dtype: str,
        ofm_dtype: str,
        kernel_h: int = 1,
        kernel_w: int = 1,
    ) -> List[BlockConfig]:
        """Get all of the valid block configs

        Parameters
        ----------
        ifm_propagator: Propagator,
            The propagator containing the data dependencies between input and output
        op_attrs: Dict,
            Dictionary containing operator attributes
        ofm_shape: List[int],
            Shape of the output tensor
        ofm_channels: int,
            Number of output channels
        ifm_channels: int,
            Number of input channels
        output_layout: str,
            The layout of the Output Feature Map tensor. Can be "NHWC" or "NHCWB16".
        input_layout: str,
            The layout of the Input Feature Map tensor. Can be "NHWC" or "NHCWB16".
        ifm_dtype: str,
            Datatype of the Input Feature Map tensor (IFM)
        ofm_dtype: str,
            Datatype of the Output Feature Map tensor (OFM)
        kernel_h: int,
            Height of kernel
        kernel_h: int
            Width of kernel

        Returns
        ----------
        List[BlockConfig]
            List containing all of the valid block configs
        """
        valid_block_configs = []

        op_type = op_attrs.get("op")
        op_str = op_attrs.get("op_str")
        activation = op_attrs.get("activation", "NONE")
        upscaling_factor = 1 if op_attrs.get("upscale", "NONE") == "NONE" else 2

        if output_layout == "NHCWB16":
            output_shape = _Shape([1, ofm_shape[1], ofm_shape[3], ofm_channels])
        else:
            output_shape = _Shape(ofm_shape)

        # Define search space
        max_height = min(output_shape.height, self._max_block_shape.height)
        min_height = max(self._micro_block.height, upscaling_factor)

        max_width = min(output_shape.width, self._max_block_shape.width)
        min_width = max(self._micro_block.width, upscaling_factor)

        max_depth = min(ofm_channels, self._max_block_shape.depth)
        min_depth = max(self._micro_block.depth, upscaling_factor)

        heights = range(min_height, max_height + min_height, min_height)
        widths = range(min_width, max_width + min_width, min_width)
        depths = range(min_depth, max_depth + min_depth, min_depth)

        # Handle user-forced block config
        options = tvm.transform.PassContext.current().config.get("relay.ext.ethos-u.options", None)
        forced = False
        if options and options.dev_force_block_config:
            block_config = [int(v) for v in options.dev_force_block_config.split("x")]
            assert len(block_config) == 3
            heights = [block_config[0]]
            widths = [block_config[1]]
            depths = [block_config[2]]
            forced = True

        input_bytewidth = 1 if ifm_dtype == "int8" else 2
        acc_bytewidth = self._get_accumulator_width(op_type, ifm_dtype)
        banks_available = self._total_banks - self._reserved_banks
        if activation == "LUT" and not self._lut_reserved:
            banks_available -= 2

        # Input block depth has additional limitations for operators that require full input depth
        input_block_depth = 0
        partkernel = self.is_partkernel(op_type, ifm_channels, ifm_dtype, kernel_h * kernel_w)
        if op_type == "ethosu_conv2d":
            if partkernel:
                input_block_depth = min(ifm_channels, 16)
            else:
                input_block_depth = min(ifm_channels, 32)

        for depth in reversed(depths):
            if (depth < output_shape.depth) and (depth % self._split_depth != 0) and not forced:
                # Block depth has to be less than full depth or a multiple of the split depth
                continue

            subkernel_propagator = self._get_subkernel_propagator(
                op_attrs, ifm_propagator, input_layout, output_layout, depth
            )

            for width in reversed(widths):
                for height in reversed(heights):
                    if output_layout == "NHCWB16":
                        output_block = (
                            1,
                            height,
                            1 + ((depth - 1) // 16),
                            width,
                            16,
                        )
                        order = [1, 2, 4, 3, 0]
                    else:
                        output_block = (1, height, width, depth)
                        order = [1, 2, 3, 4]

                    offset = [0] * len(output_block)
                    stripes = [1] * len(output_block)
                    block_stripe_config = StripeConfig(
                        output_block,
                        output_block,
                        output_block,
                        order,
                        stripes,
                        offset,
                    )

                    # Propagate output block
                    input_block = subkernel_propagator.propagate(block_stripe_config)

                    input_block_shape = _Shape(input_block.shape, input_layout)
                    input_block_shape.round_up(self._input_micro_block)

                    output_block_shape = _Shape(output_block, output_layout)

                    if op_type == "ethosu_conv2d":
                        input_block_shape.depth = input_block_depth

                    # Banks required for input block
                    input_banks = self._get_input_banks(input_block_shape, input_bytewidth)
                    # Banks required for accumulation
                    acc_banks = self._get_accumulator_banks(output_block_shape, acc_bytewidth)

                    if (input_banks + acc_banks) <= banks_available:
                        output_cycles = self._get_output_cycles(
                            op_type, op_str, ifm_dtype, ofm_dtype, activation
                        )
                        output_cycles *= np.prod(output_block).tolist()
                        output_cycles = int(math.ceil(output_cycles))
                        compute_cycles = self._estimate_compute_cycles_per_block(
                            op_type,
                            output_block_shape,
                            input_block_shape,
                            kernel_h,
                            kernel_w,
                            ifm_channels,
                            "int8",
                            partkernel,
                        )
                        block_config = BlockConfig(
                            input_block_shape.as_list(), output_block, compute_cycles, output_cycles
                        )

                        if self._disable_block_culling:
                            # Block culling disabled - add all block configs that fit
                            valid_block_configs.append(block_config)
                        else:
                            # Add block config only if it's not dominated by an existing block.
                            # A block config is dominated by another if its output_shape is greater
                            # or equal in every dimension and strictly greater in at least one
                            # dimension.
                            dominated = False
                            for valid_block in valid_block_configs:
                                if block_config < valid_block:
                                    dominated = True
                                    break

                            if not dominated:
                                valid_block_configs.append(block_config)

                            # Every consecutive block in the innermost loop will be dominated by
                            # this one so break
                            break

        return valid_block_configs

    def _estimate_compute_cycles_per_block(
        self,
        op_type: str,
        block_shape: _Shape,
        input_block_shape: _Shape,
        kernel_h: int,
        kernel_w: int,
        input_channels: int,
        ifm_dtype: str,
        partkernel: bool = False,
    ) -> Tuple[int, int]:
        # Calculate the amount of micro blocks per block, per axis
        num_quantum_x = _round_up_div(block_shape.width, self._micro_block.width)
        num_quantum_y = _round_up_div(block_shape.height, self._micro_block.height)
        num_quantum_z = _round_up_div(block_shape.depth, self._micro_block.depth)
        num_quantum_xy = num_quantum_x * num_quantum_y

        kernel_steps = self.get_kernel_steps(op_type, kernel_h, kernel_w, ifm_dtype, partkernel)

        wd_cycles = self._get_weight_decoder_cycles(op_type)
        delay_cycles = self._get_delay_cycles(op_type, ifm_dtype)
        cycle_quantum = 4

        compute_cycles = 0
        for subkernel_steps in kernel_steps:
            subkernel_cycles = 1 if op_type == "ethosu_pooling" else subkernel_steps
            compute_cycles += (
                max(wd_cycles, cycle_quantum * num_quantum_xy) * subkernel_cycles * num_quantum_z
            )

            if num_quantum_xy == 1:
                if num_quantum_z == 1:
                    compute_cycles += delay_cycles * subkernel_steps
                elif subkernel_steps > 1:
                    compute_cycles += delay_cycles * (subkernel_steps - 1) * num_quantum_z

        if partkernel:
            compute_cycles *= _round_up_div(input_block_shape.depth, 8)

        if op_type == "ethosu_conv2d":
            compute_cycles *= _round_up_div(input_channels, input_block_shape.depth)

        return compute_cycles
