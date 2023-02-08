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
# pylint: disable=invalid-name, exec-used
"""
Environment declaration. Contains Gemminis hardware parameters.
=====================
**Author**: `Federico Peccia <https://fPecc.github.io/>`_
"""

from __future__ import absolute_import as _abs
import re
from typing import List, Tuple, Dict, Callable
from .intrin import (
    gemm,
    gemm_cisc,
    conv2d_cisc,
    dw_conv2d_cisc,
    add_tensorize,
    add_mvout_tensorize,
)
from .utils import COUNTERS


class Environment(object):
    """Hardware configuration object.

    This object contains all the information
    needed for compiling to a specific Gemmini backend.

    """

    _instance = None

    @classmethod
    def init_overwrite(
        cls,
        batch=1,
        dim=32,
        max_bytes=64,
        inp_dtype="int8",
        wgt_dtype="int8",
        acc_dtype="int32",
        acc_rows=4096,
        bank_rows=8192,
        bank_num=4,
        debug=False,
        enabled_counters: Dict = None,
        supports_non_zero_padding: bool = False,
        use_experimental_qnn_add: bool = False,
    ):
        """Overwrites the init function

        Args:
            batch (int, optional): Batch size. Defaults to 1.
            dim (int, optional): Gemminis systolic array dimensions (DIM). Defaults to 32.
            max_bytes (int, optional): Limits maximum amount of mvin columns. Defaults to 64.
            inp_dtype (str, optional): Type of the Gemmini scratchpad. Defaults to "int8".
            wgt_dtype (str, optional): Type of the Gemmini weight scratchpad. Defaults to "int8".
            acc_dtype (str, optional): Type of the Gemmini accumulator. Defaults to "int32".
            acc_rows (int, optional): Rows of the accumulator. Defaults to 4096.
            bank_rows (int, optional): Rows of each bank in the scratchpad. Defaults to 8192.
            bank_num (int, optional): Banks for the scratchpad. Defaults to 4.
            debug (bool, optional): Adds debug of Gemmini counters. Defaults to False.
            enabled_counters (dict, optional): Enabled Gemmini counters for debug purposes.
                Defaults to None.
            supports_non_zero_padding (bool, optional): Gemmini supports instructions
                with non-zero padding. Defaults to False.
            use_experimental_qnn_add (bool, optional): Pattern matching for qnn.add.
                Defaults to False.
        """
        inst = Environment.instance()
        inst.init(
            batch=batch,
            dim=dim,
            max_bytes=max_bytes,
            inp_dtype=inp_dtype,
            wgt_dtype=wgt_dtype,
            acc_dtype=acc_dtype,
            acc_rows=acc_rows,
            bank_rows=bank_rows,
            bank_num=bank_num,
            debug=debug,
            enabled_counters=enabled_counters,
            supports_non_zero_padding=supports_non_zero_padding,
            use_experimental_qnn_add=use_experimental_qnn_add,
        )

    @classmethod
    def instance(cls):
        """Returns the current instance

        Returns:
            _type_: _description_
        """
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.init()
        return cls._instance

    def init(
        self,
        batch=1,
        dim=16,
        max_bytes=64,
        inp_dtype="int8",
        wgt_dtype="int8",
        acc_dtype="int32",
        acc_rows=1024,
        bank_rows=4096,
        bank_num=4,
        debug=False,
        enabled_counters: Dict = None,
        supports_non_zero_padding: bool = False,
        use_experimental_qnn_add: bool = False,
    ):
        """_summary_

        Args:
            batch (int, optional): Batch size. Defaults to 1.
            dim (int, optional): Gemminis systolic array dimensions (DIM). Defaults to 32.
            max_bytes (int, optional): Limits maximum amount of mvin columns. Defaults to 64.
            inp_dtype (str, optional): Type of the Gemmini scratchpad. Defaults to "int8".
            wgt_dtype (str, optional): Type of the Gemmini "logical" weight scratchpad.
                Defaults to "int8".
            acc_dtype (str, optional): Type of the Gemmini accumulator. Defaults to "int32".
            acc_rows (int, optional): Amount of rows of the accumulator. Defaults to 4096.
            bank_rows (int, optional): Amount of rows of each bank in the scratchpad.
                Defaults to 8192.
            bank_num (int, optional): Amount of banks for the scratchpad. Defaults to 4.
            debug (bool, optional): Adds debug of Gemmini counters. Defaults to False.
            enabled_counters (dict, optional): Enabled Gemmini counters for debug purposes.
                Defaults to None.
            supports_non_zero_padding (bool, optional): Gemmini supports instructions
                with non-zero padding. Defaults to False.
            use_experimental_qnn_add (bool, optional): Pattern matching for qnn.add.
                Defaults to False.
        """

        assert batch == 1, "Only batch size of 1 is currently supported"
        self.debug = debug

        self.BATCH = batch
        self.DIM = dim
        self.MAX_BYTES = max_bytes

        self.inp_dtype = inp_dtype
        self.wgt_dtype = wgt_dtype
        self.acc_dtype = acc_dtype

        self.inp_bits = int(
            re.match(r"((float)|(int)|(uint))(?P<width_bits>[0-9]+)", self.inp_dtype).group(
                "width_bits"
            )
        )
        self.wgt_bits = int(
            re.match(r"((float)|(int)|(uint))(?P<width_bits>[0-9]+)", self.wgt_dtype).group(
                "width_bits"
            )
        )
        self.acc_bits = int(
            re.match(r"((float)|(int)|(uint))(?P<width_bits>[0-9]+)", self.acc_dtype).group(
                "width_bits"
            )
        )

        self.size_elem = int(self.inp_bits / 8)
        self.size_acc = int(self.acc_bits / 8)

        self.ACC_ROWS = acc_rows
        self.BANK_ROWS = bank_rows
        self.BANK_NUM = bank_num

        self.WGT_SCR_BASE_ADDRESS = int(self.BANK_ROWS * self.BANK_NUM * 2 / 4)
        self.WGT_SCR_ROWS = self.BANK_ROWS * self.BANK_NUM - self.WGT_SCR_BASE_ADDRESS
        self.INP_SCR_BASE_ADDRESS = 0
        self.INP_SCR_ROWS = self.WGT_SCR_BASE_ADDRESS
        self.OUT_ACC_BASE_ADDRESS = 0xC0000000

        self.MAX_BLOCK_LEN = int(self.MAX_BYTES / self.DIM)
        if self.DIM * self.size_acc <= self.MAX_BYTES:
            self.MAX_BLOCK_LEN_ACC = int(self.MAX_BYTES / (self.DIM * self.size_acc))
        else:
            self.MAX_BLOCK_LEN_ACC = 1

        self.scr_scope = "local.scratchpad"
        self.acc_scope = "local.accumulator"
        # Actually, only one scratchpad should exist.
        # But we do this logical partition to correctly manage the pointers
        # to the buffers stored in this memories.
        # Should see how we can fix this in the future.
        self.scr_wgt_scope = "local.scratchpad_weight"

        self.A_mvin = "A_mvin"
        self.B_mvin = "B_mvin"
        self.D_mvin = "D_mvin"
        self.C_mvin = "C_mvin"
        self.C_mvin_accum = "C_mvin_accum"
        self.C_mvout = "C_mvout"
        self.C_mvout_acc_dtype = "C_mvout_acc_dtype"

        self.WEIGHT_STATIONARY = 1
        self.OUTPUT_STATIONARY = 0

        self.mvin_scale_identity = 1.0
        self.max_matrix = 64

        self.supports_non_zero_padding = supports_non_zero_padding
        self.use_experimental_qnn_add = use_experimental_qnn_add

        self.enabled_counters = enabled_counters if enabled_counters is not None else COUNTERS
        # Check that all enabled counters exist in the actual counters from Gemmini
        for key, value in self.enabled_counters.items():
            assert (
                value == COUNTERS[key]
            ), f"Enabled counter with key {key} does not exist \
            or has a different name in the actual counters dict!"

    def gemm(
        self,
        I: int,
        K: int,
        J: int,
        stride: int = 1,
        is_depthwise_conv2d: bool = False,
        mode: int = 1,
        accum_patch=None,
    ) -> Callable:
        """Wrapper to expose the gemm intrinsic

        Args:
            I (int): output first axis dimension
            K (int): reduction axis dimension
            J (int): output second axis dimension
            stride (int, optional): Stride, useful for convolutions. Defaults to 1.
            is_depthwise_conv2d (bool, optional): Flag to explain if this is a
                GEMM for a depthwise convolution. Defaults to False.
            mode (int, optional): Systolic array mode (WS=1,OS=0). Defaults to 1.
            accum_patch (_type_, optional): Var of the reduction axis loop. Defaults to None.

        Returns:
            Callable: gemm instrinsic
        """
        return gemm(self, I, K, J, stride, is_depthwise_conv2d, mode, accum_patch)

    def gemm_cisc(
        self,
        inp_shape: Tuple[int, ...],
        wgt_shape: Tuple[int, ...],
        bias_shape: Tuple[int, ...],
        scale: float,
        matmul_type: int,
    ) -> Callable:
        """Wrapper to expose the gemm_cisc intrinsic

        Args:
            inp_shape (Tuple[int,...]): Input feature map shape
            wgt_shape (Tuple[int,...]): Weights shape
            bias_shape (Tuple[int,...]): Bias shape
            scale (float): Output scaling factor
            matmul_type (int): Systolic array mode (WS=1,OS=0)

        Returns:
            Callable: gemm cisc intrinsic
        """
        return gemm_cisc(self, inp_shape, wgt_shape, bias_shape, scale, matmul_type)

    def conv2d_cisc(
        self,
        inp_shape: Tuple[int, ...],
        wgt_shape: Tuple[int, ...],
        bias_shape: Tuple[int, ...],
        out_shape: Tuple[int, ...],
        strides: int,
        padding: List[int],
        padding_value: int,
        activation: int,
        scale: float,
        pool_size: List[int],
        pool_strides: List[int],
        pool_dilation: List[int],
        pool_padding: List[int],
    ) -> Callable:
        """Wrapper to expose the conv2d_cisc intrinsic

        Args:
            inp_shape (Tuple[int,...]): Input feature map shape
            wgt_shape (Tuple[int,...]): Weights shape
            bias_shape (Tuple[int,...]): Bias shape
            out_shape (Tuple[int,...]): Output feature map shape
            strides (int): Convolution stride
            padding (List[int]): Pixels to pad in each direction
            padding_value (int): Value to use for padding
            activation (int): Has activation?
            scale (float): Output scaling factor
            pool_size (List[int]): Size of the output pooling window
            pool_strides (List[int]): Strides for the output pooling window
            pool_dilation (List[int]): Dilation for the output pooling window
            pool_padding (List[int]): Padding for the output pooling

        Returns:
            Callable: conv2d cisc intrinsic
        """
        return conv2d_cisc(
            self,
            inp_shape,
            wgt_shape,
            bias_shape,
            out_shape,
            strides,
            padding,
            padding_value,
            activation,
            scale,
            pool_size,
            pool_strides,
            pool_dilation,
            pool_padding,
        )

    def dw_conv2d_cisc(
        self,
        inp_shape: Tuple[int, ...],
        wgt_shape: Tuple[int, ...],
        bias_shape: Tuple[int, ...],
        out_shape: Tuple[int, ...],
        strides: int,
        padding: List[int],
        padding_value: int,
        activation: int,
        scale: float,
    ) -> Callable:
        """Wrapper to expose the dw_conv2d_cisc intrinsic

        Args:
            inp_shape (Tuple[int,...]): Input feature map shape
            wgt_shape (Tuple[int,...]): Weights shape
            bias_shape (Tuple[int,...]): Bias shape
            out_shape (Tuple[int,...]): Output feature map shape
            strides (int): Convolution stride
            padding (List[int]): Pixels to pad in each direction
            padding_value (int): Value to use for padding
            activation (int): Has activation?
            scale (float): Output scaling factor

        Returns:
            Callable: dw conv2d cisc intrinsic
        """
        return dw_conv2d_cisc(
            self,
            inp_shape,
            wgt_shape,
            bias_shape,
            out_shape,
            strides,
            padding,
            padding_value,
            activation,
            scale,
        )

    def add_tensorize(self, oshape: Tuple[int, ...]) -> Callable:
        """Wrapper to expose the add_tensorize intrinsic

        Args:
            oshape (Tuple[int,...]): Output feature map shape

        Returns:
            Callable: add intrinsic
        """
        return add_tensorize(self, oshape)

    def add_mvout_tensorize(self, oshape: Tuple[int, ...]) -> Callable:
        """Wrapper to expose the add_mvout_tensorize intrinsic

        Args:
            oshape (Tuple[int,...]): Output feature map shape

        Returns:
            Callable: add mvout intrinsic
        """
        return add_mvout_tensorize(self, oshape)
