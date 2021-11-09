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
"""Relay identity operator for Arm(R) Ethos(TM)-U NPU"""

import tvm
from tvm.relay.op import _make
from tvm.topi.generic import schedule_injective
from tvm.relay.op.op import OpStrategy
from tvm.relay.op import strategy as _strategy

from ..te import identity_compute


@tvm.ir.register_op_attr("contrib.ethosu.identity", "FTVMCompute")
def create_ethosu_identity_compute(attrs, args, out_type):
    """Create an ethosu_identity compute op."""
    ifm = args[0]
    lut = args[1]
    ifm_scale = attrs.ifm_scale
    ifm_zero_point = attrs.ifm_zero_point
    ofm_scale = attrs.ofm_scale
    ofm_zero_point = attrs.ofm_zero_point
    activation = attrs.activation
    op = identity_compute(
        ifm, lut, ifm_scale, ifm_zero_point, ofm_scale, ofm_zero_point, activation
    )
    return [op]


@tvm.ir.register_op_attr("contrib.ethosu.identity", "FTVMStrategy")
def identity_strategy_ethosu(attrs, inputs, out_type, target):
    strategy = OpStrategy()
    strategy.add_implementation(
        create_ethosu_identity_compute,
        _strategy.wrap_topi_schedule(schedule_injective),
        name="ethosu_identity",
    )
    return strategy


def ethosu_identity(
    ifm: tvm.relay.Expr,
    lut: tvm.relay.Expr,
    ifm_scale: float = 1,
    ifm_zero_point: int = 0,
    ofm_scale: float = 1,
    ofm_zero_point: int = 0,
    activation: str = "NONE",
) -> tvm.relay.Call:
    """The Identity operator that runs on the NPU.

    This operator takes in a tensor of any shape and returns the same tensor,
    with the data optionally requantized.

    Parameters
    ----------
    ifm : tvm.relay.Expr
        The Input Feature Map tensor (IFM).
    lut : tvm.relay.Expr
        The look-up table values to use if activation = "LUT", "TANH" or "SIGMOID".
    ifm_scale : float
        The quantization scale for the Input Feature Map tensor.
    ifm_zero_point : int
        The quantization zero point for the Input Feature Map tensor.
    ofm_scale : float
        The quantization scale for the Output Feature Map tensor.
    ofm_zero_point : int
       The quantization zero point for the Output Feature Map tensor.
    activation : str, optional
        The activation function to use.
            "NONE" - no activation function.
            "TANH" - tanh activation function.
            "SIGMOID" - sigmoid activation function.
            "LUT" - use a look-up table to perform the activation function.

    Returns
    -------
    out : tvm.relay.Call
        A call to the ethosu_identity op.
    """
    return _make.ethosu_identity(
        ifm, lut, ifm_scale, ifm_zero_point, ofm_scale, ofm_zero_point, activation
    )
