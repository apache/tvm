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
"""UMA testcase for the vanilla_accelerator accelerator"""
import pytest

import tvm
from tvm import tir
from tvm.relay.dataflow_pattern import is_op, wildcard
from tvm.relay.backend.contrib.uma.api.utils import PassPhase
from tvm.relay.backend.contrib.uma.backend import UMABackend
from tvm.relay.backend.contrib.uma._template.passes import MyAiHwConv2dPass as VanillaAcceleratorConv2dPass
from tvm.relay.backend.contrib.uma._template.codegen import gen_includes

from tvm.relay.backend.contrib.uma._template.patterns import conv2d_pattern

# def conv2d_pattern():
#     pattern = is_op("nn.conv2d")(wildcard(), wildcard())
#     pattern = pattern.has_attr({"strides": [1, 1]})
#     return pattern


class VanillaAcceleratorBackend(UMABackend):
    """UMA backend for the VanillaAccelerator accelerator."""

    def __init__(self):
        super().__init__()

        #######################################################################
        # Target configuration
        #######################################################################
        #self._register_target_attr("dimension")

        #######################################################################
        # Relay to Relay function registration
        #######################################################################
        self._register_pattern("conv2d", conv2d_pattern())

        #######################################################################
        # Relay to TIR function registration
        #######################################################################
        self._register_tir_pass(PassPhase.TIR_PHASE_0, VanillaAcceleratorConv2dPass())


        #######################################################################
        # TIR to runtime function registration
        #######################################################################
        self._register_codegen(
            fmt="c", includes=gen_includes
        )

    @property
    def target_name(self):
        return "vanilla_accelerator"
