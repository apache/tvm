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
"""UMA backend for the UltraTrail accelerator"""

from ..backend import UMABackend
from .strategies import *
from .schedules import *
from .passes import *
from .patterns import *
from .codegen import gen_includes


class UltraTrailBackend(UMABackend):
    def __init__(self):
        super(UltraTrailBackend, self).__init__()

        #######################################################################
        # Relay to Relay function registration
        #######################################################################
        self._register_pattern("conv1d_relu", conv1d_relu_pattern())

        self._register_relay_pass(1, ConfigGenerator())
        self._register_relay_pass(2, BufferScopeAnnotator())

        #######################################################################
        # Relay to TIR function registration
        #######################################################################
        self._register_operator_strategy("nn.conv1d", custom_conv1d_strategy, plevel=9)

        self._register_tir_schedule(insert_extern_calls)

        self._register_tir_pass(0, CodegenGenerateConfig())
        self._register_tir_pass(0, CodegenGenerateConstants())

        #######################################################################
        # TIR to runtime function registration
        #######################################################################
        self._register_codegen(
            fmt="c", includes=gen_includes, constants=None, replace_call_extern=None
        )

    @property
    def target_name(self):
        return "ultra_trail"
