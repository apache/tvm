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
"""Relay partitioner for the UltraTrail accelerator"""

from .passes import *
from .patterns import *
from ..partitioner import GenericPartitioner


class UltraTrailPartitioner(GenericPartitioner):
    @property
    def target_name(self):
        return "ultra_trail"

    def _register_patterns(self):
        self._register_pattern("conv1d_relu", conv1d_relu_pattern())
    
    def _register_relay_passes(self):
        self._register_relay_pass(1, ConfigGenerator())
        self._register_relay_pass(2, BufferScopeAnnotator())
