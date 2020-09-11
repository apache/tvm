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
"""Defines functions to analyze available opcodes in the ARM ISA."""


ARM_ISA_MAP = {
    "armv7e-m": ["SMLAD"],
}


class IsaAnalyzer(object):
    def __init__(self, target):
        self.target = target
        # TODO: actually parse -mcpu
        arch = "armv7e-m"
        self._isa_map = ARM_ISA_MAP[arch]

    def __contains__(self, instruction):
        return instruction in self._isa_map
