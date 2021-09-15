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

import argparse

ARM_ISA_MAP = {
    "armv7e-m": ["SMLAD", "SSUB8", "SEL"],
}


class IsaAnalyzer(object):
    def __init__(self, target):
        self.target = target

        parser = argparse.ArgumentParser()
        parser.add_argument("-mcpu", type=str)
        parser.add_argument("-march", type=str)
        args, _ = parser.parse_known_args(str(target).split())

        self._isa_map = ARM_ISA_MAP[args.march] if args.march in ARM_ISA_MAP else []


    def __contains__(self, instruction):
        return instruction in self._isa_map