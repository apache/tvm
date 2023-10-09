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
"""Common relax pass instrumentation across IR variants."""
import tvm
from tvm import relax


@tvm.instrument.pass_instrument
class WellFormedInstrument:
    """An instrument that checks the input/output IRModule of the Pass
    is well formed. It will skip specific passes, like Normalize.
    """

    def __init__(self, check_struct_info=True):
        self.skip_pass_name = ["Normalize", "ResolveGlobals"]
        self.check_struct_info = check_struct_info

    def run_before_pass(self, mod, pass_info):
        self._check(mod, pass_info.name, "Before")

    def run_after_pass(self, mod, pass_info):
        self._check(mod, pass_info.name, "After")

    def _check(self, mod, pass_name, name_prefix):
        if pass_name not in self.skip_pass_name:
            is_well_formed = relax.analysis.well_formed(mod, self.check_struct_info)
            if not is_well_formed:
                mod.show(name=f"{name_prefix}{pass_name}")
            assert is_well_formed
