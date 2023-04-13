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

    def __init__(self):
        self.skip_pass_name = ["Normalize", "ResolveGlobals"]

    def run_before_pass(self, mod, pass_info):
        if pass_info.name not in self.skip_pass_name:
            assert relax.analysis.well_formed(mod)

    def run_after_pass(self, mod, pass_info):
        if pass_info.name not in self.skip_pass_name:
            assert relax.analysis.well_formed(mod)
