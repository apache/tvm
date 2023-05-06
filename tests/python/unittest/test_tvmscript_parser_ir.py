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
"""Unittests for tvm.script.parser.ir"""

import pytest
import inspect
import tvm.testing
from tvm.script.parser import ir_module
from tvm.ir import IRModule


def test_ir_base():
    @ir_module
    class BlankIRModule:
        pass

    assert isinstance(BlankIRModule, IRModule) and len(BlankIRModule.functions.items()) == 0
    assert BlankIRModule.__name__ == "BlankIRModule"


if __name__ == "__main__":
    tvm.testing.main()
