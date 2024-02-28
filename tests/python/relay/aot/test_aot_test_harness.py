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

"""
Tests for the AOT test harness.
"""

import pytest
import numpy as np

import tvm
from tvm import relay
from tvm.testing.aot import AOTTestRunner, compile_and_run, AOTTestModel


def test_output_on_mismatch_option():
    """
    Test the print_output_on_mismatch option when there is a mismatch.
    """
    interface_api = "packed"
    use_unpacked_api = True
    test_runner = AOTTestRunner()
    dtype = "float32"

    two = relay.add(relay.const(1, dtype=dtype), relay.const(1, dtype=dtype))
    func = relay.Function([], two)
    outputs = {
        "output": np.array(
            [
                0,
            ]
        ).astype(dtype)
    }

    msg = ".*Actual, Reference\n2.000000, 0.000000\nAOT_TEST_FAILURE.*"
    with pytest.raises(RuntimeError, match=msg):
        compile_and_run(
            AOTTestModel(module=tvm.IRModule.from_expr(func), inputs={}, outputs=outputs),
            test_runner,
            interface_api,
            use_unpacked_api,
            print_output_on_mismatch=True,
        )


if __name__ == "__main__":
    tvm.testing.main()
