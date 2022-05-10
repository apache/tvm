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

import os
import pytest
import numpy as np
from tvm.contrib.hexagon.build import HexagonLauncher
from .conftest import requires_hexagon_toolchain


@requires_hexagon_toolchain
@pytest.mark.skipif(
    os.environ.get("USE_HEXAGON_GTEST") == None,
    reason="This test requires the USE_HEXAGON_GTEST to be specified with a path to a Hexagon gtest version normally located at /path/to/hexagon/sdk/utils/googletest/gtest",
)
def test_run_unit_tests(hexagon_session):
    # arguments to pass to gtest
    # e.g.
    # 1) to run all tests use:
    # gtest_args = ""
    # 2) to run all tests with "foo" in their name twice use:
    # gtest_args = "--gtest_repeat=2 --gtest_filter=*foo*"
    gtest_args = ""
    func = hexagon_session._rpc.get_function("hexagon.run_unit_tests")
    gtest_error_code_and_output = func(gtest_args)
    gtest_error_code = int(gtest_error_code_and_output.splitlines()[0])
    gtest_output = gtest_error_code_and_output.split("\n", 1)[-1]
    print(gtest_output)
    np.testing.assert_equal(gtest_error_code, 0)
