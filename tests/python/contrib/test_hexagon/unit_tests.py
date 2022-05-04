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

import pytest
import numpy as np
from tvm.contrib.hexagon.build import HexagonLauncher
from .conftest import requires_hexagon_toolchain


@requires_hexagon_toolchain
def test_cache_read_write_2d(hexagon_session):
    # arguments to pass to gtest
    # e.g.
    # 1) to run all tests use:
    # gtest_args = ""
    # 2) to run all tests with "foo" in their name twice use:
    # gtest_args = "--gtest_repeat=2 --gtest_filter=*foo*"
    gtest_args = ""
    try:
        func = hexagon_session._rpc.get_function("hexagon.run_all_tests")
        result = func(gtest_args)
    except:
        print(
            "This test requires the USE_HEXAGON_GTEST cmake flag to be specified with a path to a Hexagon gtest version normally located at /path/to/hexagon/sdk/utils/googletest/gtest"
        )
        result = 1

    np.testing.assert_equal(result, 0)
