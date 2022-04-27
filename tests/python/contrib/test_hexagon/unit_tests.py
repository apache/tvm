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
    func = hexagon_session._rpc.get_function("hexagon.run_all_tests")
    result = func(gtest_args)
    np.testing.assert_equal(result, 0)
