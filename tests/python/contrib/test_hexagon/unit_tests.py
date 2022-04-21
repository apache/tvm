import pytest
import numpy as np
from tvm.contrib.hexagon.build import HexagonLauncher
#import tvm.contrib.hexagon as hexagon
from .conftest import requires_hexagon_toolchain

@requires_hexagon_toolchain
def test_cache_read_write_2d(hexagon_session):
    func = hexagon_session._rpc.get_function("hexagon.run_all_tests")
    x = func()
    np.testing.assert_equal(x, 0)
