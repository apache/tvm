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
"""Tests for tvm.support.libinfo."""

import tvm
import tvm.testing
from tvm.contrib.cutlass import has_cutlass


def test_libinfo_use_cutlass_matches_codegen_registration():
    """USE_CUTLASS mirrors whether the native CUTLASS codegen is compiled in.

    Regression test: ``libinfo()`` used to hardcode USE_CUTLASS to "OFF", so
    ``build_flag_enabled("USE_CUTLASS")`` always reported False -- even on
    builds configured with ``USE_CUTLASS=ON`` -- silently skipping the
    CUTLASS test suite. ``has_cutlass()`` probes the same "relax.ext.cutlass"
    global function that ``libinfo()`` now checks; that function is
    registered by ``src/relax/backend/contrib/cutlass/codegen.cc``, which
    CMake only compiles under ``USE_CUTLASS=ON`` (see
    ``cmake/modules/contrib/CUTLASS.cmake``), so its presence at runtime is a
    direct, always-accurate signal of the flag -- unlike an environment
    variable that nothing in the build sets automatically.
    """
    expected = "ON" if has_cutlass() else "OFF"
    assert tvm.support.libinfo()["USE_CUTLASS"] == expected


if __name__ == "__main__":
    tvm.testing.main()
