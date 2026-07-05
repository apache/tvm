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
"""Suite-level hardware gate for the tirx tests.

The tirx kernels and codegen paths target Blackwell (sm_100a) — they emit
PTX/SASS (tcgen05, tmem, cp.async ``.async`` modifiers, fp8 conversions, ...)
that ptxas/NVRTC reject for older targets, and many tests execute on the
device. Running the suite on a CPU-only node or a pre-sm_100 GPU therefore
fails at compile/run time rather than skipping. Gate the whole directory on a
real sm_100a device so it skips cleanly where the hardware is absent and runs
in full where it is present.
"""

from pathlib import Path

import pytest

from tvm.testing import env


def pytest_collection_modifyitems(config, items):
    if env.has_cuda_compute(10):
        return
    suite_root = Path(__file__).parent
    skip = pytest.mark.skip(
        reason="tirx suite requires a CUDA compute capability 10.0 (sm_100a) device"
    )
    for item in items:
        if item.path.is_relative_to(suite_root):
            item.add_marker(skip)
