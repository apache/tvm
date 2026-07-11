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
"""Hardware requirements for CUDA tile-primitive tests."""

from pathlib import Path

import pytest

from tvm.testing import env


def pytest_collection_modifyitems(items):
    if env.has_cuda_compute(10):
        return
    suite_root = Path(__file__).resolve().parent
    skip = pytest.mark.skip(reason="requires a CUDA compute capability 10.0 device")
    for item in items:
        if Path(item.path).resolve().is_relative_to(suite_root):
            item.add_marker(skip)
