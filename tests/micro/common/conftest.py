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

import pytest

from ..zephyr.test_utils import ZEPHYR_BOARDS
from ..arduino.test_utils import ARDUINO_BOARDS


def pytest_addoption(parser):
    parser.addoption(
        "--platform",
        required=True,
        choices=["arduino", "zephyr"],
        help="Platform to run tests with",
    )
    parser.addoption(
        "--board",
        required=True,
        choices=list(ARDUINO_BOARDS.keys()) + list(ZEPHYR_BOARDS.keys()),
        help="microTVM boards for tests",
    )
    parser.addoption(
        "--test-build-only",
        action="store_true",
        help="Only run tests that don't require physical hardware.",
    )


@pytest.fixture
def platform(request):
    return request.config.getoption("--platform")


@pytest.fixture
def board(request):
    return request.config.getoption("--board")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--test-build-only"):
        skip_hardware_tests = pytest.mark.skip(reason="--test-build-only was passed")
        for item in items:
            if "requires_hardware" in item.keywords:
                item.add_marker(skip_hardware_tests)
