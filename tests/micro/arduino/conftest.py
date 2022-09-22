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

pytest_plugins = [
    "tvm.micro.testing.pytest_plugin",
]

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--arduino-cli-cmd",
        default="arduino-cli",
        help="Path to `arduino-cli` command for flashing device.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "requires_hardware: mark test to run only when an Arduino board is connected"
    )


@pytest.fixture(scope="session")
def arduino_cli_cmd(request):
    return request.config.getoption("--arduino-cli-cmd")
