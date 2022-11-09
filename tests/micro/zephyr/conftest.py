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
        "--west-cmd", default="west", help="Path to `west` command for flashing device."
    )
    parser.addoption(
        "--use-fvp",
        action="store_true",
        default=False,
        help="If set true, use the FVP emulator to run the test",
    )
    parser.addoption(
        "--serial",
        default=None,
        help="If set true, use the FVP emulator to run the test",
    )


@pytest.fixture(scope="session")
def west_cmd(request):
    return request.config.getoption("--west-cmd")


@pytest.fixture
def use_fvp(request):
    return request.config.getoption("--use-fvp")

@pytest.fixture
def serial(request):
    return request.config.getoption("--serial")

@pytest.fixture(autouse=True)
def xfail_on_fvp(request, use_fvp):
    """mark the tests as xfail if running on fvp."""
    if request.node.get_closest_marker("xfail_on_fvp"):
        if use_fvp:
            request.node.add_marker(
                pytest.mark.xfail(reason="checking corstone300 reliability on CI")
            )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "xfail_on_fvp(): mark test as xfail on fvp",
    )
