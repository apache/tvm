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


def pytest_addoption(parser):
    parser.addoption(
        "--microtvm-platforms",
        default="host",
        help=(
            "Specify a comma-separated list of test models (i.e. as passed to tvm.target.micro()) "
            "for microTVM tests."
        ),
    )
    parser.addoption(
        "--west-cmd", default="west", help="Path to `west` command for flashing device."
    )


def pytest_generate_tests(metafunc):
    if "platform" in metafunc.fixturenames:
        metafunc.parametrize("platform", metafunc.config.getoption("microtvm_platforms").split(","))


@pytest.fixture
def west_cmd(request):
    return request.config.getoption("--west-cmd")
