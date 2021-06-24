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
from pytest import ExitCode

import tvm
import tvm.testing


def pytest_configure(config):
    print("enabled targets:", "; ".join(map(lambda x: x[0], tvm.testing.enabled_targets())))
    print("pytest marker:", config.option.markexpr)


@pytest.fixture
def dev(target):
    return tvm.device(target)


def pytest_generate_tests(metafunc):
    tvm.testing._auto_parametrize_target(metafunc)
    tvm.testing._parametrize_correlated_parameters(metafunc)


def pytest_collection_modifyitems(config, items):
    tvm.testing._count_num_fixture_uses(items)
    tvm.testing._remove_global_fixture_definitions(items)


def pytest_sessionfinish(session, exitstatus):
    # Don't exit with an error if we select a subset of tests that doesn't
    # include anything
    if session.config.option.markexpr != "":
        if exitstatus == ExitCode.NO_TESTS_COLLECTED:
            session.exitstatus = ExitCode.OK
