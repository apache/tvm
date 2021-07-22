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

"""Pytest plugin for using tvm testing extensions.

TVM provides utilities for testing across all supported targets, and
to more easily parametrize across many inputs.  For more information
on usage of these features, see documentation in the tvm.testing
module.

These are enabled by default in all pytests provided by tvm, but may
be useful externally for one-off testing.  To enable, add the
following line to the test script, or to the conftest.py in the same
directory as the test scripts.

     pytest_plugins = ['tvm.testing.plugin']

"""

import pytest

import tvm.testing.utils


def pytest_configure(config):
    """Runs at pytest configure time, defines marks to be used later."""
    markers = {
        "gpu": "mark a test as requiring a gpu",
        "tensorcore": "mark a test as requiring a tensorcore",
        "cuda": "mark a test as requiring cuda",
        "opencl": "mark a test as requiring opencl",
        "rocm": "mark a test as requiring rocm",
        "vulkan": "mark a test as requiring vulkan",
        "metal": "mark a test as requiring metal",
        "llvm": "mark a test as requiring llvm",
    }
    for markername, desc in markers.items():
        config.addinivalue_line("markers", "{}: {}".format(markername, desc))

    print("enabled targets:", "; ".join(map(lambda x: x[0], tvm.testing.enabled_targets())))
    print("pytest marker:", config.option.markexpr)


def pytest_generate_tests(metafunc):
    """Called once per unit test, modifies/parametrizes it as needed."""
    tvm.testing.utils._auto_parametrize_target(metafunc)
    tvm.testing.utils._parametrize_correlated_parameters(metafunc)


def pytest_collection_modifyitems(config, items):
    """Called after all tests are chosen, currently used for bookkeeping."""
    # pylint: disable=unused-argument
    tvm.testing.utils._count_num_fixture_uses(items)
    tvm.testing.utils._remove_global_fixture_definitions(items)


@pytest.fixture
def dev(target):
    """Give access to the device to tests that need it."""
    return tvm.device(target)


def pytest_sessionfinish(session, exitstatus):
    # Don't exit with an error if we select a subset of tests that doesn't
    # include anything
    if session.config.option.markexpr != "":
        if exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED:
            session.exitstatus = pytest.ExitCode.OK
