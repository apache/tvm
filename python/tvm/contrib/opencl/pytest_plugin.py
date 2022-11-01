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

# pylint: disable=invalid-name,redefined-outer-name
""" OpenCL testing fixtures used to deduce testing argument
    values from testing parameters """

import pytest


def pytest_addoption(parser):
    """Add pytest options."""

    parser.addoption("--gtest_args", action="store", default="")


def pytest_generate_tests(metafunc):
    option_value = metafunc.config.option.gtest_args
    if "gtest_args" in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("gtest_args", [option_value])
