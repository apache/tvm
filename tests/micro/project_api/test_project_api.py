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

import sys
import numpy as np

import tvm
from tvm import relay
from tvm.micro.project_api import server
from tvm.relay.backend import Runtime
from tvm.micro.testing import get_target

from .utils import build_project_api

API_GENERATE_PROJECT = "generate_project"
API_BUILD = "build"
API_FLASH = "flash"
API_OPEN_TRANSPORT = "open_transport"

PLATFORM_ARDUINO = "arduino"
PLATFORM_ZEPHYR = "zephyr"


platform = tvm.testing.parameter(PLATFORM_ARDUINO, PLATFORM_ZEPHYR)


@tvm.testing.requires_micro
def test_default_options_exist(platform):
    sys.path.insert(0, tvm.micro.get_microtvm_template_projects(platform))
    import microtvm_api_server

    platform_options = microtvm_api_server.PROJECT_OPTIONS
    default_options = server.default_project_options()

    option_names = []
    for option in platform_options:
        option_names.append(option.name)

    for option in default_options:
        assert option.name in option_names


@tvm.testing.requires_micro
def test_project_minimal_options(platform):
    """Test template project with minimum projectOptions"""
    build_project_api(platform)


if __name__ == "__main__":
    tvm.testing.main()
