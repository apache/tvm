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

import tvm
from tvm.micro.project_api import server

API_GENERATE_PROJECT = "generate_project"
API_BUILD = "build"
API_FLASH = "flash"
API_OPEN_TRANSPORT = "open_transport"

PLATFORM_ARDUINO = "arduino"
PLATFORM_ZEPHYR = "zephyr"


platform = tvm.testing.parameter(PLATFORM_ARDUINO, PLATFORM_ZEPHYR)


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


def test_default_options_requirements(platform):
    sys.path.insert(0, tvm.micro.get_microtvm_template_projects(platform))
    import microtvm_api_server

    platform_options = microtvm_api_server.PROJECT_OPTIONS
    for option in platform_options:
        if option.name == "verbose":
            assert option.optional == [API_GENERATE_PROJECT]
        if option.name == "project_type":
            option.required == [API_GENERATE_PROJECT]
        if option.name == "board":
            option.required == [API_GENERATE_PROJECT]
            if platform == PLATFORM_ARDUINO:
                option.optional == [API_FLASH, API_OPEN_TRANSPORT]
        if option.name == "cmsis_path":
            assert option.optional == [API_GENERATE_PROJECT]
        if option.name == "compile_definitions":
            assert option.optional == [API_GENERATE_PROJECT]
        if option.name == "extra_files_tar":
            assert option.optional == [API_GENERATE_PROJECT]


def test_extra_options_requirements(platform):
    sys.path.insert(0, tvm.micro.get_microtvm_template_projects(platform))
    import microtvm_api_server

    platform_options = microtvm_api_server.PROJECT_OPTIONS

    if platform == PLATFORM_ZEPHYR:
        for option in platform_options:
            if option.name == "gdbserver_port":
                assert option.optional == [API_OPEN_TRANSPORT]
            if option.name == "nrfjprog_snr":
                assert option.optional == [API_OPEN_TRANSPORT]
            if option.name == "openocd_serial":
                assert option.optional == [API_OPEN_TRANSPORT]
            if option.name == "west_cmd":
                assert option.optional == [API_GENERATE_PROJECT]
            if option.name == "config_main_stack_size":
                assert option.optional == [API_GENERATE_PROJECT]
            if option.name == "arm_fvp_path":
                assert option.optional == [API_GENERATE_PROJECT]
            if option.name == "use_fvp":
                assert option.optional == [API_GENERATE_PROJECT]
            if option.name == "heap_size_bytes":
                assert option.optional == [API_GENERATE_PROJECT]

    if platform == PLATFORM_ARDUINO:
        for option in platform_options:
            if option.name == "port":
                assert option.optional == [API_FLASH, API_OPEN_TRANSPORT]


if __name__ == "__main__":
    tvm.testing.main()
