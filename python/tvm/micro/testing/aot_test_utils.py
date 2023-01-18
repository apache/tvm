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

"""
This file provides utilities for running AOT tests, especially for Corstone.

"""

import logging
import itertools
import shutil

import pytest

import tvm
from tvm.testing.aot import AOTTestRunner

pytest.importorskip("tvm.micro")

_LOG = logging.getLogger(__name__)


AOT_DEFAULT_RUNNER = AOTTestRunner()

# AOT Test Runner using the Arm® Corstone™-300 Reference Systems
# see: https://developer.arm.com/ip-products/subsystem/corstone/corstone-300
AOT_CORSTONE300_RUNNER = AOTTestRunner(
    makefile="corstone300",
    prologue="""
    UartStdOutInit();
    """,
    includes=["uart_stdout.h"],
    pass_config={
        "relay.ext.cmsisnn.options": {
            "mcpu": "cortex-m55",
        }
    },
)

AOT_USMP_CORSTONE300_RUNNER = AOTTestRunner(
    makefile="corstone300",
    prologue="""
    UartStdOutInit();
    """,
    includes=["uart_stdout.h"],
    pass_config={
        "relay.ext.cmsisnn.options": {
            "mcpu": "cortex-m55",
        },
        "tir.usmp.enable": True,
    },
)


def parametrize_aot_options(test):
    """Parametrize over valid option combinations"""

    requires_arm_eabi = pytest.mark.skipif(
        shutil.which("arm-none-eabi-gcc") is None, reason="ARM embedded toolchain unavailable"
    )

    interface_api = ["packed", "c"]
    use_unpacked_api = [True, False]
    test_runner = [AOT_DEFAULT_RUNNER, AOT_CORSTONE300_RUNNER]

    all_combinations = itertools.product(interface_api, use_unpacked_api, test_runner)

    # Filter out packed operators with c interface
    valid_combinations = filter(
        lambda parameters: not (parameters[0] == "c" and not parameters[1]),
        all_combinations,
    )

    # Only use reference system for C interface and unpacked API calls
    valid_combinations = filter(
        lambda parameters: not (
            parameters[2] == AOT_CORSTONE300_RUNNER
            and (parameters[0] == "packed" or not parameters[1])
        ),
        valid_combinations,
    )

    # Skip reference system tests if running in i386 container
    marked_combinations = map(
        lambda parameters: pytest.param(*parameters, marks=[requires_arm_eabi])
        if parameters[2] == AOT_CORSTONE300_RUNNER
        else parameters,
        valid_combinations,
    )

    func = pytest.mark.parametrize(
        ["interface_api", "use_unpacked_api", "test_runner"],
        marked_combinations,
    )(test)

    return tvm.testing.skip_if_32bit(reason="Reference system unavailable in i386 container")(func)
