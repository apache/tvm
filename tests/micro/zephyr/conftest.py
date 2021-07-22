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

import tvm.target.target

# The models that should pass this configuration. Maps a short, identifying platform string to
# (model, zephyr_board).
PLATFORMS = {
    "host": ("host", "qemu_x86"),
    "host_riscv32": ("host", "qemu_riscv32"),
    "host_riscv64": ("host", "qemu_riscv64"),
    "mps2_an521": ("mps2_an521", "mps2_an521-qemu"),
    "nrf5340dk": ("nrf5340dk", "nrf5340dk_nrf5340_cpuapp"),
    "stm32f746xx_disco": ("stm32f746xx", "stm32f746g_disco"),
    "stm32f746xx_nucleo": ("stm32f746xx", "nucleo_f746zg"),
    "stm32l4r5zi_nucleo": ("stm32l4r5zi", "nucleo_l4r5zi"),
    "zynq_mp_r5": ("zynq_mp_r5", "qemu_cortex_r5"),
}


def pytest_addoption(parser):
    parser.addoption(
        "--microtvm-platforms",
        default="host",
        choices=PLATFORMS.keys(),
        help=(
            "Specify a comma-separated list of test models (i.e. as passed to tvm.target.micro()) "
            "for microTVM tests."
        ),
    )
    parser.addoption(
        "--west-cmd", default="west", help="Path to `west` command for flashing device."
    )
    parser.addoption(
        "--skip-build",
        action="store_true",
        help="If set true, reuses build from the previous test run. Otherwise, build from the scratch.",
    )
    parser.addoption(
        "--tvm-debug",
        action="store_true",
        default=False,
        help="If set true, enable a debug session while the test is running. Before running the test, in a separate shell, you should run: <python -m tvm.exec.microtvm_debug_shell>",
    )


def pytest_generate_tests(metafunc):
    if "platform" in metafunc.fixturenames:
        metafunc.parametrize("platform", metafunc.config.getoption("microtvm_platforms").split(","))


@pytest.fixture
def west_cmd(request):
    return request.config.getoption("--west-cmd")


@pytest.fixture
def skip_build(request):
    return request.config.getoption("--skip-build")


@pytest.fixture
def tvm_debug(request):
    return request.config.getoption("--tvm-debug")
