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

import os
import subprocess
import sys

import pytest

import tvm
import tvm.testing


def test_check_if_device_exists():
    """kExist can be checked when no devices are present

    This test uses `CUDA_VISIBLE_DEVICES` to disable any CUDA-capable
    GPUs from being accessed by the subprocess.  Within the
    subprocess, the CUDA driver cannot be initialized.  While most
    functionality of CUDADeviceAPI would raise an exception, the
    `kExist` property can still be checked.

    """

    cmd = [
        sys.executable,
        "-c",
        "import tvm; tvm.device('cuda').exist",
    ]
    subprocess.check_call(
        cmd,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
        },
    )


@pytest.mark.gpu
@pytest.mark.skipif(
    tvm.get_global_func("device_api.vulkan", allow_missing=True) is None,
    reason="Vulkan runtime is not built",
)
@pytest.mark.parametrize("allocate_tensor", [False, True])
def test_vulkan_process_exit(allocate_tensor):
    """Device initialization and allocation must allow a clean process exit."""
    # Initialize Vulkan only in the child: exit-time failures cannot be caught
    # by an in-process assertion and must not crash the pytest process itself.
    script = """
import sys
import numpy as np
import tvm

dev = tvm.vulkan(0)
if not dev.exist:
    sys.exit(77)
if int(sys.argv[1]):
    expected = np.arange(128, dtype="float32")
    tensor = tvm.runtime.tensor(expected, dev)
    np.testing.assert_array_equal(tensor.numpy(), expected)
print("Vulkan work completed", flush=True)
"""
    proc = subprocess.run(
        [sys.executable, "-c", script, str(int(allocate_tensor))],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=120,
        check=False,
    )
    if proc.returncode == 77:
        pytest.skip("No Vulkan device is available")
    assert proc.returncode == 0, f"Vulkan subprocess exited with {proc.returncode}:\n{proc.stdout}"
    assert "Vulkan work completed" in proc.stdout


if __name__ == "__main__":
    tvm.testing.main()
