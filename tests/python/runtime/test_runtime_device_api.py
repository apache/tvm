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


if __name__ == "__main__":
    tvm.testing.main()
