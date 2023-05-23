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
# pylint: disable=no-else-return, unidiomatic-typecheck, invalid-name, W0611, C0413

"""Vitis-AI runtime test for CPU only part

This test verifies as much as possible whether the a model can be correctly offloaded
and executed for Vitis-AI acceleration. This entails:
    - Annotating and partitioning model for Vitis-AI acceleration
    - Building a Vitis-AI PyXIR runtime module with on-the-fly quantization enabled
    - Run first iteration of on-the-fly quantization flow. This will always be run
      on CPU as the first N (parameter) will be used for collecting calibration data
      for quantization.

NOTE This is not a full end-to-end test as we need the full Vitis-AI docker environment
and access to an FPGA instance for that. This test verifies the Vitis-AI flow as much as
possible without requiring access to dedicated docker environment and/or hardware setup.
NOTE Quantization is not being tested (we need to be inside Vitis-AI docker environment
for that) buth the internal representation used for quantization is being generated and
functionally tested (CPU).
"""

import sys
import numpy as np

import pytest

pytest.importorskip("pyxir")
import pyxir.contrib.target.DPUCADF8H
import pyxir.contrib.target.DPUCVDX8H
import pyxir.contrib.target.DPUCZDX8G

import tvm
import tvm.relay.testing
from tvm import relay
from tvm.testing import requires_vitis_ai

from .infrastructure import verify_result


@requires_vitis_ai
@pytest.mark.parametrize("dpu_target", ["DPUCADF8H", "DPUCVDX8H", "DPUCZDX8G-zcu104"])
def test_extern_vitis_ai_resnet18(dpu_target):
    """Test first part of Vitis AI on-the-fly quantization runtime with ResNet 18 model"""

    dtype = "float32"
    ishape = (1, 3, 224, 224)
    mod, params = relay.testing.resnet.get_workload(num_layers=18, batch_size=1)
    ref_mod, params = relay.testing.resnet.get_workload(num_layers=18, batch_size=1)

    i_data = np.random.uniform(0, 1, ishape).astype(dtype)

    ref_res = relay.create_executor("graph", mod=ref_mod, device=tvm.cpu(0)).evaluate()(
        i_data, **params
    )

    verify_result(
        mod,
        {"data": i_data},
        (1, 1000),
        ref_res.numpy(),
        tol=1e-5,
        params=params,
        dpu_target=dpu_target,
        tvm_ops=7,
    )


if __name__ == "__main__":
    if sys.platform == "win32":
        print("Skip test on Windows for now")
        sys.exit(0)
    tvm.testing.main()
